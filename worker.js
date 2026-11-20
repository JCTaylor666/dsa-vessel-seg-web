// worker.js — loads the ONNX fold models and runs nnU-Net-style sliding-window inference off the main thread.
//
// The steps mirror nnU-Net v2's predictor for this 2D model, so the browser result matches the cluster
// pipeline:
//   1. crop to the bounding box of non-zero pixels       (nnunetv2 crop_to_nonzero)
//   2. z-score over the cropped image                    (ZScoreNormalization, use_mask_for_norm = False)
//   3. pad to at least the patch size, split evenly      (acvl_utils pad_nd_image, value 0)
//   4. tiles at step 0.5, Gaussian-weighted (sigma = patch/8)   (compute_steps_for_sliding_window / compute_gaussian)
//   5. logits averaged over the selected folds and, optionally, the 4 mirror flips (nnU-Net averages logits)
//   6. softmax -> vessel probability; argmax = probability > 0.5; pasted back into the bounding box
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
// Threads need cross-origin isolation (COOP/COEP headers), which GitHub Pages does not send.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

const CACHE = 'dsa-vessel-seg-models-v1';
const sessions = new Map();     // model key -> { session, input, output }
let ep = null;                  // execution provider chosen for the first session; every later one uses it
let PH = 896, PW = 768;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

async function sha256Hex(bytes) {
  const d = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(d), b => b.toString(16).padStart(2, '0')).join('');
}

// Download once, then serve from the Cache API. A cached copy that fails its checksum is dropped and refetched.
async function fetchModel(m, report) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch (e) { cache = null; }
  if (cache) {
    try {
      const hit = await cache.match(m.url);
      if (hit) {
        const b = new Uint8Array(await hit.arrayBuffer());
        report(b.byteLength, 'cache');
        if (!m.sha256 || (await sha256Hex(b)) === m.sha256) return b;
        await cache.delete(m.url);
      }
    } catch (e) { /* fall through to the network */ }
  }
  const res = await fetch(m.url, { mode: 'cors' });
  if (!res.ok) throw new Error(`Model download failed: HTTP ${res.status} from ${m.url}`);
  const reader = res.body.getReader();
  const chunks = []; let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    report(loaded, 'download');
  }
  const bytes = new Uint8Array(loaded); let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  if (m.sha256) {
    const got = await sha256Hex(bytes);
    if (got !== m.sha256) throw new Error(`Checksum mismatch for ${m.key} (got ${got.slice(0, 12)}…, expected ${m.sha256.slice(0, 12)}…)`);
  }
  if (cache) { try { await cache.put(m.url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) { /* quota: just don't cache */ } }
  return bytes;
}

async function webgpuUsable() {
  if (!('gpu' in navigator)) return false;
  try { return !!(await navigator.gpu.requestAdapter()); } catch (e) { return false; }
}

async function createSession(bytes, prefer) {
  const order = ep ? [ep] : prefer === 'wasm' ? ['wasm'] : prefer === 'webgpu' ? ['webgpu'] : ['webgpu', 'wasm'];
  const errors = [];
  for (const e of order) {
    if (e === 'webgpu' && !(await webgpuUsable())) { errors.push('webgpu: not available in this browser'); continue; }
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: [e], graphOptimizationLevel: 'all' });
      ep = e;
      return s;
    } catch (err) { errors.push(`${e}: ${(err && err.message) || err}`); }
  }
  throw new Error('Could not start the model. ' + errors.join(' | '));
}

// Make sure every requested model has a live session. Sessions already loaded are kept.
async function load({ id, models, patch, prefer }) {
  [PH, PW] = patch;
  const todo = models.filter(m => !sessions.has(m.key));
  const total = todo.reduce((s, m) => s + (m.size || 0), 0);
  let before = 0;
  const t0 = performance.now();
  for (let i = 0; i < todo.length; i++) {
    const m = todo[i];
    let lastPost = 0;
    const bytes = await fetchModel(m, (loaded, phase) => {
      const now = performance.now();
      if (phase === 'cache' || now - lastPost > 100) {
        post({ type: 'progress', id, phase, key: m.key, index: i + 1, count: todo.length, loaded: before + loaded, total });
        lastPost = now;
      }
    });
    post({ type: 'progress', id, phase: 'compile', key: m.key, index: i + 1, count: todo.length, loaded: before + bytes.byteLength, total });
    const session = await createSession(bytes, prefer);
    sessions.set(m.key, { session, input: session.inputNames[0], output: session.outputNames[0] });
    before += m.size || bytes.byteLength;
  }
  post({ type: 'ready', id, ep, ms: performance.now() - t0, threads: ort.env.wasm.numThreads, loaded: [...sessions.keys()] });
}

function steps(imageSize, tile, stepFrac = 0.5) {
  const target = tile * stepFrac;
  const n = Math.ceil((imageSize - tile) / target) + 1;
  if (n <= 1) return [0];
  const actual = (imageSize - tile) / (n - 1);
  return Array.from({ length: n }, (_, i) => Math.round(actual * i));
}

let gauss = null;
function gaussian() {
  if (gauss && gauss.length === PH * PW) return gauss;
  const gy = new Float32Array(PH), gx = new Float32Array(PW);
  const sy = PH / 8, sx = PW / 8, cy = PH >> 1, cx = PW >> 1;
  for (let y = 0; y < PH; y++) gy[y] = Math.exp(-((y - cy) ** 2) / (2 * sy * sy));
  for (let x = 0; x < PW; x++) gx[x] = Math.exp(-((x - cx) ** 2) / (2 * sx * sx));
  gauss = new Float32Array(PH * PW);
  for (let y = 0; y < PH; y++) for (let x = 0; x < PW; x++) gauss[y * PW + x] = gy[y] * gx[x];
  return gauss;
}

async function run({ id, gray, w, h, mirror, keys }) {
  const models = keys.map(k => sessions.get(k));
  if (!models.length || models.some(m => !m)) throw new Error('Model not loaded yet');
  const t0 = performance.now();
  const prob = new Uint8Array(w * h);

  // 1. bounding box of non-zero pixels
  let y0 = h, y1 = -1, x0 = w, x1 = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) if (gray[row + x] !== 0) {
      if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x;
    }
  }
  if (y1 < 0) { post({ type: 'result', id, prob, w, h, ms: 0, runs: 0 }, [prob.buffer]); return; }
  const ch = y1 - y0 + 1, cw = x1 - x0 + 1, n = ch * cw;

  // 2. z-score over the crop
  let sum = 0, sq = 0;
  for (let y = 0; y < ch; y++) { const row = (y + y0) * w + x0; for (let x = 0; x < cw; x++) { const v = gray[row + x]; sum += v; sq += v * v; } }
  const mean = sum / n, sd = Math.max(Math.sqrt(Math.max(0, sq / n - mean * mean)), 1e-8);

  // 3. pad to >= patch, centred (the extra pixel goes after)
  const Hp = Math.max(ch, PH), Wp = Math.max(cw, PW);
  const pt = (Hp - ch) >> 1, pl = (Wp - cw) >> 1;
  const img = new Float32Array(Hp * Wp);
  for (let y = 0; y < ch; y++) {
    const src = (y + y0) * w + x0, dst = (y + pt) * Wp + pl;
    for (let x = 0; x < cw; x++) img[dst + x] = (gray[src + x] - mean) / sd;
  }

  // 4-5. Gaussian-weighted sliding window; logits averaged over folds x flips (linear, so per tile is exact)
  const ys = steps(Hp, PH), xs = steps(Wp, PW);
  const flips = mirror ? [[0, 0], [0, 1], [1, 0], [1, 1]] : [[0, 0]];
  const G = gaussian(), P = PH * PW;
  const acc = new Float32Array(Hp * Wp), wsum = new Float32Array(Hp * Wp);
  const diff = new Float32Array(P), tile = new Float32Array(P);
  const total = ys.length * xs.length * flips.length * models.length;
  let done = 0;
  for (const ty of ys) for (const tx of xs) {
    diff.fill(0);
    for (const [fy, fx] of flips) {
      for (let y = 0; y < PH; y++) {
        const sy = fy ? PH - 1 - y : y, src = (ty + sy) * Wp + tx, dst = y * PW;
        if (fx) for (let x = 0; x < PW; x++) tile[dst + x] = img[src + PW - 1 - x];
        else tile.set(img.subarray(src, src + PW), dst);
      }
      for (const m of models) {
        const out = await m.session.run({ [m.input]: new ort.Tensor('float32', tile, [1, 1, PH, PW]) });
        const L = out[m.output].data;               // [1, 2, PH, PW]: background, vessel
        for (let y = 0; y < PH; y++) {
          const dy = fy ? PH - 1 - y : y;
          for (let x = 0; x < PW; x++) {
            const i = y * PW + x, dx = fx ? PW - 1 - x : x;
            diff[dy * PW + dx] += L[P + i] - L[i];
          }
        }
        if (out[m.output].dispose) out[m.output].dispose();
        done++;
        post({ type: 'tile', id, done, total, ms: performance.now() - t0 });
      }
    }
    const k = 1 / (flips.length * models.length);
    for (let y = 0; y < PH; y++) {
      const row = (ty + y) * Wp + tx, grow = y * PW;
      for (let x = 0; x < PW; x++) { const g = G[grow + x]; acc[row + x] += diff[grow + x] * k * g; wsum[row + x] += g; }
    }
  }

  // 6. two-class softmax = sigmoid of the logit difference; paste back into the bbox
  for (let y = 0; y < ch; y++) {
    const src = (y + pt) * Wp + pl, dst = (y + y0) * w + x0;
    for (let x = 0; x < cw; x++) {
      const d = acc[src + x] / wsum[src + x];
      const p = Math.round(255 / (1 + Math.exp(-d)));
      // keep the 0.5 boundary exact: prob >= 128 <=> logit difference > 0 (argmax picks vessel)
      prob[dst + x] = d > 0 ? Math.max(128, p) : Math.min(127, p);
    }
  }
  post({ type: 'result', id, prob, w, h, ms: performance.now() - t0, runs: total }, [prob.buffer]);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') await load(m);
    else if (m.type === 'run') await run(m);
  } catch (err) {
    post({ type: 'error', id: m.id, message: (err && err.message) || String(err) });
  }
};
