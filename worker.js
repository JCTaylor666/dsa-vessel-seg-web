// worker.js — model loading, file decoding / preprocessing, nnU-Net-style inference and PNG encoding,
// all off the main thread. The page sends one input file at a time and writes the PNGs this worker returns.
//
// Inference mirrors nnU-Net v2's predictor for this 2D model:
//   1. crop to the bounding box of non-zero pixels       (nnunetv2 crop_to_nonzero)
//   2. z-score over the cropped image                    (ZScoreNormalization, use_mask_for_norm = False)
//   3. pad to at least the patch size, split evenly      (acvl_utils pad_nd_image, value 0)
//   4. tiles at step 0.5, Gaussian-weighted (sigma = patch/8)   (compute_steps_for_sliding_window / compute_gaussian)
//   5. logits averaged over the selected folds and, optionally, the 4 mirror flips (nnU-Net averages logits)
//   6. softmax -> vessel probability; argmax = probability > 0.5; pasted back into the bounding box
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs';

// Same release tag as the page, so a cached old module is never mixed with a new one. These are loaded without
// top-level await: a message that arrives while a module worker is still awaiting is dropped, so the handler at
// the bottom is installed right away and waits for libsReady itself.
const V = new URL(self.location.href).searchParams.get('v') || '0';
let decodeDicom, windowTo8bit, minOverFrames, positiveBox, cropGray, encodeGrayPng;
const libsReady = Promise.all([import(`./dicom.js?v=${V}`), import(`./png.js?v=${V}`)]).then(([d, p]) => {
  ({ decodeDicom, windowTo8bit, minOverFrames, positiveBox, cropGray } = d);
  ({ encodeGrayPng } = p);
});

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
// Threads need cross-origin isolation (COOP/COEP headers), which GitHub Pages does not send.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;

const CACHE = 'dsa-vessel-seg-models-v1';
const sessions = new Map();     // model key -> { session, input, output }
let ep = null;                  // execution provider chosen for the first session; every later one uses it
let PH = 896, PW = 768;
let cancelled = false;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

// ---------------------------------------------------------------- models

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
  if (cache) {
    try { await cache.put(m.url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream', 'x-model-bytes': String(bytes.byteLength) } })); }
    catch (e) { /* quota: just don't cache */ }
  }
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

// ---------------------------------------------------------------- inference

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

// gray: Uint8Array w*h -> vessel probability Uint8Array w*h (0..255, >= 128 = vessel)
async function segmentGray(gray, w, h, mirror, models, onRun) {
  const prob = new Uint8Array(w * h);

  // 1. bounding box of non-zero pixels
  let y0 = h, y1 = -1, x0 = w, x1 = -1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) if (gray[row + x] !== 0) {
      if (y < y0) y0 = y; if (y > y1) y1 = y; if (x < x0) x0 = x; if (x > x1) x1 = x;
    }
  }
  if (y1 < 0) return prob;
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
        if (cancelled) throw new Error('cancelled');
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
        onRun(++done, total);
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
  return prob;
}

// ---------------------------------------------------------------- input files

async function bitmapToGray(bitmap) {
  const w = bitmap.width, h = bitmap.height;
  const c = new OffscreenCanvas(w, h), ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, w, h).data, gray = new Uint8Array(w * h);
  let colour = false;
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    if (rgba[j] !== rgba[j + 1] || rgba[j + 1] !== rgba[j + 2]) colour = true;
    gray[i] = rgba[j];
  }
  if (colour) {   // PIL "L" conversion: ITU-R 601-2 luma
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = Math.round((rgba[j] * 299 + rgba[j + 1] * 587 + rgba[j + 2] * 114) / 1000);
  }
  return { gray, w, h, colour };
}

const decodeOpts = {
  // JPEG Baseline frames (not used by the Zurich data) are decoded by the browser's own JPEG decoder.
  jpegBaselineDecode: async (bytes) => (await bitmapToGray(await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }), { colorSpaceConversion: 'none' }))).gray,
};

// One input file -> list of 8-bit images to segment, all the same size, plus a description for the log.
async function prepare(file, kind, crop, withMinip) {
  if (kind === 'png') {
    const bmp = await createImageBitmap(file, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const { gray, w, h, colour } = await bitmapToGray(bmp);
    const box = crop ? positiveBox(gray, w, h) : null;
    const img = box ? cropGray(gray, w, box) : gray;
    const ow = box ? box[3] - box[2] : w, oh = box ? box[1] - box[0] : h;
    return { w: ow, h: oh, srcW: w, srcH: h, box, note: colour ? 'colour PNG converted to gray' : '', items: [{ name: 'f_1', gray: img }] };
  }
  const d = await decodeDicom(await file.arrayBuffer(), decodeOpts);
  const W = d.cols, H = d.rows;
  const grays = d.frames.map(f => windowTo8bit(f, d.wc, d.ww));
  const minip = windowTo8bit(minOverFrames(d.frames), d.wc, d.ww);
  d.frames.length = 0;                                   // raw pixels no longer needed
  // one crop box per DICOM: the box of its MinIP, i.e. of the pixels that are > 0 in every frame
  const box = crop ? positiveBox(minip, W, H) : null;
  const cut = g => (box ? cropGray(g, W, box) : g);
  const items = grays.map((g, i) => ({ name: `f_${i + 1}`, gray: cut(g) }));
  if (withMinip && grays.length > 1) items.push({ name: 'minip', gray: cut(minip) });
  const missing = d.info.nFramesHeader - grays.length;
  return {
    w: box ? box[3] - box[2] : W, h: box ? box[1] - box[0] : H, srcW: W, srcH: H, box,
    note: `${d.tsName}${missing > 0 ? ` · ${missing} unreadable frame${missing > 1 ? 's' : ''} at the end skipped` : ''}`,
    items,
  };
}

async function processFile({ id, file, kind, crop, minip, mirror, keys }) {
  cancelled = false;
  const models = keys.map(k => sessions.get(k));
  if (!models.length || models.some(m => !m)) throw new Error('Model not loaded yet');
  const t0 = performance.now();
  const p = await prepare(file, kind, crop, minip);
  post({ type: 'fileinfo', id, count: p.items.length, w: p.w, h: p.h, srcW: p.srcW, srcH: p.srcH, box: p.box, note: p.note });
  for (let i = 0; i < p.items.length; i++) {
    const it = p.items[i];
    const prob = await segmentGray(it.gray, p.w, p.h, mirror, models,
      (done, total) => post({ type: 'run', id, index: i, count: p.items.length, done, total }));
    const mask = new Uint8Array(prob.length);
    let fg = 0;
    for (let k = 0; k < prob.length; k++) if (prob[k] >= 128) { mask[k] = 255; fg++; }
    const imagePng = await encodeGrayPng(it.gray, p.w, p.h);
    const maskPng = await encodeGrayPng(mask, p.w, p.h);
    post({ type: 'frame', id, name: it.name, index: i, count: p.items.length, w: p.w, h: p.h, fg, imagePng, maskPng, gray: it.gray, prob },
      [imagePng.buffer, maskPng.buffer, it.gray.buffer, prob.buffer]);
    p.items[i] = null;
  }
  post({ type: 'done', id, ms: performance.now() - t0 });
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'cancel') { cancelled = true; return; }
  try {
    await libsReady;
    if (m.type === 'load') await load(m);
    else if (m.type === 'process') await processFile(m);
  } catch (err) {
    const msg = (err && err.message) || String(err);
    post({ type: msg === 'cancelled' ? 'cancelled' : 'error', id: m.id, message: msg });
  }
};
