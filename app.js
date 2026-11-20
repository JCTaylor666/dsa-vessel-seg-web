// app.js — UI: load an image, hand its gray values to the worker, draw the result, export PNGs.
// All pixels stay in this tab; the only network traffic is the model download (and ORT's own runtime files).

const DEFAULT_WEIGHTS = 'https://huggingface.co/cjy666/dsa-vessel-seg/resolve/main/';
const params = new URLSearchParams(location.search);
const WEIGHTS = (params.get('weights') || DEFAULT_WEIGHTS).replace(/\/?$/, '/');
const REPO_PAGE = WEIGHTS.replace(/\/resolve\/[^/]+\/$/, '');

const $ = (id) => document.getElementById(id);
const el = {
  modelInfo: $('modelInfo'), drop: $('drop'), file: $('file'), imgInfo: $('imgInfo'),
  backend: $('backend'), folds: $('folds'), foldInfo: $('foldInfo'), mirror: $('mirror'), run: $('run'),
  progressWrap: $('progressWrap'), bar: $('bar'), status: $('status'), opacity: $('opacity'),
  dlMask: $('dlMask'), dlProb: $('dlProb'), result: $('result'),
  stage: $('stage'), empty: $('empty'), view: $('view'), weightsLink: $('weightsLink'),
  cacheInfo: $('cacheInfo'), clearCache: $('clearCache'),
};
const MODEL_CACHE = /^dsa-vessel-seg-models/;   // every cache name the worker has ever used
let cacheState = { files: 0, bytes: 0 }, disarmTimer = 0;

let manifest = null;          // model.json from the weights repo
let worker = null, workerBackend = null, loadedKeys = new Set(), runtimeLabel = '';
let image = null;             // { name, w, h, gray: Uint8Array }
let prob = null;              // Uint8Array w*h (0..255), >= 128 = vessel
let busy = false, msgId = 0;
const waiting = new Map();    // message id -> { resolve, reject, onTile, onProgress }

const fmtMB = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0) + ' MB';
const fmtS = (ms) => (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + ' s';

function setStatus(text, isError = false) {
  el.status.textContent = text;
  el.status.classList.toggle('err', isError);
}
function setProgress(frac) {
  if (frac == null) { el.progressWrap.hidden = true; return; }
  el.progressWrap.hidden = false;
  el.bar.style.width = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + '%';
}
const selectedFolds = () => manifest ? manifest.folds.filter((f) => el.folds.querySelector(`input[value="${f.fold}"]`).checked) : [];
const keyOf = (f) => `fold${f.fold}`;

function refreshButtons() {
  el.run.disabled = busy || !image || !manifest || selectedFolds().length === 0;
  el.dlMask.disabled = el.dlProb.disabled = busy || !prob;
  el.folds.querySelectorAll('input').forEach((i) => { i.disabled = busy; });
  el.clearCache.disabled = busy || !cacheState.files;
}
function refreshFoldInfo() {
  if (!manifest) return;
  const sel = selectedFolds();
  const toFetch = sel.filter((f) => !loadedKeys.has(keyOf(f)));
  el.foldInfo.textContent = !sel.length ? 'Pick at least one fold.'
    : `${sel.length === 1 ? '1 fold' : sel.length + ' folds averaged'} · ` +
      (toFetch.length ? `${fmtMB(toFetch.reduce((s, f) => s + f.size, 0))} to load (cached after the first download)` : 'loaded');
  refreshButtons();
}

// ---------- model manifest + worker ----------

async function loadManifest() {
  try {
    const res = await fetch(WEIGHTS + 'model.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    manifest = await res.json();
    el.modelInfo.textContent = `${manifest.name} · ${manifest.folds.length} folds · ${manifest.patch[0]}×${manifest.patch[1]} patches`;
    el.weightsLink.href = REPO_PAGE;
    el.folds.innerHTML = '';
    for (const f of manifest.folds) {
      const l = document.createElement('label');
      l.innerHTML = `<input type="checkbox" value="${f.fold}" checked> ${f.fold}`;
      el.folds.appendChild(l);
    }
    el.folds.addEventListener('change', refreshFoldInfo);
    refreshFoldInfo();
  } catch (e) {
    el.modelInfo.textContent = 'Model info unavailable';
    setStatus(`Could not read ${WEIGHTS}model.json (${e.message}).`, true);
  }
  refreshButtons();
}

function ensureWorker() {
  const backend = el.backend.value;
  if (worker && workerBackend === backend) return;
  if (worker) worker.terminate();
  loadedKeys = new Set();
  worker = new Worker('./worker.js', { type: 'module' });
  workerBackend = backend;
  worker.onmessage = (e) => {
    const m = e.data, w = waiting.get(m.id);
    if (!w) return;
    if (m.type === 'progress') w.onProgress && w.onProgress(m);
    else if (m.type === 'tile') w.onTile && w.onTile(m);
    else if (m.type === 'error') { waiting.delete(m.id); w.reject(new Error(m.message)); }
    else { waiting.delete(m.id); w.resolve(m); }
  };
  worker.onerror = (e) => {
    const err = new Error(e.message || 'The inference worker failed to start');
    for (const w of waiting.values()) w.reject(err);
    waiting.clear(); worker = null; workerBackend = null; loadedKeys = new Set();
  };
}

function ask(msg, hooks = {}, transfer = []) {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject, ...hooks });
    worker.postMessage({ ...msg, id }, transfer);
  });
}

async function ensureLoaded(folds) {
  ensureWorker();
  const models = folds.map((f) => ({ key: keyOf(f), url: WEIGHTS + f.file, size: f.size, sha256: f.sha256 }));
  const r = await ask({ type: 'load', models, patch: manifest.patch, prefer: el.backend.value }, {
    onProgress: (m) => {
      const what = `fold ${m.key.replace('fold', '')} (${m.index}/${m.count})`;
      if (m.phase === 'download') { setStatus(`Downloading ${what}… ${fmtMB(m.loaded)} / ${fmtMB(m.total)}`); setProgress(m.total ? m.loaded / m.total : null); }
      else if (m.phase === 'cache') setStatus(`Loading ${what} from browser cache…`);
      else if (m.phase === 'compile') setStatus(`Preparing ${what} on this device…`);
    },
  });
  loadedKeys = new Set(r.loaded);
  runtimeLabel = r.ep === 'webgpu' ? 'WebGPU (GPU)' : `CPU / WebAssembly, ${r.threads} thread${r.threads > 1 ? 's' : ''}`;
  el.modelInfo.textContent = `${manifest.name} · running on ${runtimeLabel}`;
  refreshFoldInfo();
}

async function segment() {
  if (!image || busy) return;
  const folds = selectedFolds();
  if (!folds.length) return;
  busy = true; refreshButtons();
  try {
    setStatus('Starting…'); setProgress(0);
    await ensureLoaded(folds);
    const gray = image.gray.slice();   // the worker gets its own copy
    const res = await ask({ type: 'run', gray, w: image.w, h: image.h, mirror: el.mirror.checked, keys: folds.map(keyOf) }, {
      onTile: (m) => { setStatus(`Segmenting… model run ${m.done} / ${m.total} · ${fmtS(m.ms)}`); setProgress(m.done / m.total); },
    }, [gray.buffer]);
    prob = res.prob;
    let fg = 0; for (let i = 0; i < prob.length; i++) if (prob[i] >= 128) fg++;
    el.result.textContent = `${fg.toLocaleString()} vessel pixels (${(100 * fg / prob.length).toFixed(1)} %) · ` +
      `${folds.length} fold${folds.length > 1 ? 's' : ''}${el.mirror.checked ? ' + mirroring' : ''} · ${res.runs} model runs in ${fmtS(res.ms)} on ${runtimeLabel}`;
    setStatus('Done.'); setProgress(null);
    if (document.querySelector('input[name=mode]:checked').value === 'image') document.querySelector('input[name=mode][value=overlay]').checked = true;
    draw();
  } catch (e) {
    setStatus(e.message, true); setProgress(null);
  } finally {
    busy = false; refreshButtons();
    refreshCacheInfo();
  }
}

// ---------- downloaded-model storage ----------
// The worker keeps each downloaded fold in the Cache API (per browser, per site). This section shows what is
// stored and deletes it on request; nothing else on the page writes to browser storage.

async function cacheSummary() {
  if (!('caches' in self)) return null;
  let files = 0, bytes = 0;
  for (const name of await caches.keys()) {
    if (!MODEL_CACHE.test(name)) continue;
    const cache = await caches.open(name);
    for (const req of await cache.keys()) {
      const res = await cache.match(req);
      if (!res) continue;
      files++;
      bytes += Number(res.headers.get('x-model-bytes')) || (await res.blob()).size;
    }
  }
  return { files, bytes };
}

async function refreshCacheInfo() {
  try {
    const s = await cacheSummary();
    if (!s) {
      cacheState = { files: 0, bytes: 0 };
      el.cacheInfo.textContent = 'This browser has no cache storage, so models are not kept between visits.';
    } else {
      cacheState = s;
      el.cacheInfo.textContent = s.files
        ? `${s.files} model file${s.files > 1 ? 's' : ''} stored in this browser · ${fmtMB(s.bytes)}`
        : 'No model files stored in this browser.';
    }
  } catch (e) {
    el.cacheInfo.textContent = `Could not read this browser's storage (${e.message}).`;
  }
  refreshButtons();
}

function disarmClear() {
  clearTimeout(disarmTimer);
  el.clearCache.classList.remove('armed');
  el.clearCache.textContent = 'Delete downloaded models';
}

async function clearCache() {
  if (busy || !cacheState.files) return;
  if (!el.clearCache.classList.contains('armed')) {   // first click only arms the button
    el.clearCache.classList.add('armed');
    el.clearCache.textContent = `Click again to delete ${fmtMB(cacheState.bytes)}`;
    disarmTimer = setTimeout(disarmClear, 5000);
    return;
  }
  disarmClear();
  busy = true; refreshButtons();
  try {
    const before = await cacheSummary();
    for (const name of await caches.keys()) if (MODEL_CACHE.test(name)) await caches.delete(name);
    // the models loaded in memory go too, so the next run really downloads again
    if (worker) { worker.terminate(); worker = null; workerBackend = null; }
    loadedKeys = new Set();
    el.modelInfo.textContent = manifest ? `${manifest.name} · ${manifest.folds.length} folds · ${manifest.patch[0]}×${manifest.patch[1]} patches` : el.modelInfo.textContent;
    setStatus(`Deleted ${before.files} model file${before.files === 1 ? '' : 's'} (${fmtMB(before.bytes)}). The next run downloads them again.`);
  } catch (e) {
    setStatus(`Could not delete the stored models (${e.message}).`, true);
  } finally {
    busy = false;
    refreshFoldInfo();
    await refreshCacheInfo();
  }
}

// ---------- image input ----------

async function loadBlob(blob, name = blob.name || 'image') {
  if (busy) return;
  let bitmap;
  try { bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }); }
  catch (e) { setStatus(`Could not decode ${name}: ${e.message}`, true); return; }
  const w = bitmap.width, h = bitmap.height;
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const gray = new Uint8Array(w * h);
  let colour = false;
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    if (rgba[j] !== rgba[j + 1] || rgba[j + 1] !== rgba[j + 2]) colour = true;
    gray[i] = rgba[j];
  }
  if (colour) {   // the model has one input channel; a real colour image is reduced to luminance
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = Math.round(0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2]);
  }
  image = { name, w, h, gray };
  prob = null;
  el.imgInfo.textContent = `${name} · ${w} × ${h}${colour ? ' · colour image converted to gray' : ''}`;
  el.result.textContent = '';
  el.empty.hidden = true; el.view.hidden = false;
  document.querySelector('input[name=mode][value=image]').checked = true;
  if (manifest) setStatus('Ready — press Segment.');
  draw(); refreshButtons();
}

// ---------- drawing + export ----------

function draw() {
  if (!image) return;
  const { w, h, gray } = image;
  const mode = document.querySelector('input[name=mode]:checked').value;
  el.view.width = w; el.view.height = h;
  const ctx = el.view.getContext('2d');
  const out = ctx.createImageData(w, h);
  const d = out.data, a = el.opacity.value / 100;
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    let r = gray[i], g = gray[i], b = gray[i];
    if (prob && mode === 'mask') { r = g = b = prob[i] >= 128 ? 255 : 0; }
    else if (prob && mode === 'prob') { r = g = b = prob[i]; }
    else if (prob && mode === 'overlay' && prob[i] >= 128) {
      r = r * (1 - a) + 255 * a; g = g * (1 - a) + 64 * a; b = b * (1 - a) + 64 * a;
    }
    d[j] = r; d[j + 1] = g; d[j + 2] = b; d[j + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

async function downloadPng(kind) {
  if (!prob || !image) return;
  const { w, h } = image;
  const c = new OffscreenCanvas(w, h), ctx = c.getContext('2d');
  const out = ctx.createImageData(w, h), d = out.data;
  for (let i = 0, j = 0; i < prob.length; i++, j += 4) {
    const v = kind === 'mask' ? (prob[i] >= 128 ? 255 : 0) : prob[i];
    d[j] = d[j + 1] = d[j + 2] = v; d[j + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  const blob = await c.convertToBlob({ type: 'image/png' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = image.name.replace(/\.[^.]+$/, '') + (kind === 'mask' ? '_vessel_mask.png' : '_vessel_prob.png');
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

// ---------- wiring ----------

el.file.addEventListener('change', () => { if (el.file.files[0]) loadBlob(el.file.files[0]); el.file.value = ''; });
el.drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); } });
['dragenter', 'dragover'].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); el.stage.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((t) => document.addEventListener(t, (e) => { e.preventDefault(); if (t === 'drop' || e.target === document.documentElement) el.stage.classList.remove('dragging'); }));
document.addEventListener('drop', (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) loadBlob(f); });
document.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData ? e.clipboardData.items : [])].find((i) => i.type.startsWith('image/'));
  if (item) loadBlob(item.getAsFile(), 'pasted.png');
});
el.run.addEventListener('click', segment);
el.backend.addEventListener('change', () => {
  if (worker && !busy) { worker.terminate(); worker = null; workerBackend = null; loadedKeys = new Set(); refreshFoldInfo(); }
});
el.opacity.addEventListener('input', draw);
document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', draw));
el.dlMask.addEventListener('click', () => downloadPng('mask'));
el.dlProb.addEventListener('click', () => downloadPng('prob'));
el.clearCache.addEventListener('click', clearCache);
el.clearCache.addEventListener('blur', disarmClear);

// Small hook for automated checks: window.__vesselSeg.loadBlob(blob) / .segment() / .getProb()
window.__vesselSeg = { loadBlob, segment, getProb: () => prob, isBusy: () => busy };

if (!('gpu' in navigator)) el.backend.querySelector('option[value=webgpu]').disabled = true;
loadManifest();
refreshCacheInfo();
