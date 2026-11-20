// app.js — UI and batch driver: pick an input folder, find its DICOM / PNG files, send them one by one to the
// worker, and write the PNGs it returns into a new output folder (one subfolder per input file).
// Pixels never leave this machine; the only network traffic is the model download and ORT's runtime files.

const DEFAULT_WEIGHTS = 'https://huggingface.co/cjy666/dsa-vessel-seg/resolve/main/';
const params = new URLSearchParams(location.search);
const WEIGHTS = (params.get('weights') || DEFAULT_WEIGHTS).replace(/\/?$/, '/');
const REPO_PAGE = WEIGHTS.replace(/\/resolve\/[^/]+\/$/, '');
const OUT_PREFIX = 'vessel_seg_output_';            // our output folders; skipped when scanning an input folder
const MODEL_CACHE = /^dsa-vessel-seg-models/;        // every cache name the worker has ever used

const $ = (id) => document.getElementById(id);
const el = {
  modelInfo: $('modelInfo'), unsupported: $('unsupported'),
  openFolder: $('openFolder'), inputInfo: $('inputInfo'), fileList: $('fileList'),
  crop: $('crop'), minip: $('minip'),
  backend: $('backend'), folds: $('folds'), foldInfo: $('foldInfo'), mirror: $('mirror'),
  outInfo: $('outInfo'), chooseOut: $('chooseOut'), run: $('run'), cancel: $('cancel'),
  progressWrap: $('progressWrap'), bar: $('bar'), status: $('status'),
  opacity: $('opacity'), previewInfo: $('previewInfo'),
  cacheInfo: $('cacheInfo'), clearCache: $('clearCache'),
  stage: $('stage'), empty: $('empty'), view: $('view'), weightsLink: $('weightsLink'),
};

let manifest = null;
let worker = null, workerBackend = null, loadedKeys = new Set(), runtimeLabel = '';
let msgId = 0;
const waiting = new Map();          // message id -> { resolve, reject, on }
let inputDir = null, outParent = null, entries = [];
let busy = false, cancelRequested = false;
let preview = null;                 // { w, h, gray, prob }
let cacheState = { files: 0, bytes: 0 }, disarmTimer = 0;

const fmtMB = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0) + ' MB';
const fmtS = (ms) => ms < 60000 ? (ms / 1000).toFixed(ms < 10000 ? 1 : 0) + ' s' : `${Math.floor(ms / 60000)} min ${Math.round((ms % 60000) / 1000)} s`;
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

function setStatus(text, isError = false) { el.status.textContent = text; el.status.classList.toggle('err', isError); }
function setProgress(frac) {
  if (frac == null) { el.progressWrap.hidden = true; return; }
  el.progressWrap.hidden = false;
  el.bar.style.width = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + '%';
}
const selectedFolds = () => manifest ? manifest.folds.filter((f) => el.folds.querySelector(`input[value="${f.fold}"]`).checked) : [];
const keyOf = (f) => `fold${f.fold}`;

function refreshButtons() {
  const supported = 'showDirectoryPicker' in window;
  el.openFolder.disabled = busy || !supported;
  el.chooseOut.disabled = busy || !inputDir;
  el.run.disabled = busy || !entries.length || !manifest || selectedFolds().length === 0;
  el.run.textContent = entries.length ? `Segment all (${plural(entries.length, 'file')})` : 'Segment all';
  for (const i of [el.crop, el.minip, el.backend, el.mirror, ...el.folds.querySelectorAll('input')]) i.disabled = busy;
  el.clearCache.disabled = busy || !cacheState.files;
}
function refreshFoldInfo() {
  if (!manifest) return;
  const sel = selectedFolds(), toFetch = sel.filter((f) => !loadedKeys.has(keyOf(f)));
  el.foldInfo.textContent = !sel.length ? 'Pick at least one fold.'
    : `${sel.length === 1 ? '1 fold' : sel.length + ' folds averaged'} · ` +
      (toFetch.length ? `${fmtMB(toFetch.reduce((s, f) => s + f.size, 0))} to load (cached after the first download)` : 'loaded');
  refreshButtons();
}

// ---------------------------------------------------------------- model manifest + worker

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
  const v = new URL(import.meta.url).searchParams.get('v');   // same release tag as app.js (see index.html)
  worker = new Worker('./worker.js' + (v ? '?v=' + encodeURIComponent(v) : ''), { type: 'module' });
  workerBackend = backend;
  worker.onmessage = (e) => {
    const m = e.data, w = waiting.get(m.id);
    if (!w) return;
    if (m.type === 'error' || m.type === 'cancelled') { waiting.delete(m.id); w.reject(new Error(m.type === 'cancelled' ? 'cancelled' : m.message)); }
    else if (m.type === 'ready' || m.type === 'done') { waiting.delete(m.id); w.resolve(m); }
    else if (w.on) w.on(m);
  };
  worker.onerror = (e) => {
    const err = new Error(e.message || 'The processing worker failed to start');
    for (const w of waiting.values()) w.reject(err);
    waiting.clear(); worker = null; workerBackend = null; loadedKeys = new Set();
  };
}

function ask(msg, on, transfer = []) {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject, on });
    worker.postMessage({ ...msg, id }, transfer);
  });
}

async function ensureLoaded(folds) {
  ensureWorker();
  const models = folds.map((f) => ({ key: keyOf(f), url: WEIGHTS + f.file, size: f.size, sha256: f.sha256 }));
  const r = await ask({ type: 'load', models, patch: manifest.patch, prefer: el.backend.value }, (m) => {
    const what = `fold ${m.key.replace('fold', '')} (${m.index}/${m.count})`;
    if (m.phase === 'download') { setStatus(`Downloading ${what}… ${fmtMB(m.loaded)} / ${fmtMB(m.total)}`); setProgress(m.total ? m.loaded / m.total : null); }
    else if (m.phase === 'cache') setStatus(`Loading ${what} from browser storage…`);
    else if (m.phase === 'compile') setStatus(`Preparing ${what} on this device…`);
  });
  loadedKeys = new Set(r.loaded);
  runtimeLabel = r.ep === 'webgpu' ? 'WebGPU (GPU)' : `CPU / WebAssembly, ${plural(r.threads, 'thread')}`;
  el.modelInfo.textContent = `${manifest.name} · running on ${runtimeLabel}`;
  refreshFoldInfo();
}

// ---------------------------------------------------------------- input folder

const natural = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
const stemOf = (name) => (name.lastIndexOf('.') > 0 ? name.slice(0, name.lastIndexOf('.')) : name);

async function scanFolder(dir) {
  const found = [];
  async function walk(d, dirs) {
    const kids = [];
    for await (const [name, h] of d.entries()) kids.push([name, h]);
    kids.sort((a, b) => natural(a[0], b[0]));
    for (const [name, h] of kids) {
      if (name.startsWith('.')) continue;
      if (h.kind === 'directory') { if (!name.startsWith(OUT_PREFIX)) await walk(h, [...dirs, name]); continue; }
      const lower = name.toLowerCase();
      let kind = /\.(dcm|dicom)$/.test(lower) ? 'dcm' : lower.endsWith('.png') ? 'png' : null;
      if (!kind && !lower.includes('.')) {                 // DICOM exports often have no extension: look for "DICM"
        try {
          const f = await h.getFile();
          const head = new Uint8Array(await f.slice(128, 132).arrayBuffer());
          if (head[0] === 0x44 && head[1] === 0x49 && head[2] === 0x43 && head[3] === 0x4d) kind = 'dcm';
        } catch (e) { /* unreadable: ignore */ }
      }
      if (kind) found.push({ dirs, name, stem: stemOf(name), kind, handle: h, li: null });
    }
  }
  await walk(dir, []);
  return found;
}

const relPath = (e) => [...e.dirs, e.name].join('/');

function renderFileList() {
  el.fileList.innerHTML = '';
  el.fileList.hidden = !entries.length;
  const SHOW = 400;
  for (const e of entries.slice(0, SHOW)) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="kind ${e.kind}">${e.kind.toUpperCase()}</span><span class="name"></span><span class="st"></span>`;
    li.querySelector('.name').textContent = relPath(e);
    li.title = relPath(e);
    e.li = li;
    el.fileList.appendChild(li);
  }
  if (entries.length > SHOW) {
    const li = document.createElement('li');
    li.className = 'more';
    li.textContent = `… and ${entries.length - SHOW} more`;
    el.fileList.appendChild(li);
  }
}

function setEntryStatus(e, state, text) {
  if (!e.li) return;
  e.li.dataset.state = state;
  const st = e.li.querySelector('.st');
  st.textContent = text || '';
  st.title = text || '';
}

async function useInputFolder(dir) {
  inputDir = dir; outParent = null;
  el.inputInfo.textContent = `Scanning “${dir.name}”…`;
  entries = [];
  renderFileList(); refreshButtons();
  try { entries = await scanFolder(dir); }
  catch (e) { el.inputInfo.textContent = `Could not read “${dir.name}” (${e.message}).`; refreshButtons(); return; }
  const nd = entries.filter((e) => e.kind === 'dcm').length, np = entries.length - nd;
  el.inputInfo.textContent = entries.length
    ? `“${dir.name}”: ${plural(nd, 'DICOM file')}, ${plural(np, 'PNG file')} (subfolders included).`
    : `“${dir.name}” contains no DICOM or PNG files.`;
  renderFileList(); refreshOutInfo(); refreshButtons();
  setStatus(entries.length ? 'Ready. Press “Segment all”.' : 'Pick a folder that contains .dcm or .png files.');
}

async function openFolder() {
  let dir;
  try { dir = await window.showDirectoryPicker({ id: 'vesselSegInput', mode: 'read' }); }
  catch (e) { if (e.name !== 'AbortError') setStatus(`Could not open the folder (${e.message}).`, true); return; }
  await useInputFolder(dir);
}

function refreshOutInfo() {
  const parent = outParent || inputDir;
  el.outInfo.textContent = parent
    ? `Results go into a new folder “${OUT_PREFIX}<date>-<time>” inside “${parent.name}”${outParent ? '' : ' (the input folder)'}: one subfolder per input file, with f_1.png, f_1_mask.png, …`
    : 'Results go into a new folder inside the input folder.';
}

async function chooseOut() {
  try { outParent = await window.showDirectoryPicker({ id: 'vesselSegOutput', mode: 'readwrite' }); refreshOutInfo(); }
  catch (e) { if (e.name !== 'AbortError') setStatus(`Could not open that folder (${e.message}).`, true); }
}

// ---------------------------------------------------------------- batch run

function stamp() {
  const d = new Date(), p = (n) => String(n).padStart(2, '0');
  return `${OUT_PREFIX}${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function writeBytes(dir, name, bytes) {
  const fh = await dir.getFileHandle(name, { create: true });
  const w = await fh.createWritable();
  await w.write(bytes);
  await w.close();
}

async function ensureWritable(dir) {
  const opts = { mode: 'readwrite' };
  if ((await dir.queryPermission(opts)) === 'granted') return true;
  return (await dir.requestPermission(opts)) === 'granted';
}

async function processEntry(e, idx, target, onFrameTime) {
  const file = await e.handle.getFile();
  let info = null, writes = Promise.resolve(), frames = 0, lastFrameAt = performance.now();
  const n = entries.length;
  const res = await ask({ type: 'process', file, kind: e.kind, crop: el.crop.checked, minip: el.minip.checked, mirror: el.mirror.checked, keys: selectedFolds().map(keyOf) }, (m) => {
    if (m.type === 'fileinfo') {
      info = m;
      setEntryStatus(e, 'working', `0/${m.count}`);
    } else if (m.type === 'run') {
      const frac = (idx + (m.index + m.done / m.total) / m.count) / n;
      setProgress(frac);
      setStatus(`File ${idx + 1}/${n} · ${relPath(e)} · ${m.index + 1}/${m.count}`);
    } else if (m.type === 'frame') {
      const now = performance.now(); onFrameTime(now - lastFrameAt); lastFrameAt = now;
      frames++;
      writes = writes.then(() => writeBytes(target, `${m.name}.png`, m.imagePng)).then(() => writeBytes(target, `${m.name}_mask.png`, m.maskPng));
      setEntryStatus(e, 'working', `${m.index + 1}/${m.count}`);
      preview = { w: m.w, h: m.h, gray: m.gray, prob: m.prob };
      el.previewInfo.textContent = `${relPath(e)} · ${m.name} · ${m.w}×${m.h} · vessel ${(100 * m.fg / (m.w * m.h)).toFixed(1)} %`;
      draw();
    }
  });
  await writes;
  return { frames, info, ms: res.ms };
}

async function runAll() {
  if (busy || !entries.length) return;
  const folds = selectedFolds();
  if (!folds.length) return;
  const parent = outParent || inputDir;
  busy = true; cancelRequested = false;            // set before any await, so a double click cannot start two runs
  refreshButtons();
  try { if (!(await ensureWritable(parent))) throw new Error('permission was not granted'); }
  catch (e) {
    busy = false; refreshButtons();
    setStatus(`Cannot write into “${parent.name}” (${e.message}). Use “Save results elsewhere…”.`, true);
    return;
  }

  el.cancel.hidden = false; el.cancel.disabled = false;
  refreshButtons();
  for (const e of entries) setEntryStatus(e, 'waiting', '');
  const t0 = performance.now();
  let nOk = 0, nErr = 0, nFrames = 0, outName = '', stopped = false;
  const frameTimes = [];
  try {
    setStatus('Loading the model…'); setProgress(0);
    await ensureLoaded(folds);
    outName = stamp();
    const outRoot = await parent.getDirectoryHandle(outName, { create: true });
    const used = new Map();     // output directory path -> folder names already taken there
    for (let i = 0; i < entries.length; i++) {
      if (cancelRequested) { stopped = true; break; }
      const e = entries[i];
      setEntryStatus(e, 'working', 'reading…');
      try {
        let dir = outRoot;
        for (const part of e.dirs) dir = await dir.getDirectoryHandle(part, { create: true });
        const key = e.dirs.join('/'), taken = used.get(key) || new Set();
        used.set(key, taken);
        let folder = e.stem;                                   // a.dcm and a.png in one folder -> a, a_png
        if (taken.has(folder)) folder = `${e.stem}_${e.kind}`;
        for (let k = 2; taken.has(folder); k++) folder = `${e.stem}_${e.kind}_${k}`;
        taken.add(folder);
        const target = await dir.getDirectoryHandle(folder, { create: true });
        const r = await processEntry(e, i, target, (ms) => frameTimes.push(ms));
        nOk++; nFrames += r.frames;
        const sz = r.info ? `${r.info.w}×${r.info.h}` : '';
        const cropNote = r.info && (r.info.w !== r.info.srcW || r.info.h !== r.info.srcH) ? ` (cropped from ${r.info.srcW}×${r.info.srcH})` : '';
        setEntryStatus(e, 'done', `${plural(r.frames, 'image')} · ${sz}${cropNote}${r.info && r.info.note ? ' · ' + r.info.note : ''}`);
      } catch (err) {
        if (err.message === 'cancelled') { setEntryStatus(e, 'error', 'stopped (partly written)'); stopped = true; break; }
        nErr++; setEntryStatus(e, 'error', err.message);
      }
    }
    const per = frameTimes.length ? ` · ${fmtS(frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length)} per image` : '';
    setStatus(`${stopped ? 'Stopped' : 'Done'}: ${plural(nOk, 'file')}${nErr ? `, ${nErr} failed` : ''}, ${plural(nFrames, 'image')} in ${fmtS(performance.now() - t0)}${per}. Saved to “${parent.name}/${outName}”.`, nErr > 0);
    setProgress(null);
  } catch (e) {
    setStatus(e.message, true); setProgress(null);
  } finally {
    busy = false; el.cancel.hidden = true;
    refreshButtons(); refreshCacheInfo();
  }
}

function cancelRun() {
  if (!busy) return;
  cancelRequested = true;
  el.cancel.disabled = true;
  if (worker) worker.postMessage({ type: 'cancel' });
  setStatus('Stopping…');
}

// ---------------------------------------------------------------- preview

function draw() {
  if (!preview) return;
  const { w, h, gray, prob } = preview;
  const mode = document.querySelector('input[name=mode]:checked').value;
  el.empty.hidden = true; el.view.hidden = false;
  el.view.width = w; el.view.height = h;
  const ctx = el.view.getContext('2d');
  const out = ctx.createImageData(w, h);
  const d = out.data, a = el.opacity.value / 100;
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    let r = gray[i], g = gray[i], b = gray[i];
    if (mode === 'mask') { r = g = b = prob[i] >= 128 ? 255 : 0; }
    else if (mode === 'prob') { r = g = b = prob[i]; }
    else if (mode === 'overlay' && prob[i] >= 128) { r = r * (1 - a) + 255 * a; g = g * (1 - a) + 64 * a; b = b * (1 - a) + 64 * a; }
    d[j] = r; d[j + 1] = g; d[j + 2] = b; d[j + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

// ---------------------------------------------------------------- downloaded-model storage
// The worker keeps each downloaded fold in the Cache API (per browser, per site). This section shows what is
// stored and deletes it on request.

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
    cacheState = s || { files: 0, bytes: 0 };
    el.cacheInfo.textContent = !s ? 'This browser has no cache storage, so models are not kept between visits.'
      : s.files ? `${plural(s.files, 'model file')} stored in this browser · ${fmtMB(s.bytes)}` : 'No model files stored in this browser.';
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
  if (!el.clearCache.classList.contains('armed')) {   // the first click only arms the button
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
    if (worker) { worker.terminate(); worker = null; workerBackend = null; }   // drop the in-memory copies too
    loadedKeys = new Set();
    if (manifest) el.modelInfo.textContent = `${manifest.name} · ${manifest.folds.length} folds · ${manifest.patch[0]}×${manifest.patch[1]} patches`;
    setStatus(`Deleted ${plural(before.files, 'model file')} (${fmtMB(before.bytes)}). The next run downloads them again.`);
  } catch (e) {
    setStatus(`Could not delete the stored models (${e.message}).`, true);
  } finally {
    busy = false;
    refreshFoldInfo();
    await refreshCacheInfo();
  }
}

// ---------------------------------------------------------------- wiring

el.openFolder.addEventListener('click', openFolder);
el.chooseOut.addEventListener('click', chooseOut);
el.run.addEventListener('click', runAll);
el.cancel.addEventListener('click', cancelRun);
el.backend.addEventListener('change', () => {
  if (worker && !busy) { worker.terminate(); worker = null; workerBackend = null; loadedKeys = new Set(); refreshFoldInfo(); }
});
el.opacity.addEventListener('input', draw);
document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', draw));
el.clearCache.addEventListener('click', clearCache);
el.clearCache.addEventListener('blur', disarmClear);

// Hooks for automated checks: window.__vesselSeg.useInputFolder(dirHandle), .runAll(), .isBusy(), .entries()
window.__vesselSeg = { useInputFolder, runAll, isBusy: () => busy, entries: () => entries.map((e) => ({ path: relPath(e), kind: e.kind, state: e.li && e.li.dataset.state, info: e.li && e.li.querySelector('.st').textContent })) };

if (!('showDirectoryPicker' in window)) el.unsupported.hidden = false;
if (!('gpu' in navigator)) el.backend.querySelector('option[value=webgpu]').disabled = true;
refreshButtons();
loadManifest();
refreshCacheInfo();
