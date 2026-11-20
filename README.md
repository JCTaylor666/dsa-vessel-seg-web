# DSA vessel segmentation in the browser

Segment cerebral vessels in a DSA (digital subtraction angiography) frame with a 5-fold nnU-Net — entirely
inside your web browser. No server, no install, and the image never leaves your computer.

**Live app:** <https://jctaylor666.github.io/dsa-vessel-seg-web/>
**Weights:** <https://huggingface.co/cjy666/dsa-vessel-seg> (5 folds, fp32 ONNX, ~185 MB each)

> Research use only. Not a medical device.

## Use it

1. Open the live app in Chrome or Edge (any browser with WebGPU works best; others fall back to the CPU).
2. Choose, drop or paste an 8-bit grayscale DSA frame (PNG / JPG).
3. Pick the folds to average (default: all 5) and, optionally, mirroring TTA.
4. **Segment.** The first run downloads the selected folds (~185 MB each) and keeps them in the browser
   cache; later runs start immediately.
5. View the overlay, mask or probability map and download them as PNG.
6. To free the disk space again, use **Delete downloaded models** (section 4). It shows how many model files
   this browser holds and how big they are, and removes them after a second click. Clearing the site's data
   in the browser settings does the same.

Typical time for one 791 × 1022 frame on an Apple-silicon laptop:

| folds | mirroring | WebGPU | CPU (WebAssembly, 1 thread) |
|---|---|---|---|
| 1 | no | 1.1 s | 22 s |
| 1 | yes | 3.2 s | — |
| 5 | no | 4.1 s | — |
| 5 | yes | 15.6 s | — |

## How it works

- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.31.1 runs the model, on WebGPU when
  available and on WebAssembly otherwise. Inference happens in a Web Worker so the page stays responsive.
- [`worker.js`](worker.js) reproduces nnU-Net v2's predictor step by step: crop to the non-zero bounding box,
  z-score, pad to the 896 × 768 patch, Gaussian-weighted sliding window at step 0.5, optional mirroring, and
  the **mean of the logits** over the selected folds (nnU-Net's ensembling). On a test frame the browser mask
  is pixel-identical to `nnUNetPredictor` with 5 folds + mirroring, and differs in 2 of 808k pixels without
  mirroring (float rounding at the 0.5 boundary).
- The model list comes from `model.json` in the weights repo (file names, sizes, SHA-256). Every download is
  checksum-verified before use and cached with the Cache API.

## Run locally / use other weights

Any static file server works:

```bash
python3 -m http.server 8000
```

then open <http://localhost:8000/>. To point the app at another weights folder with the same `model.json`
layout, add `?weights=<base URL>`, e.g. `?weights=/my-weights/`.

The CPU backend runs single-threaded on GitHub Pages because Pages cannot send the COOP/COEP headers that
WebAssembly threads need.

## Files

| file | role |
|---|---|
| `index.html`, `style.css` | the page |
| `app.js` | image loading, fold selection, drawing, PNG export |
| `worker.js` | model download + cache, ONNX Runtime sessions, nnU-Net-style inference |
