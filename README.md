# DSA vessel segmentation in the browser

Batch-segment cerebral vessels in DSA (digital subtraction angiography) **DICOM and PNG files** with a 5-fold
nnU-Net — entirely inside your web browser. No server, no install, and no image leaves your computer.

**Live app:** <https://jctaylor666.github.io/dsa-vessel-seg-web/>
**Weights:** <https://huggingface.co/cjy666/dsa-vessel-seg> (5 folds, fp32 ONNX, ~185 MB each)

> Research use only. Not a medical device.

## Use it

Open the live app in **Chrome or Edge** on a desktop (folder access needs their File System Access API;
WebGPU makes it fast, otherwise it falls back to the CPU).

1. **Open folder…** — the app finds every `.dcm` / `.dicom` file, extension-less DICOM file and `.png` in the
   folder and its subfolders.
2. **Preprocess** — *Remove black border* (on by default) crops to the area of non-zero pixels. A DICOM is
   cropped with one box for all its frames (the box of its MinIP), so its frames keep one size. Off: images are
   used as they are. Optionally also save each DICOM's MinIP and its mask.
3. **Model** — pick the folds to average (default: all 5) and, optionally, mirroring TTA.
4. **Segment all.** The first run downloads the selected folds and keeps them in the browser; later runs start
   right away. **Stop** ends the run at once; the file being processed is left partly written (its status says so).

### Output

A new folder `vessel_seg_output_<date>-<time>` is created inside the input folder (or in a folder you choose
with *Save results elsewhere…*). It mirrors the input's subfolders, with **one folder per input file**, named
after the file without its extension:

```
vessel_seg_output_20261120-160306/
├─ plain_frame/            ← plain_frame.png
│  ├─ f_1.png              the image that was segmented (cropped if "Remove black border" is on)
│  └─ f_1_mask.png         vessel mask, 0 / 255
├─ run_0042/               ← run_0042.dcm, 14 frames
│  ├─ f_1.png … f_14.png
│  ├─ f_1_mask.png … f_14_mask.png
│  └─ minip.png, minip_mask.png   (only with "Also save each DICOM's MinIP")
└─ sub/…                   ← files from subfolders keep their place
```

Every PNG is **8-bit single-channel** (H × W), like the pipeline's PIL `"L"` images; each mask has the same
size as its image. If two inputs in one folder share a name (`a.dcm`, `a.png`), the second folder is `a_png`.

## DICOM support

| transfer syntax | |
|---|---|
| Implicit / Explicit VR Little Endian (uncompressed) | ✓ |
| JPEG Lossless, Process 14 (incl. Selection Value 1) | ✓ |
| JPEG Baseline (8-bit) | via the browser's JPEG decoder — not yet tested on real files |
| JPEG 2000, JPEG-LS, RLE, colour images | not supported — the file is reported and skipped |

Conversion to 8-bit follows the original preprocessing exactly: raw stored values, first `WindowCenter` /
`WindowWidth`, `v = clip((x − (c − 0.5)) / (w − 1) + 0.5, 0, 1)`, `uint8(v · 255)` in float32 (min–max when
the window is missing); MinIP = per-pixel minimum over the raw frames, then windowed; black border = outside the
bounding box of pixels > 0. On four test runs (Siemens 8-bit and 12-bit JPEG Lossless, Philips 12-bit JPEG
Lossless, uncompressed) all frames and MinIPs are pixel-identical to the pipeline's PNGs and the cropped sizes are
identical. Frames whose own border differed from the run's were *resized* by the pipeline; here they are cropped
with the run's box, so the size is the same but those frames are not resampled.

A JPEG Lossless frame with a point transform (Pt > 0) is shifted back after decoding — the bundled decoder skips
that step.

## Speed

One 791 × 1022 frame on an Apple-silicon laptop: WebGPU 1.1 s (1 fold), 4.1 s (5 folds), 15.6 s (5 folds +
mirroring); CPU/WebAssembly about 22 s per fold. A batch of 4 DICOMs + 1 PNG (50 images up to 1231 × 1231, 5
folds) took 2 min 52 s on WebGPU.

## How it works

- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.31.1 runs the model, on WebGPU when
  available and on WebAssembly otherwise, in a Web Worker.
- [`worker.js`](worker.js) reproduces nnU-Net v2's predictor: crop to the non-zero bounding box, z-score, pad to
  the 896 × 768 patch, Gaussian-weighted sliding window at step 0.5, optional mirroring, and the **mean of the
  logits** over the selected folds. Against `nnUNetPredictor` the masks are pixel-identical with 5 folds +
  mirroring and differ in 2 of 808k pixels without mirroring (float rounding at the 0.5 boundary).
- [`dicom.js`](dicom.js) decodes DICOM and does the 8-bit conversion and crop; [`png.js`](png.js) writes
  grayscale PNGs. `vendor/` holds the DICOM parser and JPEG Lossless decoder (MIT, see `vendor/LICENSES.md`).
- The model list comes from `model.json` in the weights repo (file names, sizes, SHA-256). Downloads are
  checksum-verified and cached with the Cache API; **Delete downloaded models** removes them again.

## Run locally / use other weights

Any static file server works, e.g. `python3 -m http.server 8000`, then open <http://localhost:8000/>. To point
the app at another weights folder with the same `model.json` layout, add `?weights=<base URL>`.

When you change any file, bump `?v=` on `style.css` and `app.js` in `index.html` (app.js passes it on to the
worker and its modules): GitHub Pages lets browsers cache files for 10 minutes, and a mix of old and new files
breaks the page.

The CPU backend runs single-threaded on GitHub Pages because Pages cannot send the COOP/COEP headers that
WebAssembly threads need.
