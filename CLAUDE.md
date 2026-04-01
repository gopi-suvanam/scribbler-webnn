# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Standalone Stable Diffusion image generation module using the WebNN API and ONNX Runtime Web. Fully self-contained — no parent directory references. Designed to be hosted on GitHub Pages or any static server.

Supports three models: **SD Turbo** (1 step), **SDXL Turbo** (1 step), **SD 1.5** (20/25/50 steps).

## Commands

```bash
node server.js    # Dev server at http://localhost:8080 (sets COOP/COEP/CORS headers)
```

No build step, no npm install required — just Node.js.

## Architecture

### Files

| File | Purpose |
|------|---------|
| `sd.js` | Core module — exports `SDPipeline`, `AVAILABLE_MODELS`, `SD15_STEP_OPTIONS` |
| `app.js` | UI glue wiring `index.html` controls to `SDPipeline` |
| `index.html` | Single-page demo with model/steps dropdowns, prompt input, canvas |
| `style.css` | Dark theme |
| `server.js` | Minimal Node HTTP server with required COOP/COEP/CORS headers |
| `WebNN-Stable-Diffusion.jsnb` | Scribbler notebook — batch generates 10 images, zips & downloads |
| `tokenizers/` | Local tokenizer JSON files per model (small, committed to repo) |

### `sd.js` Internals

**Dependencies (all from CDN, no local copies):**
- ONNX Runtime Web — loaded dynamically by `index.html` from jsdelivr; `sd.js` uses `ort` as a global
- Transformers.js `@xenova/transformers@2.13.4` — imported as ES module from jsdelivr for tokenization

**ONNX models** are fetched from HuggingFace at runtime and cached in OPFS (Origin Private File System). Cache key format: `${modelId}_${modelName}`.

**Tokenizer paths** resolve relative to `sd.js` via `import.meta.url`, not the importing page. This allows cross-origin import from Scribbler or other hosts.

**Pipeline per model:**
- **SD Turbo**: text encoder (int32 input) → UNet (float16) → EulerA 1-step in float32 → VAE decoder (float32 input)
- **SDXL Turbo**: TE1 (int32) + TE2 (int64) → concat hidden states in JS → UNet (float16, needs text_embeds + time_ids) → EulerA 1-step → VAE decoder (float16 input)
- **SD 1.5**: text encoder batch=2 (positive + negative) → 20/25/50 UNet iterations with CFG guidance scale 7.5 → VAE decoder (float16 input)

**Scheduler constants** (`SD15_SCHEDULES`): precomputed sigma and timestep arrays for 20/25/50 steps, derived from SD 1.5 noise schedule (beta_start=0.00085, beta_end=0.012, quadratic scaling).

### Session Options

Each model requires different ONNX Runtime session options:
- **SD Turbo**: `enableMemPattern: false`, `enableCpuMemArena: false`, `extra.session` memory options, `graphOptimizationLevel: "disabled"` for text_encoder and unet
- **SDXL Turbo / SD 1.5**: minimal options — just `freeDimensionOverrides` per sub-model

These are configured in `MODEL_CONFIGS[modelId].sessionOptions` (base) and `MODEL_CONFIGS[modelId].modelOptions[name]` (per sub-model). Mixing them up causes WebNN dispatch failures.

## Critical Patterns

### Float16Array vs Uint16Array

Browsers with native `Float16Array` return float values from tensor `.data` (e.g. `0.5`). Older browsers return `Uint16Array` with raw bit patterns (e.g. `14336`). Two helpers handle this:
- `readF16(data, i)` — reads a float from either type
- `asUint16(data)` — gets raw uint16 bit-pattern view for concatenation/copying between tensors

**Always use `asUint16(tensor.data).slice()` when copying tensor data between sessions.** Using `new Uint16Array(float16Array)` will corrupt data by converting float values to integers.

### WebNN Tensor Reuse

Output tensors from one WebNN session **cannot** be passed as inputs to another session. Always copy into a fresh `ort.Tensor`:
```js
const fresh = new ort.Tensor("float16", asUint16(outputTensor.data).slice(), outputTensor.dims);
```

### No Concurrent Dispatch

WebNN does not support concurrent `dispatch()` on the same MLContext. SDXL text encoders must run sequentially (not `Promise.all`).

### Latent Math Precision

Turbo models: keep latents in **float32** for scheduler math, convert to float16 only for UNet input tensors. The scheduler step uses the **original unscaled** latents (not the `1/sqrt(sigma^2+1)` scaled version sent to UNet).

SD 1.5: latents are float32 throughout the denoising loop, converted to float16 only when creating UNet input tensors and the final VAE input.
