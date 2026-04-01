# WebNN Stable Diffusion

Standalone Stable Diffusion image generation in the browser using the [WebNN API](https://www.w3.org/TR/webnn/) and [ONNX Runtime Web](https://onnxruntime.ai/). Fully self-contained — no build step, no npm install required.

Supports three models:

| Model | Steps | Output |
|-------|-------|--------|
| **SD Turbo** | 1 | 512x512 |
| **SDXL Turbo** | 1 | 512x512 |
| **Stable Diffusion 1.5** | 20 / 25 / 50 | 512x512 |

## Quick Start

```bash
node server.js
# Open http://localhost:8080
```

The dev server sets the required `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers for `SharedArrayBuffer` support.

## `sd.js` API Reference

`sd.js` is the core module. It exports `SDPipeline`, `AVAILABLE_MODELS`, and `SD15_STEP_OPTIONS`.

### Exports

```js
import { SDPipeline, AVAILABLE_MODELS, SD15_STEP_OPTIONS } from './sd.js';
```

- **`AVAILABLE_MODELS`** — Object mapping model IDs to display names:
  ```js
  {
    "sd-turbo": "SD Turbo",
    "sdxl-turbo": "SDXL Turbo",
    "sd-1.5": "Stable Diffusion 1.5"
  }
  ```
- **`SD15_STEP_OPTIONS`** — `[20, 25, 50]` — available step counts for SD 1.5 (turbo models always use 1 step).

### `new SDPipeline(modelId, options?)`

Creates a pipeline instance for the given model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `modelId` | `string` | *(required)* | One of `'sd-turbo'`, `'sdxl-turbo'`, `'sd-1.5'` |
| `options.provider` | `string` | `'webnn'` | Execution provider: `'webnn'`, `'wasm'`, or `'webgpu'` |
| `options.deviceType` | `string` | `'gpu'` | WebNN device type: `'gpu'`, `'npu'`, or `'cpu'` |
| `options.tokenizerBase` | `string` | `'./tokenizers'` (relative to `sd.js`) | URL to tokenizer directory |

### `pipeline.loadModels(onProgress?)`

Downloads ONNX models from HuggingFace (cached in OPFS after first load), initializes tokenizers, and creates ONNX Runtime inference sessions.

**Progress callback:**

```js
await pipeline.loadModels(({ stage, model, index, total, loaded, totalBytes, cached }) => {
    // stage: 'download' | 'cache' | 'compile'
    // model: sub-model name (e.g. 'text_encoder', 'unet', 'vae_decoder')
    // index: current model index (0-based)
    // total: total number of models
    // loaded/totalBytes: download progress in bytes
    // cached: true if loaded from OPFS cache
});
```

### `pipeline.generate(prompt, options?)`

Generates a 512x512 image from a text prompt.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `string` | *(required)* | Text prompt for image generation |
| `options.seed` | `number` | random | Seed for reproducible results |
| `options.steps` | `number` | model default | Step count (SD 1.5 only: 20, 25, or 50) |
| `options.onProgress` | `function` | — | Callback `({ step, totalSteps })` |

**Returns:** `Promise<{ imageData: ImageData, timing: object }>`

- `imageData` — standard `ImageData` (512x512), ready for `canvas.putImageData()`
- `timing` — object with `textEncode`, `unet`, `vaeDecode`, and `total` (all in ms)

### `pipeline.release()`

Releases all ONNX Runtime inference sessions and frees resources.

### Usage Example

```html
<!-- Load ONNX Runtime Web first -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js"></script>
```

```js
import { SDPipeline, AVAILABLE_MODELS } from './sd.js';

const pipeline = new SDPipeline('sd-turbo');

await pipeline.loadModels(progress => {
    console.log(`${progress.stage} ${progress.model} (${progress.index + 1}/${progress.total})`);
});

const { imageData, timing } = await pipeline.generate('a cat sitting on a windowsill');

const canvas = document.getElementById('canvas');
canvas.getContext('2d').putImageData(imageData, 0, 0);

console.log(`Generated in ${timing.total.toFixed(0)}ms`);

// When done
await pipeline.release();
```

## Architecture

### Files

| File | Purpose |
|------|---------|
| `sd.js` | Core module — `SDPipeline`, model configs, schedulers, float16 helpers |
| `sample/app.js` | UI glue wiring `index.html` controls to `SDPipeline` |
| `sample/index.html` | Single-page demo with model/steps dropdowns, prompt input, canvas |
| `sample/style.css` | Dark theme styling |
| `server.js` | Minimal Node HTTP server with required COOP/COEP/CORS headers |
| `tokenizers/` | Local tokenizer JSON files per model |

### Dependencies

All loaded from CDN at runtime — nothing to install:

- **ONNX Runtime Web** — loaded by `index.html` from jsdelivr; `sd.js` uses `ort` as a global
- **Transformers.js** (`@xenova/transformers@2.13.4`) — imported as ES module by `sd.js` for tokenization

### Model Pipeline Details

**SD Turbo:**
Text encoder (int32 input) → UNet (float16) → EulerA 1-step in float32 → VAE decoder (float32 input)

**SDXL Turbo:**
TE1 (int32) + TE2 (int64) → concatenate hidden states [1,77,2048] → UNet (float16, with text_embeds + time_ids) → EulerA 1-step → VAE decoder (float16 input)

**Stable Diffusion 1.5:**
Text encoder batch=2 (positive + negative prompt) → 20/25/50 UNet iterations with classifier-free guidance (scale 7.5) → VAE decoder (float16 input)

### Model Caching

ONNX models are fetched from HuggingFace at runtime and cached in the browser's [Origin Private File System (OPFS)](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system). Subsequent loads skip the download entirely.

## Requirements

- A browser with [WebNN support](https://webmachinelearning.github.io/webnn-status/) (e.g., Chrome/Edge with appropriate flags)
- Node.js (any recent version) for the dev server
- `SharedArrayBuffer` support (provided by the COOP/COEP headers from `server.js`)

## License

See repository for license details.
