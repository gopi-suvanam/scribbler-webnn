/**
 * sd.js - WebNN Stable Diffusion Pipeline
 *
 * Supports SD Turbo, SDXL Turbo, and Stable Diffusion 1.5.
 * Requires ort (ONNX Runtime Web) to be loaded as a global before import.
 *
 * Usage:
 *   import { SDPipeline, AVAILABLE_MODELS } from './sd.js';
 *   const pipeline = new SDPipeline('sd-turbo');
 *   await pipeline.loadModels(progress => console.log(progress));
 *   const { imageData, timing } = await pipeline.generate('a cat');
 */

import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.13.4";

// Resolve tokenizer paths relative to sd.js, not the HTML page
const _sdBaseUrl = new URL(".", import.meta.url).href;

export const AVAILABLE_MODELS = {
    "sd-turbo": "SD Turbo",
    "sdxl-turbo": "SDXL Turbo",
    "sd-1.5": "Stable Diffusion 1.5",
};

// Available step counts for SD 1.5 (turbo models are always 1 step)
export const SD15_STEP_OPTIONS = [20, 25, 50];

const SD15_MODELS = [
    { name: "text_encoder", file: "text-encoder.onnx" },
    { name: "unet", file: "sd-unet-v1.5-model-b2c4h64w64s77-float16-compute-and-inputs-layernorm.onnx" },
    { name: "vae_decoder", file: "Stable-Diffusion-v1.5-vae-decoder-float16-fp32-instancenorm.onnx" },
];
const SD15_MODEL_OPTIONS = {
    text_encoder: { freeDimensionOverrides: { batch: 2, sequence: 77 } },
    unet: {
        freeDimensionOverrides: {
            batch: 2, channels: 4, height: 64, width: 64, sequence: 77,
            unet_sample_batch: 2, unet_sample_channels: 4,
            unet_sample_height: 64, unet_sample_width: 64,
            unet_time_batch: 2, unet_hidden_batch: 2, unet_hidden_sequence: 77,
        },
    },
    vae_decoder: { freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 } },
};

const MODEL_CONFIGS = {
    "sd-turbo": {
        hfRepo: "microsoft/sd-turbo-webnn",
        width: 512,
        height: 512,
        latentChannels: 4,
        steps: 1,
        sessionOptions: {
            enableMemPattern: false,
            enableCpuMemArena: false,
            extra: {
                session: {
                    disable_prepacking: "1",
                    use_device_allocator_for_initializers: "1",
                    use_ort_model_bytes_directly: "1",
                    use_ort_model_bytes_for_initializers: "1",
                },
            },
        },
        models: [
            { name: "text_encoder", file: "text_encoder/model_layernorm.onnx" },
            { name: "unet", file: "unet/model_layernorm.onnx" },
            { name: "vae_decoder", file: "vae_decoder/model.onnx" },
        ],
        modelOptions: {
            text_encoder: { graphOptimizationLevel: "disabled" },
            unet: { graphOptimizationLevel: "disabled" },
            vae_decoder: {
                freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 },
            },
        },
    },
    "sdxl-turbo": {
        hfRepo: "webnn/sdxl-turbo",
        width: 512,
        height: 512,
        latentChannels: 4,
        steps: 1,
        models: [
            { name: "text_encoder_1", file: "onnx/text_encoder_model_qdq_q4f16.onnx" },
            { name: "text_encoder_2", file: "onnx/text_encoder_2_model_qdq_q4f16.onnx" },
            { name: "unet", file: "onnx/unet_model_qdq_q4f16.onnx" },
            { name: "vae_decoder", file: "onnx/vae_decoder_model_qdq_q4f16.onnx" },
        ],
        modelOptions: {
            text_encoder_1: {
                freeDimensionOverrides: { batch_size: 1, sequence_length: 77 },
            },
            text_encoder_2: {
                freeDimensionOverrides: { batch_size: 1, sequence_length: 77 },
            },
            unet: {
                freeDimensionOverrides: {
                    unet_sample_batch: 1, unet_sample_channels: 4,
                    unet_sample_height: 64, unet_sample_width: 64,
                    unet_time_batch: 1, unet_hidden_batch: 1, unet_hidden_sequence: 77,
                    unet_text_embeds_batch: 1, unet_text_embeds_size: 1280,
                    unet_time_ids_batch: 1, unet_time_ids_size: 6,
                },
            },
            vae_decoder: {
                freeDimensionOverrides: {
                    batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64,
                },
            },
        },
    },
    "sd-1.5": {
        hfRepo: "microsoft/stable-diffusion-v1.5-webnn",
        width: 512, height: 512, latentChannels: 4, steps: 25,
        models: SD15_MODELS, modelOptions: SD15_MODEL_OPTIONS,
    },
};

// SD 1.5 EulerA scheduler constants for different step counts
// sigma(t) = sqrt((1 - alpha_cumprod(t)) / alpha_cumprod(t)), beta schedule: linspace(sqrt(0.00085), sqrt(0.012), 1000)^2
const SD15_SCHEDULES = {
    20: {
        sigmas: [
            14.614641, 10.746721, 8.0814910, 6.2049076, 4.8556332, 3.8653735, 3.1237518, 2.5571647, 2.1156539, 1.7648208,
            1.4805796, 1.2458125, 1.0481420, 0.87842847, 0.72971897, 0.59643457, 0.47358605, 0.35554688, 0.23217032, 0.029167158,
            0.0,
        ],
        timesteps: [
            999.0, 946.421, 893.842, 841.263, 788.684, 736.105, 683.526, 630.947, 578.368, 525.789,
            473.211, 420.632, 368.053, 315.474, 262.895, 210.316, 157.737, 105.158, 52.579, 0.0,
        ],
    },
    25: {
        sigmas: [
            14.614647, 11.435942, 9.076809, 7.3019943, 5.9489183, 4.903778, 4.0860896, 3.4381795, 2.9183085, 2.495972,
            2.1485956, 1.8593576, 1.6155834, 1.407623, 1.2280698, 1.0711612, 0.9323583, 0.80802417, 0.695151, 0.5911423,
            0.49355352, 0.3997028, 0.30577788, 0.20348993, 0.02916753, 0.0,
        ],
        timesteps: [
            999.0, 957.375, 915.75, 874.125, 832.5, 790.875, 749.25, 707.625, 666.0, 624.375, 582.75, 541.125, 499.5, 457.875,
            416.25, 374.625, 333.0, 291.375, 249.75, 208.125, 166.5, 124.875, 83.25, 41.625, 0.0,
        ],
    },
    50: {
        sigmas: [
            14.614641, 12.936614, 11.491548, 10.242847, 9.1602540, 8.2187178, 7.3971129, 6.6779678, 6.0465082, 5.4902753,
            4.9988414, 4.5632815, 4.1760835, 3.8308389, 3.5220445, 3.2450491, 2.9958193, 2.7709050, 2.5673388, 2.3825343,
            2.2142801, 2.0606394, 1.9199299, 1.7906919, 1.6716395, 1.5616553, 1.4597541, 1.3650671, 1.2768319, 1.1943695,
            1.1170794, 1.0444258, 0.97592826, 0.91115626, 0.84971899, 0.79126039, 0.73545198, 0.68198767, 0.63057536, 0.58093118,
            0.53277015, 0.48579205, 0.43966735, 0.39400450, 0.34830263, 0.30185606, 0.25353113, 0.20118204, 0.13934672, 0.029167158,
            0.0,
        ],
        timesteps: [
            999.0, 978.612, 958.224, 937.837, 917.449, 897.061, 876.673, 856.286, 835.898, 815.510,
            795.122, 774.735, 754.347, 733.959, 713.571, 693.184, 672.796, 652.408, 632.020, 611.633,
            591.245, 570.857, 550.469, 530.082, 509.694, 489.306, 468.918, 448.531, 428.143, 407.755,
            387.367, 366.980, 346.592, 326.204, 305.816, 285.429, 265.041, 244.653, 224.265, 203.878,
            183.490, 163.102, 142.714, 122.327, 101.939, 81.551, 61.163, 40.776, 20.388, 0.0,
        ],
    },
};

// ─── Float16 helpers ───────────────────────────────────────────────────────────

function toHalf(val) {
    const f32 = new Float32Array(1);
    const i32 = new Int32Array(f32.buffer);
    f32[0] = val;
    const x = i32[0];
    let bits = (x >> 16) & 0x8000;
    let m = (x >> 12) & 0x07ff;
    const e = (x >> 23) & 0xff;
    if (e < 103) return bits;
    if (e > 142) {
        bits |= 0x7c00;
        bits |= (e === 255 ? 0 : 1) && x & 0x007fffff;
        return bits;
    }
    if (e < 113) {
        m |= 0x0800;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

const _hasFloat16 = typeof Float16Array !== "undefined";

function f16(v) {
    return toHalf(v);
}

function f16Array(arr) {
    if (_hasFloat16) return Float16Array.from(arr);
    const out = new Uint16Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = toHalf(arr[i]);
    return out;
}

// Get raw uint16 bit-pattern view of float16 tensor data (works for both Float16Array and Uint16Array)
function asUint16(data) {
    if (data instanceof Uint16Array) return data;
    // Float16Array — view the same buffer as Uint16Array to get raw bits
    return new Uint16Array(data.buffer, data.byteOffset, data.length);
}

// Read a float value from tensor data (handles Float16Array, Uint16Array bit patterns, and Float32Array)
function readF16(data, i) {
    if (data instanceof Float32Array) return data[i];
    if (_hasFloat16 && data instanceof Float16Array) return data[i];
    return _f16toF32(data[i]);
}

// ─── OPFS cache ────────────────────────────────────────────────────────────────

async function cacheGet(key) {
    try {
        const root = await navigator.storage.getDirectory();
        const dir = await root.getDirectoryHandle("sd-model-cache", { create: true });
        const fh = await dir.getFileHandle(key);
        const file = await fh.getFile();
        return await file.arrayBuffer();
    } catch {
        return null;
    }
}

async function cacheSet(key, buffer) {
    try {
        const root = await navigator.storage.getDirectory();
        const dir = await root.getDirectoryHandle("sd-model-cache", { create: true });
        const fh = await dir.getFileHandle(key, { create: true });
        const writable = await fh.createWritable();
        await writable.write(buffer);
        await writable.close();
    } catch (e) {
        console.warn("OPFS cache write failed:", e);
    }
}

async function fetchWithCache(url, key, onProgress) {
    const cached = await cacheGet(key);
    if (cached) {
        onProgress && onProgress({ cached: true });
        return cached;
    }

    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);

    const total = parseInt(response.headers.get("content-length") || "0", 10);
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        onProgress && onProgress({ loaded, total, cached: false });
    }

    const buffer = await new Blob(chunks).arrayBuffer();
    cacheSet(key, buffer); // fire-and-forget
    return buffer;
}

// ─── PRNG for latent noise (SD 1.5) ──────────────────────────────────────────

function makePrng(seed) {
    let a = seed >>> 0,
        b = 0x9e3779b9,
        c = 0x6c62272e,
        d = 0xca367e8d;
    function next() {
        let t = b << 9;
        let r = a * 5;
        r = (((r << 7) | (r >>> 25)) * 9) >>> 0;
        c ^= a;
        d ^= b;
        b ^= c;
        a ^= d;
        c ^= t;
        d = ((d << 11) | (d >>> 21)) >>> 0;
        return (r >>> 0) / 0x100000000;
    }
    // Box-Muller normal distribution
    let spare = null;
    return () => {
        if (spare !== null) {
            const v = spare;
            spare = null;
            return v;
        }
        let u, v2, s;
        do {
            u = next() * 2 - 1;
            v2 = next() * 2 - 1;
            s = u * u + v2 * v2;
        } while (s >= 1 || s === 0);
        const mul = Math.sqrt((-2 * Math.log(s)) / s);
        spare = v2 * mul;
        return u * mul;
    };
}

// ─── SDPipeline ────────────────────────────────────────────────────────────────

export class SDPipeline {
    /**
     * @param {string} modelId - one of 'sd-turbo', 'sdxl-turbo', 'sd-1.5'
     * @param {object} options
     * @param {string} [options.provider='webnn'] - 'webnn' | 'wasm' | 'webgpu'
     * @param {string} [options.deviceType='gpu'] - 'gpu' | 'npu' | 'cpu'
     * @param {string} [options.tokenizerBase] - URL to tokenizer dirs (defaults to ./tokenizers/ relative to sd.js)
     */
    constructor(modelId, { provider = "webnn", deviceType = "gpu", tokenizerBase = `${_sdBaseUrl}tokenizers` } = {}) {
        if (!AVAILABLE_MODELS[modelId]) throw new Error(`Unknown model: ${modelId}`);
        this.modelId = modelId;
        this.config = MODEL_CONFIGS[modelId];
        this.provider = provider;
        this.deviceType = deviceType;
        this.tokenizerBase = tokenizerBase;
        this.sessions = {};
        this.tokenizer = null;
        this.tokenizer2 = null;
    }

    // ── Tokenizer init ────────────────────────────────────────────────────────

    async initTokenizer() {
        env.allowLocalModels = true;
        env.allowRemoteModels = false;
        env.localModelPath = `${this.tokenizerBase}/`;

        if (this.modelId === "sdxl-turbo") {
            this.tokenizer = await AutoTokenizer.from_pretrained("sdxl-1");
            this.tokenizer2 = await AutoTokenizer.from_pretrained("sdxl-2");
        } else if (this.modelId === "sd-1.5") {
            this.tokenizer = await AutoTokenizer.from_pretrained("sd-1.5");
        } else {
            this.tokenizer = await AutoTokenizer.from_pretrained("sd-turbo");
        }
    }

    // ── Model loading ─────────────────────────────────────────────────────────

    /**
     * Load all ONNX models (with OPFS caching).
     * @param {function} onProgress - callback({ stage, model, index, total, loaded, totalBytes, cached })
     */
    async loadModels(onProgress) {
        const cfg = this.config;
        const baseUrl = `https://huggingface.co/${cfg.hfRepo}/resolve/main/`;
        const total = cfg.models.length;

        await this.initTokenizer();

        const ep = this.provider === "webnn"
            ? { name: "webnn", deviceType: this.deviceType }
            : this.provider;

        for (let i = 0; i < cfg.models.length; i++) {
            const { name, file } = cfg.models[i];
            const url = baseUrl + file;
            const cacheKey = `${this.modelId}_${name}`;

            onProgress &&
                onProgress({ stage: "download", model: name, index: i, total, loaded: 0, totalBytes: 0 });

            const buffer = await fetchWithCache(url, cacheKey, ({ loaded, totalBytes, cached }) => {
                onProgress &&
                    onProgress({ stage: cached ? "cache" : "download", model: name, index: i, total, loaded, totalBytes });
            });

            const modelOpt = cfg.modelOptions?.[name] || {};
            const baseOpt = cfg.sessionOptions || {};
            const ortOptions = {
                executionProviders: [ep],
                logSeverityLevel: 3,
                ...baseOpt,
                ...modelOpt,
            };
            onProgress && onProgress({ stage: "compile", model: name, index: i, total });
            this.sessions[name] = await ort.InferenceSession.create(buffer, ortOptions);
        }
    }

    // ── Text encoding ─────────────────────────────────────────────────────────

    async _tokenize(text, tokenizer) {
        const { input_ids } = await tokenizer(text, { padding: "max_length", maxLength: 77, truncation: true });
        return Array.from(input_ids.data, Number);
    }

    async _encodeText(text, tokenizer, sessionName) {
        const { input_ids } = await tokenizer(text, { padding: "max_length", maxLength: 77, truncation: true });
        const ids = Array.from(input_ids.data, Number);
        const inputTensor = new ort.Tensor("int32", new Int32Array(ids), [1, 77]);
        const result = await this.sessions[sessionName].run({ input_ids: inputTensor });
        return result[Object.keys(result)[0]];
    }

    async _encodeTextInt64(text, tokenizer, sessionName) {
        const { input_ids } = await tokenizer(text, { padding: "max_length", maxLength: 77, truncation: true });
        const ids = Array.from(input_ids.data, BigInt);
        const inputTensor = new ort.Tensor("int64", new BigInt64Array(ids), [1, 77]);
        return await this.sessions[sessionName].run({ input_ids: inputTensor });
    }

    // ── Generate ──────────────────────────────────────────────────────────────

    /**
     * Generate an image from a text prompt.
     * @param {string} prompt
     * @param {object} options
     * @param {number} [options.seed] - random seed (default: random)
     * @param {function} [options.onProgress] - callback({ step, totalSteps })
     * @returns {Promise<{ imageData: ImageData, timing: object }>}
     */
    async generate(prompt, { seed, steps, onProgress } = {}) {
        const timing = {};
        const t0 = performance.now();

        if (this.modelId === "sd-turbo") {
            return await this._generateSdTurbo(prompt, seed, onProgress, timing, t0);
        } else if (this.modelId === "sdxl-turbo") {
            return await this._generateSdxlTurbo(prompt, seed, onProgress, timing, t0);
        } else {
            return await this._generateSd15(prompt, seed, steps, onProgress, timing, t0);
        }
    }

    // ── SD Turbo ──────────────────────────────────────────────────────────────

    async _generateSdTurbo(prompt, seed, onProgress, timing, t0) {
        const { width, height } = this.config;
        const latentW = width / 8, latentH = height / 8, latentC = 4;
        const latentSize = latentC * latentH * latentW;

        // Text encode
        const t1 = performance.now();
        const embedResult = await this._encodeText(prompt, this.tokenizer, "text_encoder");
        timing.textEncode = performance.now() - t1;

        // Random latents in float32 (matching original demo)
        const sigma = 14.6146;
        const rng = makePrng(seed !== undefined ? seed : Math.floor(Math.random() * 2 ** 32));
        const latentF32 = new Float32Array(latentSize);
        for (let i = 0; i < latentSize; i++) latentF32[i] = rng() * sigma;

        // Scale model inputs: latent / sqrt(sigma^2 + 1)
        const scale = Math.sqrt(sigma * sigma + 1);
        const scaledF16 = f16Array(Array.from(latentF32, v => v / scale));
        const sampleTensor = new ort.Tensor("float16", scaledF16, [1, latentC, latentH, latentW]);

        // Timestep
        const timestep = new ort.Tensor("float16", new Uint16Array([f16(999.0)]), [1]);

        onProgress && onProgress({ step: 1, totalSteps: 1 });

        // UNet
        const t2 = performance.now();
        const unetOut = await this.sessions.unet.run({
            sample: sampleTensor,
            timestep,
            encoder_hidden_states: embedResult,
        });
        const noiseOutTensor = unetOut["out_sample"] || unetOut[Object.keys(unetOut)[0]];
        const noiseOutData = noiseOutTensor.data;
        timing.unet = performance.now() - t2;

        // Convert UNet output to float32
        const noiseF32 = new Float32Array(latentSize);
        for (let i = 0; i < latentSize; i++) noiseF32[i] = readF16(noiseOutData, i);

        // EulerA 1-step: prev = (sample - sigma * noise) / vae_scale
        // Uses original unscaled float32 latents (matching original demo)
        const vaeScale = 0.18215;
        const denoised = new Float32Array(latentSize);
        for (let i = 0; i < latentSize; i++) {
            denoised[i] = (latentF32[i] - sigma * noiseF32[i]) / vaeScale;
        }
        const denoisedTensor = new ort.Tensor("float32", denoised, [1, latentC, latentH, latentW]);

        // VAE decode
        const t3 = performance.now();
        const { sample } = await this.sessions.vae_decoder.run({ latent_sample: denoisedTensor });
        timing.vaeDecode = performance.now() - t3;

        const imageData = _vaeOutputToImageData(sample.data, width, height);
        timing.total = performance.now() - t0;
        return { imageData, timing };
    }

    // ── SDXL Turbo ────────────────────────────────────────────────────────────

    async _generateSdxlTurbo(prompt, seed, onProgress, timing, t0) {
        const { width, height } = this.config;
        const latentW = width / 8, latentH = height / 8, latentC = 4;
        const latentSize = latentC * latentH * latentW;

        // Text encoders (run sequentially — WebNN doesn't support concurrent dispatch)
        const t1 = performance.now();
        const te1Result = await this._encodeText(prompt, this.tokenizer, "text_encoder_1");
        const te2Result = await this._encodeTextInt64(prompt, this.tokenizer2, "text_encoder_2");
        timing.textEncode = performance.now() - t1;

        // Copy TE output data to fresh CPU tensors (WebNN output tensors can't be reused as inputs)
        // Use asUint16 to get raw bit patterns (handles both Float16Array and Uint16Array)
        const te1Bits = asUint16(te1Result.data).slice(); // TE1 [1,77,768]
        const te2HiddenBits = asUint16(te2Result["hidden_states.31"].data).slice(); // TE2 [1,77,1280]
        const te2EmbedsBits = asUint16(te2Result["text_embeds"].data).slice(); // TE2 [1,1280]

        // Concatenate TE1 [1,77,768] + TE2 [1,77,1280] → [1,77,2048]
        const seqLen = 77;
        const concatData = new Uint16Array(seqLen * 2048);
        for (let s = 0; s < seqLen; s++) {
            for (let d = 0; d < 768; d++) concatData[s * 2048 + d] = te1Bits[s * 768 + d];
            for (let d = 0; d < 1280; d++) concatData[s * 2048 + 768 + d] = te2HiddenBits[s * 1280 + d];
        }
        const encoderHiddenStates = new ort.Tensor("float16", concatData, [1, seqLen, 2048]);

        // text_embeds from TE2 [1, 1280]
        const textEmbeds = new ort.Tensor("float16", te2EmbedsBits, [1, 1280]);

        // time_ids [1, 6]: h,w,crop_top,crop_left,target_h,target_w
        const timeIds = new ort.Tensor(
            "float16",
            f16Array([height, width, 0, 0, height, width]),
            [1, 6],
        );

        // Random latents in float32, then scale for UNet input
        const sigma = 14.6146;
        const rng = makePrng(seed !== undefined ? seed : Math.floor(Math.random() * 2 ** 32));
        const latentF32 = new Float32Array(latentSize);
        for (let i = 0; i < latentSize; i++) latentF32[i] = rng() * sigma;

        // Scale model inputs: latent / sqrt(sigma^2 + 1)
        const scale = Math.sqrt(sigma * sigma + 1);
        const scaledF16 = f16Array(Array.from(latentF32, v => v / scale));
        const sampleTensor = new ort.Tensor("float16", scaledF16, [1, latentC, latentH, latentW]);

        // Timestep
        const timestep = new ort.Tensor("float16", new Uint16Array([f16(999.0)]), [1]);

        onProgress && onProgress({ step: 1, totalSteps: 1 });

        // UNet
        const t2 = performance.now();
        const unetOut = await this.sessions.unet.run({
            sample: sampleTensor,
            timestep,
            encoder_hidden_states: encoderHiddenStates,
            text_embeds: textEmbeds,
            time_ids: timeIds,
        });
        const noiseOutTensor = unetOut["out_sample"] || unetOut[Object.keys(unetOut)[0]];
        const noiseData = noiseOutTensor.data;
        timing.unet = performance.now() - t2;

        // EulerA 1-step: prev = (sample - sigma * noise) / vae_scale
        // Uses original unscaled latents, output as float16 for SDXL VAE
        const vaeScale = 0.18215;
        const denoised = f16Array(new Array(latentSize));
        for (let i = 0; i < latentSize; i++) {
            const nf = readF16(noiseData, i);
            denoised[i] = _hasFloat16 ? (latentF32[i] - sigma * nf) / vaeScale : f16((latentF32[i] - sigma * nf) / vaeScale);
        }
        const denoisedTensor = new ort.Tensor("float16", denoised, [1, latentC, latentH, latentW]);

        // VAE decode
        const t3 = performance.now();
        const { sample } = await this.sessions.vae_decoder.run({ latent_sample: denoisedTensor });
        timing.vaeDecode = performance.now() - t3;

        const imageData = _vaeOutputToImageData(sample.data, width, height);
        timing.total = performance.now() - t0;
        return { imageData, timing };
    }

    // ── Stable Diffusion 1.5 ──────────────────────────────────────────────────

    async _generateSd15(prompt, seed, stepsOverride, onProgress, timing, t0) {
        const { width, height } = this.config;
        const latentW = width / 8, latentH = height / 8, latentC = 4;
        const latentSize = latentC * latentH * latentW;
        const guidanceScale = 7.5;
        const steps = stepsOverride || this.config.steps;
        const schedule = SD15_SCHEDULES[steps];
        if (!schedule) throw new Error(`No schedule for ${steps} steps. Available: ${Object.keys(SD15_SCHEDULES).join(", ")}`);

        // Text encode: batch=2 (positive + negative/unconditional)
        const t1 = performance.now();
        const posTokens = await this._tokenize(prompt, this.tokenizer);
        const negTokens = await this._tokenize("", this.tokenizer);
        // Concatenate as [positive, negative] in a single batch=2 tensor
        const batchTokens = new Int32Array([...posTokens, ...negTokens]);
        const inputTensor = new ort.Tensor("int32", batchTokens, [2, 77]);
        const teResult = await this.sessions.text_encoder.run({ input_ids: inputTensor });
        const teOutput = teResult[Object.keys(teResult)[0]]; // last_hidden_state [2,77,768]
        // Copy output data into a fresh tensor (output tensors may not be reusable as inputs)
        const hiddenStates = new ort.Tensor("float16", asUint16(teOutput.data).slice(), teOutput.dims);
        timing.textEncode = performance.now() - t1;

        // Random latents
        const rng = makePrng(seed !== undefined ? seed : Math.floor(Math.random() * 2 ** 32));
        let latents = new Float32Array(latentSize);
        for (let i = 0; i < latentSize; i++) latents[i] = rng() * schedule.sigmas[0];

        // Denoising loop
        timing.unet = 0;
        for (let step = 0; step < steps; step++) {
            const sigma = schedule.sigmas[step];
            const sigmaNext = schedule.sigmas[step + 1];
            const timestep = schedule.timesteps[step];

            // Scale latents: input_latent = latent / sqrt(sigma^2 + 1)
            const scale = Math.sqrt(sigma * sigma + 1);
            const scaledData = f16Array(Array.from(latents, v => v / scale));
            // Duplicate for batch=2 (use raw uint16 bits for concatenation)
            const scaledBits = asUint16(scaledData);
            const batchData = new Uint16Array(scaledBits.length * 2);
            batchData.set(scaledBits, 0);
            batchData.set(scaledBits, scaledBits.length);
            const sampleTensor = new ort.Tensor("float16", batchData, [2, latentC, latentH, latentW]);

            const timestepValue = BigInt(Math.round(timestep));
            const timestepTensor = new ort.Tensor(
                "int64",
                new BigInt64Array([timestepValue, timestepValue]),
                [2],
            );

            const t2 = performance.now();
            const unetOut = await this.sessions.unet.run({
                sample: sampleTensor,
                timestep: timestepTensor,
                encoder_hidden_states: hiddenStates,
            });
            timing.unet += performance.now() - t2;

            const outTensor = unetOut["out_sample"] || unetOut[Object.keys(unetOut)[0]];
            // Always get properly typed data — avoid cpuData.buffer offset bugs
            const noise = outTensor.data;
            const half = noise.length / 2;

            // CFG: noise = pos * guidance + neg * (1 - guidance)
            // EulerA step: latent = latent + noise * dt
            const dt = sigmaNext - sigma;
            for (let i = 0; i < latentSize; i++) {
                const posN = readF16(noise, i);
                const negN = readF16(noise, half + i);
                const noisePred = posN * guidanceScale + negN * (1 - guidanceScale);
                latents[i] += noisePred * dt;
            }

            onProgress && onProgress({ step: step + 1, totalSteps: steps });
        }

        // Apply VAE scale factor and decode
        const t3 = performance.now();
        const vaeInput = f16Array(Array.from(latents, v => v / 0.18215));
        const vaeInputTensor = new ort.Tensor("float16", vaeInput, [1, latentC, latentH, latentW]);
        const { sample } = await this.sessions.vae_decoder.run({ latent_sample: vaeInputTensor });
        timing.vaeDecode = performance.now() - t3;

        const imageData = _vaeOutputToImageData(sample.data, width, height);
        timing.total = performance.now() - t0;
        return { imageData, timing };
    }

    async release() {
        for (const session of Object.values(this.sessions)) {
            try { await session.release(); } catch {}
        }
        this.sessions = {};
    }
}

// ─── Module-level helpers ──────────────────────────────────────────────────────

function _f16toF32(bits) {
    const fraction = bits & 0x03ff;
    const exponent = (bits & 0x7c00) >> 10;
    const sign = bits >> 15 ? -1 : 1;
    if (exponent === 0x1f) return sign * (fraction ? NaN : Infinity);
    if (exponent === 0) return sign * 6.103515625e-5 * (fraction / 0x400);
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 0x400);
}

function _vaeOutputToImageData(pixels, width, height) {
    const imageData = new ImageData(width, height);
    const H = height, W = width;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const idx = y * W + x;
            const r = readF16(pixels, 0 * H * W + idx);
            const g = readF16(pixels, 1 * H * W + idx);
            const b = readF16(pixels, 2 * H * W + idx);
            const base = (y * W + x) * 4;
            imageData.data[base + 0] = Math.min(255, Math.max(0, Math.round((r / 2 + 0.5) * 255)));
            imageData.data[base + 1] = Math.min(255, Math.max(0, Math.round((g / 2 + 0.5) * 255)));
            imageData.data[base + 2] = Math.min(255, Math.max(0, Math.round((b / 2 + 0.5) * 255)));
            imageData.data[base + 3] = 255;
        }
    }
    return imageData;
}
