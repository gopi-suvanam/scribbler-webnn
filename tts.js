/**
 * tts.js - WebGPU Text-to-Speech Pipeline using Kokoro TTS
 *
 * Usage:
 *   import { TTSPipeline, AVAILABLE_VOICES, DTYPE_OPTIONS, DEVICE_OPTIONS } from './tts.js';
 *   const pipeline = new TTSPipeline({ dtype: 'fp32', device: 'webgpu' });
 *   await pipeline.load(progress => console.log(progress));
 *   const audio = await pipeline.generate('Hello world', { voice: 'bf_alice' });
 *   audio.play();
 */

const KOKORO_CDN = "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.0/dist/kokoro.web.js";
const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";

export const DTYPE_OPTIONS = ["fp32", "q4"];

export const DEVICE_OPTIONS = ["webgpu", "wasm"];

export const AVAILABLE_VOICES = {
    af_heart: "Heart (Female)",
    af_alloy: "Alloy (Female)",
    af_aoede: "Aoede (Female)",
    af_bella: "Bella (Female)",
    af_jessica: "Jessica (Female)",
    af_kore: "Kore (Female)",
    af_nicole: "Nicole (Female)",
    af_nova: "Nova (Female)",
    af_river: "River (Female)",
    af_sarah: "Sarah (Female)",
    af_sky: "Sky (Female)",
    am_adam: "Adam (Male)",
    am_echo: "Echo (Male)",
    am_eric: "Eric (Male)",
    am_liam: "Liam (Male)",
    am_michael: "Michael (Male)",
    am_onyx: "Onyx (Male)",
    bf_alice: "Alice (British Female)",
    bf_emma: "Emma (British Female)",
    bf_isabella: "Isabella (British Female)",
    bf_lily: "Lily (British Female)",
    bm_daniel: "Daniel (British Male)",
    bm_fable: "Fable (British Male)",
    bm_george: "George (British Male)",
    bm_lewis: "Lewis (British Male)",
};

export class TTSPipeline {
    constructor(options = {}) {
        this.dtype = options.dtype || "fp32";
        this.device = options.device || "webgpu";
        this.tts = null;
        this._KokoroTTS = null;
    }

    async load(onProgress) {
        if (onProgress) onProgress({ stage: "import", message: "Loading Kokoro library..." });

        const { KokoroTTS } = await import(KOKORO_CDN);
        this._KokoroTTS = KokoroTTS;

        if (onProgress) onProgress({ stage: "download", message: `Loading model (${this.dtype})...` });

        this.tts = await KokoroTTS.from_pretrained(MODEL_ID, {
            dtype: this.dtype,
            device: this.device,
        });

        if (onProgress) onProgress({ stage: "ready", message: "Model ready." });
    }

    listVoices() {
        if (!this.tts) return Object.keys(AVAILABLE_VOICES);
        try {
            return this.tts.list_voices();
        } catch {
            return Object.keys(AVAILABLE_VOICES);
        }
    }

    async generate(text, options = {}) {
        if (!this.tts) throw new Error("Model not loaded. Call load() first.");
        const voice = options.voice || "bf_alice";

        const t0 = performance.now();
        const audio = await this.tts.generate(text, { voice });
        const elapsed = performance.now() - t0;

        const wavBytes = audio.toWav();
        const blob = new Blob([wavBytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);

        return {
            blob,
            url,
            duration: audio.audio.length / audio.sampling_rate,
            generationTime: elapsed,
            play() {
                const el = new Audio(url);
                el.play();
                return el;
            },
        };
    }

    async release() {
        this.tts = null;
        this._KokoroTTS = null;
    }
}
