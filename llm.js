/**
 * llm.js - WebGPU LLM Pipeline using Transformers.js
 *
 * Uses AutoTokenizer + AutoModelForCausalLM for direct model control.
 *
 * Usage:
 *   import { LLMPipeline, AVAILABLE_MODELS } from './llm.js';
 *   const pipeline = new LLMPipeline('phi-4-mini');
 *   await pipeline.load(progress => console.log(progress));
 *   const response = await pipeline.generate('What is WebNN?', {
 *       onToken: token => console.log(token),
 *   });
 */

const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.3";

export const AVAILABLE_MODELS = {
    "qwen2.5-0.5b": {
        name: "Qwen 2.5 0.5B Instruct",
        repo: "onnx-community/Qwen2.5-0.5B-Instruct",
        dtype: "q4",
        maxTokens: 512,
    },
};

export const DEVICE_OPTIONS = ["webgpu", "wasm"];

export class LLMPipeline {
    constructor(modelId, options = {}) {
        const config = AVAILABLE_MODELS[modelId];
        if (!config) throw new Error(`Unknown model: ${modelId}. Choose from: ${Object.keys(AVAILABLE_MODELS).join(", ")}`);
        this.modelId = modelId;
        this.config = config;
        this.device = options.device || "webgpu";
        this.tokenizer = null;
        this.model = null;
        this._transformers = null;
    }

    async load(onProgress) {
        if (onProgress) onProgress({ stage: "import", message: "Loading Transformers.js library..." });

        const transformers = await import(TRANSFORMERS_CDN);
        this._transformers = transformers;
        const { AutoTokenizer, AutoModelForCausalLM } = transformers;

        // Check WebGPU if selected
        if (this.device === "webgpu") {
            const adapter = await navigator.gpu?.requestAdapter();
            if (!adapter) {
                throw new Error("WebGPU is not supported (no adapter found)");
            }
        }

        if (onProgress) onProgress({ stage: "download", message: `Loading tokenizer for ${this.config.name}...` });

        this.tokenizer = await AutoTokenizer.from_pretrained(this.config.repo);

        if (onProgress) onProgress({ stage: "download", message: `Loading ${this.config.name} model...` });

        try {
            this.model = await AutoModelForCausalLM.from_pretrained(this.config.repo, {
                dtype: this.config.dtype,
                device: this.device,
                ...(onProgress && {
                    progress_callback: (p) => {
                        if (p.status === "progress" && p.total) {
                            const pct = Math.round((p.loaded / p.total) * 100);
                            onProgress({
                                stage: "download",
                                message: `Downloading ${p.file || "model"}... ${pct}%`,
                                progress: pct,
                            });
                        }
                    },
                }),
            });
        } catch (e) {
            throw new Error(`Failed to load model: ${e?.message || e}`);
        }

        if (onProgress) onProgress({ stage: "ready", message: "Model ready." });
    }

    async generate(prompt, options = {}) {
        if (!this.model || !this.tokenizer) throw new Error("Model not loaded. Call load() first.");

        const maxNewTokens = options.maxNewTokens || this.config.maxTokens;

        // Build chat messages
        const messages = Array.isArray(prompt)
            ? prompt
            : [{ role: "user", content: prompt }];

        // Apply chat template to get input token IDs
        const inputs = this.tokenizer.apply_chat_template(messages, {
            add_generation_prompt: true,
            return_dict: true,
        });

        const t0 = performance.now();

        if (options.onToken) {
            // Streaming mode with TextStreamer
            let fullText = "";
            const streamer = new this._transformers.TextStreamer(this.tokenizer, {
                skip_prompt: true,
                skip_special_tokens: true,
                callback_function: (text) => {
                    fullText += text;
                    options.onToken(text);
                },
            });

            await this.model.generate({
                ...inputs,
                max_new_tokens: maxNewTokens,
                temperature: options.temperature ?? 0,
                do_sample: (options.temperature ?? 0) > 0,
                ...(options.temperature > 0 && { top_p: options.topP ?? 0.9 }),
                streamer,
            });

            const elapsed = performance.now() - t0;
            return { text: fullText, generationTime: elapsed };
        } else {
            // Non-streaming mode
            const outputIds = await this.model.generate({
                ...inputs,
                max_new_tokens: maxNewTokens,
                temperature: options.temperature ?? 0,
                do_sample: (options.temperature ?? 0) > 0,
                ...(options.temperature > 0 && { top_p: options.topP ?? 0.9 }),
            });

            const elapsed = performance.now() - t0;

            // Decode only the newly generated tokens
            const promptLength = inputs.input_ids.dims[1];
            const newTokenIds = outputIds.slice(null, [promptLength, null]);
            const text = this.tokenizer.batch_decode(newTokenIds, { skip_special_tokens: true })[0];

            return { text, generationTime: elapsed };
        }
    }

    async release() {
        if (this.model) {
            try { await this.model.dispose(); } catch {}
        }
        this.model = null;
        this.tokenizer = null;
        this._transformers = null;
    }
}
