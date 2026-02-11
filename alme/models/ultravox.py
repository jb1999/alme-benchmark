"""Ultravox v0.6 model adapter (local GPU).

Optimized for batched inference with flash attention 2 and bfloat16.
Loads model and processor separately (not via pipeline) to enable
proper batched generation with left-padding.
"""

import signal
import time
from typing import Optional, List

import torch


class _GenerationTimeout(Exception):
    """Raised when model.generate() exceeds the wall-clock timeout."""
    pass


def _timeout_handler(signum, frame):
    raise _GenerationTimeout("model.generate() timed out")

from .base import ModelAdapter, ModelResponse
from .prompts import build_user_prompt


class UltravoxAdapter(ModelAdapter):
    """Adapter for Ultravox v0.6 (local GPU).

    Loads via pipeline (to get custom chat template + processor configured),
    then extracts tokenizer, processor, and model for direct batched generation.
    Enables flash attention 2 when available.
    """

    DEFAULT_MODEL = "fixie-ai/ultravox-v0_6-llama-3_1-8b"

    def __init__(self, model: str = ""):
        self._model_name = model or self.DEFAULT_MODEL
        self._model = None
        self._tokenizer = None
        self._processor = None

    @property
    def model_id(self) -> str:
        return "ultravox"

    def supports_mode(self, mode: str) -> bool:
        return mode in (
            "audio_only", "text_only",
            "audio_text_aligned", "audio_text_conflict",
        )

    @property
    def sampling_rate(self) -> int:
        if self._processor is None:
            self.load()
        # Check the processor's audio sub-processor for sampling_rate
        for attr in ("audio_processor", "feature_extractor"):
            sub = getattr(self._processor, attr, None)
            if sub is not None:
                sr = getattr(sub, "sampling_rate", None)
                if sr is not None:
                    return sr
        return 16000

    def load(self) -> None:
        if self._model is not None:
            return

        import transformers
        import transformers.modeling_utils as _mu

        # The Ultravox HF custom code reads _init_weights to choose
        # between from_pretrained and from_config for sub-models.
        # The flag was removed in transformers >=4.46.  Restore it
        # so the custom code doesn't crash with AttributeError.
        if not hasattr(_mu, "_init_weights"):
            _mu._init_weights = True

        print(f"Loading Ultravox ({self._model_name})...")
        start = time.time()

        model_kwargs = {}

        # Use 8-bit quantization if bitsandbytes is available
        try:
            import bitsandbytes  # noqa: F401
            model_kwargs["load_in_8bit"] = True
            # FA2 + 8-bit causes segfaults (memory corruption), use SDPA
            print("  8-bit quantization enabled (SDPA attention)")
        except ImportError:
            # FA2 only safe without quantization
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("  Flash Attention 2 enabled")
            except ImportError:
                print("  Using default attention")
            print("  bitsandbytes not available, using bfloat16")

        # Load via pipeline to get the custom chat template + processor
        # properly initialized, then extract components for direct batched
        # generation.  We cannot use AutoProcessor alone because the
        # Ultravox custom code configures the chat template only during
        # the pipeline's __init__.
        pipe = transformers.pipeline(
            model=self._model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_kwargs=model_kwargs,
        )
        self._model = pipe.model
        self._model.eval()

        # tokenizer — has chat template configured by pipeline init
        self._tokenizer = pipe.tokenizer

        # processor — handles text+audio → model inputs (input_ids,
        # attention_mask, audio_values, etc.).  The pipeline stores this
        # as feature_extractor or processor depending on transformers version.
        self._processor = getattr(pipe, "feature_extractor", None)
        if self._processor is None:
            self._processor = getattr(pipe, "processor", None)
        if self._processor is None:
            raise RuntimeError(
                "Could not find audio processor in Ultravox pipeline. "
                "Check pipeline attributes."
            )

        print(f"Model loaded in {time.time() - start:.1f}s")

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        del self._processor
        self._model = None
        self._tokenizer = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _infer_condition(self, audio_path, text):
        """Infer condition from audio_path/text presence (legacy fallback)."""
        if audio_path and text:
            return "audio_text_conflict"
        if audio_path:
            return "audio_only"
        return "text_only"

    def _build_conversation(self, audio_path, text, question, choices,
                            system_prompt, condition=""):
        """Build conversation and load audio for a single stimulus.

        Ultravox expects a single <|audio|> pseudo-token in the user
        message text.  The processor then replaces it with audio
        embeddings.  We must NOT use the multi-modal content format
        ({"type": "audio", ...}) because the chat template would
        insert its own placeholder, resulting in duplicates.
        """
        import librosa

        if not condition:
            condition = self._infer_condition(audio_path, text)

        audio = None
        prompt = build_user_prompt(condition, text, question, choices)

        if audio_path:
            audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
            user_text = f"<|audio|>\n{prompt}"
        else:
            user_text = prompt

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return conversation, audio

    def _process_single(self, conversation, audio=None):
        """Process a single conversation through the processor, returning model inputs."""
        text = self._tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        if audio is not None:
            inputs = self._processor(
                text=text,
                audio=audio,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
        else:
            inputs = self._tokenizer(text=text, return_tensors="pt", padding=True)
        return inputs

    def _generate_single(self, inputs, max_tokens=150, timeout_sec=60):
        """Run generation on a single set of model inputs."""
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)
        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def run(
        self,
        audio_path: str,
        text: Optional[str],
        question: str,
        choices: List[str],
        system_prompt: str,
        condition: str = "",
    ) -> ModelResponse:
        if self._model is None:
            self.load()

        start = time.time()

        try:
            conversation, audio = self._build_conversation(
                audio_path, text, question, choices, system_prompt, condition,
            )
            inputs = self._process_single(conversation, audio)
            response = self._generate_single(inputs)
            latency_ms = int((time.time() - start) * 1000)
            return ModelResponse(raw_output=response, latency_ms=latency_ms)
        except _GenerationTimeout:
            latency_ms = int((time.time() - start) * 1000)
            print(f"    TIMEOUT: generate() hung for >{60}s, skipping stimulus")
            return ModelResponse(raw_output="", latency_ms=latency_ms, error="generation_timeout")
        except Exception as e:
            latency_ms = int((time.time() - start) * 1000)
            return ModelResponse(raw_output="", latency_ms=latency_ms, error=str(e))

    def run_batch(
        self,
        items: List[tuple],
        system_prompt: str,
        condition: str = "",
    ) -> List[ModelResponse]:
        """Run batched inference on multiple stimuli.

        Processes each item individually through the processor, then
        left-pads and stacks into a single batch for model.generate().
        Falls back to sequential processing on error.
        """
        if self._model is None:
            self.load()

        start = time.time()

        # Step 1: Process each item individually through the processor
        individual_inputs = []
        for audio_path, text, question, choices in items:
            conversation, audio = self._build_conversation(
                audio_path, text, question, choices, system_prompt, condition,
            )
            inputs = self._process_single(conversation, audio)
            individual_inputs.append(inputs)

        # Step 2: Left-pad input_ids and attention_mask
        input_lengths = [inp["input_ids"].shape[1] for inp in individual_inputs]
        max_len = max(input_lengths)

        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id or 0

        batch_input_ids = []
        batch_attention_mask = []
        for inp, seq_len in zip(individual_inputs, input_lengths):
            pad_len = max_len - seq_len
            ids = inp["input_ids"][0]
            mask = inp["attention_mask"][0]
            if pad_len > 0:
                ids = torch.cat([
                    torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids,
                ])
                mask = torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            batch_input_ids.append(ids)
            batch_attention_mask.append(mask)

        batched = {
            "input_ids": torch.stack(batch_input_ids).to(self._model.device),
            "attention_mask": torch.stack(batch_attention_mask).to(self._model.device),
        }

        # Step 3: Stack audio features if present
        # Ultravox may use "audio_values", "input_features", or similar
        audio_key = None
        for key in ("audio_values", "input_features"):
            if key in individual_inputs[0]:
                audio_key = key
                break

        if audio_key:
            # Audio features: shape is typically (1, n_mels, time) or (1, time, features)
            sample_feat = individual_inputs[0][audio_key]
            ndim = sample_feat.dim()

            if ndim == 3:
                # (1, channels, time) — pad along the time dimension
                feat_lengths = [inp[audio_key].shape[2] for inp in individual_inputs]
                max_feat = max(feat_lengths)

                batch_features = []
                batch_feat_mask = []
                for inp, fl in zip(individual_inputs, feat_lengths):
                    feat = inp[audio_key][0]  # (channels, time)
                    feat_pad = max_feat - fl
                    if feat_pad > 0:
                        feat = torch.cat([
                            feat,
                            torch.zeros(feat.shape[0], feat_pad, dtype=feat.dtype),
                        ], dim=1)
                    batch_features.append(feat)

                    # Handle feature attention mask if present
                    for mask_key in ("feature_attention_mask", "audio_attention_mask"):
                        if mask_key in inp:
                            fmask = inp[mask_key][0]
                            if feat_pad > 0:
                                fmask = torch.cat([
                                    fmask,
                                    torch.zeros(feat_pad, dtype=fmask.dtype),
                                ])
                            batch_feat_mask.append(fmask)
                            break

                batched[audio_key] = torch.stack(batch_features).to(self._model.device)
                if batch_feat_mask:
                    mask_key_used = next(
                        k for k in ("feature_attention_mask", "audio_attention_mask")
                        if k in individual_inputs[0]
                    )
                    batched[mask_key_used] = torch.stack(batch_feat_mask).to(
                        self._model.device
                    )

            elif ndim == 2:
                # (1, time) — simple 1D features, pad along time
                feat_lengths = [inp[audio_key].shape[1] for inp in individual_inputs]
                max_feat = max(feat_lengths)
                batch_features = []
                for inp, fl in zip(individual_inputs, feat_lengths):
                    feat = inp[audio_key][0]
                    feat_pad = max_feat - fl
                    if feat_pad > 0:
                        feat = torch.cat([
                            feat,
                            torch.zeros(feat_pad, dtype=feat.dtype),
                        ])
                    batch_features.append(feat)
                batched[audio_key] = torch.stack(batch_features).to(self._model.device)

        # Pass through any other model-specific keys (e.g. audio_token_start_idx)
        for key in individual_inputs[0]:
            if key not in batched and key not in ("input_ids", "attention_mask"):
                try:
                    vals = [inp[key] for inp in individual_inputs]
                    if isinstance(vals[0], torch.Tensor):
                        # Stack if all same shape, otherwise skip
                        if all(v.shape == vals[0].shape for v in vals):
                            batched[key] = torch.stack(
                                [v.squeeze(0) if v.dim() > 1 else v for v in vals]
                            ).to(self._model.device)
                except Exception:
                    pass  # Skip keys that can't be batched

        # VRAM check
        if torch.cuda.is_available():
            free_mb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / 1e6
            if free_mb < 500:
                import warnings
                warnings.warn(
                    f"Low VRAM before generate(): {free_mb:.0f}MB free with "
                    f"batch_size={len(items)}. Reduce --batch-size.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        try:
            # Timeout: 60s per item in batch, minimum 120s
            batch_timeout = max(120, 60 * len(items))
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(batch_timeout)
            try:
                with torch.no_grad():
                    outputs = self._model.generate(
                        **batched, max_new_tokens=150, do_sample=False,
                    )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            # Decode each output, stripping the per-item input prefix
            responses = []
            elapsed_ms = int((time.time() - start) * 1000)
            per_item_ms = elapsed_ms // len(items)
            n_empty = 0

            for i in range(len(items)):
                decoded = self._tokenizer.decode(
                    outputs[i][max_len:], skip_special_tokens=True,
                )
                if not decoded.strip():
                    n_empty += 1
                responses.append(
                    ModelResponse(raw_output=decoded, latency_ms=per_item_ms)
                )

            # Free batch tensors now that we have decoded strings
            del outputs, batched, individual_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Detect silent OOM: if most outputs are empty, retry sequentially
            if n_empty > len(items) * 0.5 and len(items) > 1:
                import warnings
                gpu_used = (
                    torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0
                )
                warnings.warn(
                    f"POSSIBLE SILENT OOM: {n_empty}/{len(items)} outputs are "
                    f"empty (batch_size={len(items)}, GPU peak={gpu_used:.1f}GB). "
                    f"Retrying items sequentially.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                for i in range(len(items)):
                    if not responses[i].raw_output.strip():
                        audio_path, text, question, choices = items[i]
                        fallback = self.run(
                            audio_path, text, question, choices,
                            system_prompt, condition,
                        )
                        responses[i] = fallback

            return responses

        except _GenerationTimeout:
            del batched, individual_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  Batch generate TIMED OUT, falling back to sequential")
            responses = []
            for audio_path, text, question, choices in items:
                resp = self.run(
                    audio_path, text, question, choices,
                    system_prompt, condition,
                )
                responses.append(resp)
            return responses

        except Exception as e:
            # Free all GPU tensors before sequential fallback — without
            # this, the failed batch's tensors stay in VRAM and the
            # sequential fallback OOMs immediately too.
            del batched, individual_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  Batch generate failed ({e}), falling back to sequential")
            responses = []
            for audio_path, text, question, choices in items:
                resp = self.run(
                    audio_path, text, question, choices,
                    system_prompt, condition,
                )
                responses.append(resp)
            return responses
