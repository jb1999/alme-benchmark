# ALME: Audio-LLM Modality Evaluation Benchmark

ALME evaluates how audio-LLMs handle conflicting audio and text inputs across **8 typologically diverse languages**: English, German, French, Italian, Portuguese, Arabic, Japanese, and Chinese.

The core metric is **Text Dominance Ratio (TDR)**: the proportion of conflict trials where the model follows the (incorrect) text over the (correct) audio signal.

## Quick Start

### Prerequisites

- Linux with NVIDIA GPU (12GB+ VRAM with 8-bit quantization, 24GB+ without)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Common Voice Corpus 22.0 for all 8 languages (see [INSTRUCTIONS.md](INSTRUCTIONS.md))

### Install

```bash
git clone https://github.com/jb1999/alme-benchmark.git
cd alme-benchmark
uv sync
```

### Verify audio data

```bash
uv run python scripts/verify_audio.py --cv-root /path/to/cv-corpus-22.0-2025-06-20
```

### Run evaluation

```bash
# Quick test (100 stimuli)
uv run alme-eval --cv-root /path/to/cv-corpus-22.0-2025-06-20 --max-stimuli 100

# Full evaluation (57,602 stimuli, ~24 hours on a single GPU)
uv run alme-eval --cv-root /path/to/cv-corpus-22.0-2025-06-20 --output results/ultravox.json
```

### Check results

```bash
uv run python scripts/regression_test.py --results results/ultravox_metrics.json
```

## Reference Results (Ultravox v0.6)

### Overall

| Metric | Value |
|--------|-------|
| TDR (all stimuli) | 48.8% |
| Total stimuli | 57,602 |

### TDR by Language

| Language | TDR | n |
|----------|-----|---|
| EN | 39.2% | 7,325 |
| DE | 48.5% | 7,199 |
| FR | 47.7% | 7,405 |
| IT | 40.1% | 7,292 |
| PT | 49.4% | 7,235 |
| AR | 53.8% | 6,883 |
| JA | 55.1% | 7,013 |
| ZH | 57.6% | 7,116 |

### TDR by Flip Type

| Flip Type | TDR | n |
|-----------|-----|---|
| adjective_swap | 49.1% | 14,187 |
| negation_add | 59.5% | 6,426 |
| negation_remove | 43.0% | 8,236 |
| number_swap | 47.0% | 14,082 |
| time_swap | 49.0% | 14,537 |

## TTS Resynthesis Data

Pre-synthesized Azure Neural TTS audio for all 57,602 stimuli is available as a [GitHub Release](https://github.com/jb1999/alme-benchmark/releases/tag/tts-audio-v1). This enables the TTS resynthesis experiment (replacing natural Common Voice audio with synthetic speech).

Download and extract:

```bash
# Download all languages (~7.4 GB total)
for lang in ar de en fr it ja pt zh; do
  gh release download tts-audio-v1 -p "tts_audio_${lang}.tar.gz"
done

# Extract into data/tts_audio/
mkdir -p data/tts_audio
for f in tts_audio_*.tar.gz; do
  tar xzf "$f" -C data/tts_audio/
done
```

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed TTS evaluation instructions.

## Extending with New Models

To add a new model, create a class that inherits from `alme.models.base.ModelAdapter`:

```python
from alme.models.base import ModelAdapter, ModelResponse

class MyModelAdapter(ModelAdapter):
    @property
    def model_id(self) -> str:
        return "my-model"

    def supports_mode(self, mode: str) -> bool:
        return mode in ("audio_only", "text_only",
                        "audio_text_aligned", "audio_text_conflict")

    def run(self, audio_path, text, question, choices,
            system_prompt, condition="") -> ModelResponse:
        # Your inference code here
        ...

    def load(self) -> None:
        # Load model weights
        ...

    def unload(self) -> None:
        # Free GPU memory
        ...
```

Then register it in `alme/models/__init__.py`:

```python
_REGISTRY = {
    "ultravox": ("ultravox", "UltravoxAdapter", {}),
    "my-model": ("my_model", "MyModelAdapter", {}),
}
```

## Citation

If you use ALME in your research, please cite:

```bibtex
@article{billa2026alme,
  title={When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration},
  author={Billa, Jayadev},
  journal={arXiv preprint arXiv:2602.11488},
  year={2026}
}
```

Paper: [arXiv:2602.11488](https://arxiv.org/abs/2602.11488)

## License

Apache 2.0. See [LICENSE](LICENSE).

For detailed setup instructions, troubleshooting, and advanced usage, see [INSTRUCTIONS.md](INSTRUCTIONS.md).
