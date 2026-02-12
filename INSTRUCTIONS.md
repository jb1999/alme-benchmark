# ALME Benchmark — Detailed Instructions

## System Requirements

- **OS**: Linux (uses `signal.SIGALRM` for inference timeout)
- **GPU**: NVIDIA with 12GB+ VRAM (8-bit quantization) or 24GB+ (full precision)
- **Python**: 3.10 or later
- **Disk**: ~15GB for Common Voice audio data (8 languages)

## Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and install

```bash
git clone https://github.com/jb1999/alme-benchmark.git
cd alme-benchmark
uv sync
```

## Common Voice Data

ALME uses [Common Voice Corpus 22.0](https://commonvoice.mozilla.org/en/datasets) as its audio source. You need to download the following 8 languages:

- English (`en`)
- German (`de`)
- French (`fr`)
- Italian (`it`)
- Portuguese (`pt`)
- Arabic (`ar`)
- Japanese (`ja`)
- Chinese (China) (`zh-CN`)

### Download

1. Go to https://commonvoice.mozilla.org/en/datasets
2. Select "Common Voice Corpus 22.0" and download each language
3. Extract all archives into a single directory, preserving the structure:

```
cv-corpus-22.0-2025-06-20/
├── en/
│   └── clips/
│       ├── common_voice_en_20862902.mp3
│       └── ...
├── de/
│   └── clips/
│       └── ...
├── fr/
│   └── clips/
│       └── ...
└── ...
```

### Verify

```bash
uv run python scripts/verify_audio.py --cv-root /path/to/cv-corpus-22.0-2025-06-20
```

This checks all 57,602 audio files. Expected output:

```
Stimuli: 57602
Found:   57602
Missing: 0

All audio files found!
```

## Running Evaluation

### Environment variable (optional)

Set `ALME_CV_ROOT` to avoid passing `--cv-root` every time:

```bash
export ALME_CV_ROOT=/path/to/cv-corpus-22.0-2025-06-20
```

### Quick test

```bash
uv run alme-eval --cv-root /path/to/cv-corpus-22.0-2025-06-20 --max-stimuli 100
```

### Full evaluation

```bash
uv run alme-eval \
  --cv-root /path/to/cv-corpus-22.0-2025-06-20 \
  --output results/ultravox.json
```

This runs all 4 conditions on all 57,602 stimuli. Expected runtime: ~24 hours on a single GPU.

### Evaluation conditions

| Condition | Audio | Text | Purpose |
|-----------|-------|------|---------|
| `audio_only` | Yes | No | Baseline audio comprehension |
| `text_only` | No | Yes (original) | Text-only diagnostic |
| `audio_text_aligned` | Yes | Yes (original) | Multimodal ceiling |
| `audio_text_conflict` | Yes | Yes (modified) | Core conflict test |

### Checkpointing

Evaluation checkpoints automatically after each batch. If interrupted, re-run the same command — it resumes from where it left off.

### Single condition

```bash
uv run alme-eval \
  --cv-root /path/to/cv-corpus-22.0-2025-06-20 \
  --conditions audio_text_conflict \
  --output results/ultravox_conflict_only.json
```

## Understanding Results

### TDR (Text Dominance Rate)

The primary metric. For each conflict trial, the model either:
- **Follows audio** (correct): answers based on what it heard
- **Follows text** (incorrect): answers based on the modified transcript
- **Other**: neither answer matches

TDR = followed_text / (followed_text + followed_audio)

- TDR > 50%: model is text-dominant (trusts text over audio)
- TDR < 50%: model is audio-dominant (trusts audio over text)

### Output files

After evaluation, you'll find:
- `results/ultravox.json` — all trial results
- `results/ultravox_metrics.json` — aggregated metrics (TDR, per-language, per-flip-type)
- `results/monitoring.json` — error monitoring data
- `results/ultravox_{condition}.json` — per-condition results

## Running Statistical Analysis

The `alme.stats` module provides McNemar's test, Wilson CIs, Cohen's h, and chi-squared tests for comparing natural vs TTS audio. See the module docstring for API details.

## Regression Testing

After a full evaluation, verify results match reference values:

```bash
uv run python scripts/regression_test.py --results results/ultravox_metrics.json
```

This compares TDR overall, per-language, and per-flip-type within 2 percentage points of the reference values. All checks should PASS for a correct reproduction.

## TTS Resynthesis Experiment

The TTS resynthesis experiment replaces natural Common Voice audio with Azure Neural TTS to test whether speaker variability affects modality arbitration.

### Download TTS audio

TTS audio is hosted as a GitHub Release (~7.4 GB compressed, ~11 GB extracted):

```bash
# Requires GitHub CLI (gh)
for lang in ar de en fr it ja pt zh; do
  gh release download tts-audio-v1 -p "tts_audio_${lang}.tar.gz" -R jb1999/alme-benchmark
done

# Extract
mkdir -p data/tts_audio
for f in tts_audio_*.tar.gz; do
  tar xzf "$f" -C data/tts_audio/
done
```

After extraction, you should have:

```
data/tts_audio/
├── ar/
│   ├── tts_cv__XXXXXXXXXXXX.wav
│   └── ...
├── de/
│   └── ...
└── ...    (8 language directories, 57,602 WAV files total)
```

### Prepare TTS stimuli

The TTS WAV files mirror the Common Voice clip structure. To evaluate with TTS audio, create a stimuli file with remapped audio paths:

```bash
uv run python scripts/make_tts_stimuli.py \
  --stimuli data/stimuli.jsonl \
  --tts-audio-dir data/tts_audio \
  --output data/stimuli_tts.jsonl
```

### Run TTS evaluation

```bash
uv run alme-eval \
  --stimuli data/stimuli_tts.jsonl \
  --cv-root data/tts_audio \
  --conditions audio_only audio_text_conflict \
  --output results/ultravox_tts.json
```

Only `audio_only` and `audio_text_conflict` conditions are needed — text-only and aligned conditions are unchanged by the audio source.

## Adding New Models

1. Create `alme/models/my_model.py` implementing `ModelAdapter`
2. Register in `alme/models/__init__.py`
3. Run: `uv run alme-eval --model my-model --cv-root ...`

See `alme/models/ultravox.py` for a complete example.

## Troubleshooting

### CUDA out of memory

- 8-bit quantization is enabled by default (bitsandbytes is a core dependency)
- Reduce batch size: `--batch-size 1`

### Flash Attention errors

Flash Attention 2 is used automatically when available and NOT using 8-bit quantization.
FA2 + 8-bit causes segfaults — the code disables FA2 when bitsandbytes is loaded.

### transformers version

Ultravox requires `transformers >=4.40,<4.46`. The custom model code uses internal APIs
that break outside this range. The pinned version in `pyproject.toml` handles this.

### Generation timeout

Individual inference calls have a 60-second timeout (using `signal.SIGALRM`).
Timed-out trials are recorded as errors and retried on resume.
This mechanism requires Linux; it will not work on macOS or Windows.

### Checkpoint corruption

If a checkpoint file is corrupted, delete it and re-run:

```bash
rm -rf results/.checkpoints/
uv run alme-eval --cv-root ... --output results/ultravox.json
```
