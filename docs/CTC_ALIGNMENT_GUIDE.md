
# CTC Forced-Alignment Quality Checker — IndicConformer 600M (Simplified Guide)

## What this tool does

This script checks whether a transcript matches an audio file using CTC forced alignment with the AI4Bharat IndicConformer 600M model.

For each audio-transcript pair, it:

1. Gets frame-level CTC probabilities from the model.
2. Finds the best alignment for the reference transcript (forced path).
3. Finds the model's own best unconstrained path (greedy path).
4. Compares the two paths.

If the transcript matches the audio, the forced path score will be close to the greedy path score. If the transcript is wrong, the score becomes much lower.

The tool is useful for cleaning ASR training and evaluation datasets.



## Main scoring metrics

Primary score:


score = (forced_logprob - greedy_logprob) / L


where:
L = number of reference tokens

Additional metric:


lr_per_frame = (forced_logprob - greedy_logprob) / T


where:

T = number of audio frames

Other useful diagnostics:

- `mean_token_score` = average token quality
- `min_token_score` = score of the worst token

The worst-token score helps detect a single incorrect word inside an otherwise correct sentence.



## Pass / Fail rule


pass = score >= score_threshold


and, if enabled,


min_token_score >= min_token_score_threshold




## Important model details

### Hybrid RNNT + CTC model

The IndicConformer 600M checkpoint is a hybrid RNNT-CTC model.

Forced alignment uses:

python
model.ctc_decoder


and not the RNNT decoder.


### Language slicing is required

The model supports 22 Indic languages using one large output layer.

Each language owns a block of 256 tokens.

Examples:

- Hindi → block 1280–1535
- Marathi → block 3072–3327

The tokenizer returns local token IDs (0–255), so the script must slice the correct language block before alignment.

Because of this, the `--language` argument is required.

Example:

bash
--language mr


The startup log should show:

text
Language 'mr' is index 12/22
offset = 3072


If language slicing fails, results become unreliable.

---

### Frame timing

The script calculates frame duration using model configuration values.

For IndicConformer 600M:

text
80 ms per frame
12.5 frames per second


---

## Requirements

| Component | Requirement |
|------------|------------|
| Python | 3.10+ |
| PyTorch | CUDA-compatible build |
| torchaudio | 2.1 or newer |
| NeMo | Source tree available |
| Model | indicconformer_600m_multi.nemo |
| GPU | Recommended |
| Packages | numpy, tqdm |

---

## Environment setup

Temporary setup:

bash
export NEMO_ROOT=/workspace/debaditya/nemo/NeMo
export PYTHONPATH=/workspace/debaditya/nemo/NeMo:$PYTHONPATH


Permanent setup:
bash
pip install -e /workspace/debaditya/nemo/NeMo --no-deps --no-build-isolation


Verify:

bash
python -c "import nemo; print(nemo.__file__)"


---

## Input files

### text

Contains transcripts.

Example:

text
utt001  transcript text


### wav.scp

Maps recording IDs to audio paths.

Example:

text
rec001 /path/audio.wav


Only real audio files are supported.

Kaldi pipe commands are not supported.

### segments (optional)

Defines utterance boundaries inside recordings.

Example:

text
utt001 rec001 0.5 5.9


---

## Supported corpus layouts

### Layout A: Segmented recordings

Uses:

- text
- wav.scp
- segments

Audio is stored as long recordings and segments define utterances.

### Layout B: One audio file per utterance

Uses:

- text
- wav.scp

Each utterance has its own audio file.

---

## Running the script

Example:

```bash
python ctc_forced_alignment_indic_conformer.py \
  --model indicconformer_600m_multi.nemo \
  --wav_scp wav.scp \
  --text text \
  --segments segments \
  --out_dir output \
  --device cuda:0 \
  --language mr
```

Useful options:

| Option | Meaning |
|----------|----------|
| --model | Model path |
| --wav_scp | wav.scp file |
| --text | text file |
| --segments | segments file |
| --out_dir | Output directory |
| --device | cpu or cuda |
| --language | Language code |
| --score_threshold | Acceptance threshold |
| --min_token_score | Worst-token threshold |
| --debug | Print extra diagnostics |

---

## Output files

The output directory contains:

### quality_report.tsv

One row per utterance.

Important columns:

- utt_id
- pass
- score
- lr_per_frame
- mean_token_score
- min_token_score
- forced_logprob
- greedy_logprob
- text
- error

### accepted_utts.txt

Accepted utterance IDs.

### rejected_utts.txt

Rejected utterance IDs.

---

## Interpreting scores

Typical interpretation:

| Score | Meaning |
|---------|---------|
| 0 to -0.5 | Good match |
| -0.5 to -2 | Minor mismatch |
| Below -2 | Major mismatch |

Examples of major mismatches:

- wrong transcript
- placeholder text
- truncated audio
- severe disagreement with model prediction

Always calibrate thresholds using your own dataset.

---

## Troubleshooting

### NeMo import error

text
cannot import name '__version__' from 'nemo'


Fix:

bash
export NEMO_ROOT=...
export PYTHONPATH=...


or install NeMo in editable mode.

---

### Language slice not found

Warning:

text
Could not determine slice for language


Use the correct language code such as:

text
mr
hi


Check startup logs to verify slicing.

---

### Numeric token IDs appear in predictions

Language slicing failed.

Verify the language argument and startup logs.

---

### Audio too short

The transcript is longer than the available audio frames.

Possible causes:

- incorrect segment boundaries
- very short audio clips

---

### Missing recording in wav.scp

A recording ID referenced by `segments` is not present in `wav.scp`.

---

### CUDA out of memory

The utterance is too large for available GPU memory.

The script skips the utterance and continues.

---

## Limitations

- Supports one language per run.
- Uses only the CTC branch.
- Audio paths must be real files.
- The tool only flags suspicious utterances; it does not modify data automatically.