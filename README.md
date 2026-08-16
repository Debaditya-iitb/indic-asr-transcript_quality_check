# Indic ASR transcript QC — CTC forced-alignment quality checker

Tooling to find **transcripts that do not match their audio** in large Indic ASR
corpora, using CTC forced alignment with the AI4Bharat
[IndicConformer-600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
multilingual model.

Built for and measured on SPRING-INX (Assamese, Bengali, Marathi) and
IIIT-H Telugu — **685,345 verifiable utterances / 1,486 hours** across six
corpora.

---

## Why not just threshold WER

The obvious way to find bad transcripts is to run ASR and flag high WER. It
does not work, because a high WER conflates three unrelated causes:

1. the reference is genuinely wrong,
2. the audio is noisy, clipped or truncated,
3. the model is simply weak on that speech.

This tool asks a sharper question instead:

> Given this audio, **how much worse is the reference than what the model would
> have said on its own?**

```
score = (forced_logprob − greedy_logprob) / n_reference_tokens
```

- **forced path** — the best alignment *constrained* to emit exactly the
  reference token sequence (`torchaudio.functional.forced_align`)
- **greedy path** — the unconstrained per-frame argmax, an upper bound on any
  path

The difference is always ≤ 0 and **self-normalises for difficulty**: noisy
audio with a correct transcript pushes both scores down, so their ratio stays
near 0. Only a genuine mismatch collapses it. That is the whole idea, and it is
why this beats thresholding the forced log-probability alone.

| score | reading |
|---|---|
| 0 to −0.5 | good match |
| −0.5 to −2 | minor mismatch |
| below −2 | major mismatch: wrong transcript, placeholder text, truncated audio |

Always calibrate on your own data — score ~50 trusted utterances, then score
the *same audio against shuffled transcripts*, and put the threshold between
the two distributions.

---

## The bug this tool exists to get right

IndicConformer-600M is a **multi-softmax** model: 22 languages × 256 tokens
**+ 1 shared blank = 5,633 outputs**, where language *L*'s tokens occupy the
contiguous block `[L_idx × 256, L_idx × 256 + 255]`.

But `MultilingualTokenizer.text_to_ids(text, lang)` returns **language-local**
ids (0–255), not global ones. A naive pipeline therefore corrupts itself
silently:

- `forced_align` receives Marathi-local id 27 and aligns it to **global**
  column 27 — which belongs to Assamese;
- the unmasked greedy argmax roams across all 22 blocks and decodes a salad of
  unrelated languages;
- every reference token lands on a wrong column, so the score floors out around
  **−14 on clean data** and *everything* is rejected.

That is not hypothetical. The pre-fix report in this project's history scored
**84,159 utterances, accepted 0, mean score −14.39**.

The fix, in `ctc_forced_alignment_qc.py`, is language-aware logit slicing:
discover the language's block offset, take its 256 columns plus the shared
blank, and re-`log_softmax` the slice so both paths operate in the same 257-dim
space. Offset discovery tries explicit tokenizer attributes first, falls back to
`lang_index × (vocab_width / n_langs)`, and **refuses to slice** rather than
slice wrongly if the geometry is ambiguous.

**`--language` is therefore mandatory in practice.** Check the startup log:

```
Language 'mr' is index 12/22 → offset = 12 × 256 = 3072
Effective vocab : 257  (blank @ 256)
```

If you see numeric token ids in the predictions, slicing failed.

---

## Scripts, in the order you would run them

| script | task |
|---|---|
| `resample_audio_to_16k.py` | multi-GPU batch resample of a corpus to 16 kHz mono |
| `audit_dataset_coverage.py` | **run this first.** How many utterances are actually checkable, and how many speakers exist — before spending GPU time |
| `run_indicconformer_on_kaldi_corpus.py` | IndicConformer inference (CTC or RNN-T) over a Kaldi corpus → per-utterance WER/CER + NeMo manifests |
| `run_indicconformer_on_folder.py` | same, for a plain `audio/` + `transcripts/` layout |
| `ctc_forced_alignment_qc.py` | **the checker.** Forced-alignment likelihood-ratio score per utterance, with per-token diagnostics and a blind SNR estimate |
| `crosscheck_transcripts_with_whisper.py` | triangulation: re-transcribe flagged clips with Whisper large-v3 and compare against *both* the reference and the IndicConformer hypothesis, to decide which one is wrong |

Full operator guide, including troubleshooting: [`docs/CTC_ALIGNMENT_GUIDE.md`](docs/CTC_ALIGNMENT_GUIDE.md).

### Running the checker

```bash
export NEMO_ROOT=/path/to/NeMo          # or pip install -e /path/to/NeMo --no-deps

python ctc_forced_alignment_qc.py \
    --model      indicconformer_600m_multi.nemo \
    --language   mr \
    --wav_scp    corpus/train/wav.scp \
    --text       corpus/train/text \
    --segments   corpus/train/segments \
    --out_dir    qc_out \
    --device     cuda:0 \
    --score_threshold -2.0 \
    --min_token_score -6.0
```

Outputs into `--out_dir`:

| file | contents |
|---|---|
| `quality_report.csv` / `.tsv` | one row per utterance: `pass`, `score`, `lr_per_frame`, `mean_token_score`, `min_token_score`, `forced_logprob`, `greedy_logprob`, `snr`, timings, `ref_tokens`, `pred_tokens`, `error` |
| `accepted_utts.txt` | utterance ids that passed |
| `rejected_utts.txt` | utterance ids that failed |

### Three axes, deliberately separate

- `score` — **is the transcript right?**
- `min_token_score` — **is one word wrong** inside an otherwise correct
  sentence? (a sentence-level average hides exactly this case)
- `snr` — **is the audio usable?** A blind WADA-SNR estimate
  ([Kim & Stern, Interspeech 2008](https://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf))
  computed *before* alignment and reported even when alignment fails.

---

## Dataset audit — [`data/dataset_audit.csv`](data/dataset_audit.csv)

`audit_dataset_coverage.py` applies the checker's *own* precondition to a corpus
and reports what is genuinely checkable. Measured on the six SPRING-INX corpora:

| corpus | verifiable utts | recordings | hours | speaker labels |
|---|---|---|---|---|
| Assamese R1 | 46,395 | 259 | 60.7 | 22 stereo channels |
| Assamese R3 | 131,262 | 1,650 | 310.5 | **none** |
| Bengali R1 | 184,027 | 1,778 | 419.8 | 326 stereo channels |
| Bengali R2 | 127,419 | 1,060 | 307.3 | **none** |
| Marathi R1 | 96,850 | 27,190 | 149.9 | **76 real speakers** + 70 channels |
| Marathi R2 | 99,392 | 1,070 | 237.6 | **none** |
| **total** | **685,345** | **33,007** | **1,485.7** | 76 real + 418 channels |

Coverage is essentially perfect once paths are fixed: zero missing segments,
zero missing `wav.scp` entries, zero missing audio files, zero empty
transcripts. The per-split CSV carries all of the loss columns so this is
auditable rather than asserted.

### Three findings that change how you use these corpora

**1. `wav.scp` in the raw releases is broken.** Every raw corpus ships paths
relative to the original download root:

```
be_CRESC_C_r001_s001  downloads/SPRING_INX/SPRING_INX_Bengali_R1//Audio/be_CRESC_C_r001_s001.wav
```

These do not resolve from the corpus directory, and the checker does not accept
Kaldi pipe commands — so a QC run against an unrewritten corpus **fails on
100 % of utterances**. Only the `*_cleaned` variants carry absolute paths. The
audit resolves by basename and reports `recordings_path_as_given` vs
`recordings_resolved` so the gap is visible before it costs you a run.

**2. There are almost no speaker labels.** `utt2spk` is degenerate in every
corpus — literally `utt_id utt_id`, one "speaker" per utterance. And the `r`
field in vendor recording ids is a *collection round*, not a person: Bengali
R2's 1,060 recordings span **6 distinct `r` values**, so grouping by it would
claim 15 speakers for 1,060 recordings. Real speaker identity exists only for:

- **76 monologue speakers** in Marathi R1, ids of the form
  `<spk>_<city>_<gender>_<age>` — 45 F / 30 M / 1 unspecified, ages **16–54**
  (mean 25.1), from Nagpur (31), unspecified (19), Pune (18), Mumbai (8);
- **418 stereo channels** (`*_Left` / `*_Right`), one speaker per channel of a
  two-party recording, with no identity persisting across recordings.

Everything else — ~93 % of the data — carries **no speaker field at all**, and
the conversational (`_C_`) recordings contain two or more speakers each. You
**cannot build speaker-disjoint splits from the shipped metadata**; that needs
diarisation or speaker embeddings.

**3. Assamese R3 has band duplicates.** Its 1,879 recordings reduce to **1,650
unique** after stripping `_WB`/`_NB`, with **280 recordings present in both
bands** — the same speech at two bandwidths. Scored twice, and a leak across
any split that ignores it. It also ships no `utt2dur`, so durations here are
derived from `segments`.

```bash
python audit_dataset_coverage.py --root /path/to/corpora --out data/dataset_audit.csv
```

---

## Requirements

| component | version |
|---|---|
| Python | 3.10+ |
| PyTorch | CUDA build |
| torchaudio | **≥ 2.1** (needs `functional.forced_align`) |
| NeMo | source tree on `NEMO_ROOT`, or editable install |
| model | `indicconformer_600m_multi.nemo` |
| GPU | strongly recommended |

```bash
pip install -r requirements.txt
export NEMO_ROOT=/path/to/NeMo
python -c "import nemo; print(nemo.__file__)"
```

---

## Limitations

- **One language per run.** The logit slice is fixed at startup.
- **CTC branch only.** The checkpoint is a hybrid RNNT-CTC; alignment uses
  `model.ctc_decoder`, never the RNN-T prediction network.
- **Real audio files only** — Kaldi pipe commands in `wav.scp` are not
  supported.
- **One utterance per forward pass.** At 84 k+ utterances this dominates
  runtime; batching is the obvious next optimisation.
- **The tool flags, it does not delete.** Nothing is modified automatically.
- `--language` defaults to `None` and only warns. Given that omitting it
  produces the silent −14 failure mode, treat it as required.

## What is not in this repo

No audio, no transcripts, no model checkpoints — the corpora are not
redistributable. `data/dataset_audit.csv` contains aggregate counts only, no
transcript text. Everything else regenerates from these scripts.
