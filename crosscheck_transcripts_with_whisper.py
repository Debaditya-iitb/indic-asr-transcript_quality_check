"""
Analyze audio segments from wer_train_checkpoint.csv by transcribing each one
with Whisper (large-v3, Bengali) and comparing against:
  - reference  : ground-truth transcript
  - hypothesis : Indic Conformer ASR output (already in CSV)
  - whisper    : Whisper transcription from actual audio

Outputs:
  audio_analysis_results.csv  — all rows with whisper transcript + similarity scores
  audio_analysis_flagged.csv  — rows where audio disagrees with both ref and hyp
"""

import csv
import io
import math
import os
import tempfile
import unicodedata

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

CSV_IN   = "/workspace/debaditya/nemo/wer_train_checkpoint.csv"
CSV_ALL  = "/workspace/debaditya/nemo/audio_analysis_results.csv"
CSV_FLAG = "/workspace/debaditya/nemo/audio_analysis_flagged.csv"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def char_jaccard(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    a, b = a.replace(" ", ""), b.replace(" ", "")
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def word_jaccard(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def is_empty(text):
    return not isinstance(text, str) or text.strip() == "" or text.strip().lower() == "nan"

def categorize(whisper_txt, ref, hyp, wjac_wr, wjac_wh, cjac_wr, cjac_wh):
    """Assign a category based on similarity scores."""
    w_empty = is_empty(whisper_txt)
    if w_empty:
        return "WHISPER_SILENT"
    # Whisper agrees with both
    if wjac_wr >= 0.5 and wjac_wh >= 0.5:
        return "AGREE_BOTH"
    # Whisper agrees with ref (ASR error)
    if wjac_wr >= 0.5 and wjac_wh < 0.3:
        return "AGREE_REF_ONLY"
    # Whisper agrees with hyp (ref may be wrong)
    if wjac_wh >= 0.5 and wjac_wr < 0.3:
        return "AGREE_HYP_ONLY"
    # Whisper agrees with neither — audio content differs from both
    if cjac_wr < 0.3 and cjac_wh < 0.3:
        return "DISAGREE_BOTH_STRONG"
    if wjac_wr < 0.3 and wjac_wh < 0.3:
        return "DISAGREE_BOTH"
    return "PARTIAL_MATCH"


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────

print("Loading Whisper large-v3 on GPU …")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print("Model ready.\n")

# ──────────────────────────────────────────────
# Read CSV
# ──────────────────────────────────────────────

rows = []
with open(CSV_IN, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Total utterances: {len(rows)}")

# Cache open WAV files to avoid re-reading the same large file repeatedly
wav_cache = {}

def get_audio_segment(audio_path, start_sec, end_sec):
    if audio_path not in wav_cache:
        data, sr = sf.read(audio_path, dtype="float32")
        wav_cache[audio_path] = (data, sr)
    data, sr = wav_cache[audio_path]
    s = int(float(start_sec) * sr)
    e = int(float(end_sec)   * sr)
    seg = data[s:e]
    if seg.ndim > 1:           # stereo → mono
        seg = seg.mean(axis=1)
    return seg, sr

# ──────────────────────────────────────────────
# Process
# ──────────────────────────────────────────────

out_rows = []
flagged  = []

for i, row in enumerate(rows):
    utt_id     = row.get("utt_id", "")
    audio_path = row.get("audio_path", "")
    start_sec  = float(row.get("start_sec", 0))
    end_sec    = float(row.get("end_sec",   0))
    reference  = row.get("reference",  "")
    hypothesis = row.get("hypothesis", "")
    wer        = row.get("wer", "")
    cer        = row.get("cer", "")

    if i % 100 == 0:
        print(f"  [{i}/{len(rows)}] processing …")

    # ── Extract segment ──
    try:
        seg, sr = get_audio_segment(audio_path, start_sec, end_sec)
    except Exception as exc:
        whisper_txt = f"ERROR:{exc}"
        category    = "AUDIO_ERROR"
        wjac_wr = wjac_wh = cjac_wr = cjac_wh = 0.0
    else:
        # ── Transcribe with Whisper ──
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, seg, sr)
                tmp_path = tmp.name

            segments_gen, info = model.transcribe(
                tmp_path,
                language="bn",
                beam_size=5,
                vad_filter=True,
            )
            whisper_txt = " ".join(s.text for s in segments_gen).strip()
            os.unlink(tmp_path)
        except Exception as exc:
            whisper_txt = f"ERROR:{exc}"
            category    = "WHISPER_ERROR"
            wjac_wr = wjac_wh = cjac_wr = cjac_wh = 0.0
        else:
            wjac_wr = word_jaccard(whisper_txt, reference)
            wjac_wh = word_jaccard(whisper_txt, hypothesis)
            cjac_wr = char_jaccard(whisper_txt, reference)
            cjac_wh = char_jaccard(whisper_txt, hypothesis)
            category = categorize(whisper_txt, reference, hypothesis,
                                  wjac_wr, wjac_wh, cjac_wr, cjac_wh)

    out_row = {
        "utt_id":        utt_id,
        "recording_id":  row.get("recording_id", ""),
        "start_sec":     start_sec,
        "end_sec":       end_sec,
        "duration_sec":  row.get("duration_sec", ""),
        "audio_path":    audio_path,
        "reference":     reference,
        "hypothesis":    hypothesis,
        "whisper":       whisper_txt,
        "wer_orig":      wer,
        "cer_orig":      cer,
        "wjac_whisper_ref": round(wjac_wr, 4),
        "wjac_whisper_hyp": round(wjac_wh, 4),
        "cjac_whisper_ref": round(cjac_wr, 4),
        "cjac_whisper_hyp": round(cjac_wh, 4),
        "category":      category,
    }
    out_rows.append(out_row)

    if category in ("DISAGREE_BOTH", "DISAGREE_BOTH_STRONG", "AGREE_HYP_ONLY"):
        flagged.append(out_row)

# ──────────────────────────────────────────────
# Write outputs
# ──────────────────────────────────────────────

fields = list(out_rows[0].keys())

with open(CSV_ALL, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)

with open(CSV_FLAG, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(flagged)

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────

from collections import Counter
cats = Counter(r["category"] for r in out_rows)

print("\n" + "="*60)
print("AUDIO ANALYSIS SUMMARY")
print("="*60)
print(f"Total utterances analysed : {len(out_rows)}")
print()
for cat, cnt in cats.most_common():
    pct = cnt / len(out_rows) * 100
    print(f"  {cat:<28} {cnt:>5}  ({pct:.1f}%)")

print()
print(f"Flagged (mismatch) rows   : {len(flagged)}")
print(f"\nAll results → {CSV_ALL}")
print(f"Flagged     → {CSV_FLAG}")
print("\nDONE.")
