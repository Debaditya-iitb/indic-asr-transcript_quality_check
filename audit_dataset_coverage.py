#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_dataset_coverage.py
=========================

Answers two questions about a set of Kaldi-format SPRING-INX corpora, before
any GPU time is spent on them:

  1. How many utterances can `ctc_forced_alignment_qc.py` actually verify?
  2. How many distinct speakers do those utterances come from?

VERIFIABLE is not "rows in text". It is the checker's own precondition, taken
straight from its run_batch():

    utt_id present in `text` AND its transcript is non-empty
    AND utt_id present in `segments`      (segmented layout)
        or utt_id present in `wav.scp`    (one-file-per-utterance layout)
    AND that recording is in `wav.scp`
    AND the audio file actually exists on disk

The last condition matters: the raw SPRING-INX releases ship `wav.scp` paths
that are RELATIVE to the original download root, e.g.

    be_CRESC_C_r001_s001  downloads/SPRING_INX/SPRING_INX_Bengali_R1//Audio/be_CRESC_C_r001_s001.wav

Those do not resolve from the corpus directory, and the checker does not accept
Kaldi pipe commands, so a QC run against an unrewritten corpus fails on 100 %
of utterances. This script resolves each entry by basename against the corpus's
own Audio/ tree and reports how many resolve, so the gap is visible.

SPEAKERS: do not trust `utt2spk` here. In every one of these corpora it is
degenerate — `utt_id utt_id`, i.e. one "speaker" per utterance. Speaker
identity, where it exists at all, is encoded in the recording id:

    001_Nagpur_M_34_monologue_00012  ->  speaker 001, Nagpur, male, age 34
    bn_IN_12_3_Left                  ->  one channel of a 2-party recording
    be_SHA1P_C2_r020_s002            ->  vendor batch r020, session s002
                                         *** NO SPEAKER FIELD AT ALL ***

The `r` field is a collection round, not a person: Bengali R2 has 1,060
recordings spanning only 6 distinct `r` values. Grouping by it would report 15
"speakers" for 1,060 recordings. So this script counts the three families
separately and never invents a speaker id that the corpus does not contain.

Usage
-----
    python audit_dataset_coverage.py --root /path/to/corpora
    python audit_dataset_coverage.py --root /path/to/corpora --out audit.csv
    python audit_dataset_coverage.py --root /path/to/corpora --prefix SPRING_INX
"""

import argparse
import collections
import csv
import os
import re
import sys

# recording-id families ------------------------------------------------------
RE_VENDOR = re.compile(r'^([a-z]{2})_([A-Za-z0-9]+)_([A-Z]+\d*)_r(\d+)_[sS](\d+)$')
RE_IN     = re.compile(r'^([a-z]{2})_IN_(\d+)_(\d+)_(Left|Right)$')
RE_MONO   = re.compile(r'^(\d+)_([A-Za-z]+)_([A-Z]{1,2})_(\d+)_')
RE_BAND   = re.compile(r'_(WB|NB)$')      # wideband / narrowband of one recording


def classify(recording_id):
    """-> (family, speaker_id or None, base_recording_id)"""
    base = RE_BAND.sub('', recording_id)
    m = RE_MONO.match(base)
    if m:
        return "monologue", "_".join(m.groups()), base
    if RE_IN.match(base):
        return "stereo_channel", base, base
    if RE_VENDOR.match(base):
        return "vendor", None, base
    return "unknown", None, base


def read_two_column(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(maxsplit=1)
            if parts:
                out[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return out


def read_segments(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.split()
            if len(p) == 4:
                out[p[0]] = (p[1], float(p[2]), float(p[3]))
    return out


def index_audio(corpus_dir):
    """basename -> path, for every wav under the corpus."""
    idx = {}
    for sub in ("Audio", "audio", "audio_16k", "wav"):
        top = os.path.join(corpus_dir, sub)
        if not os.path.isdir(top):
            continue
        for dirpath, _, files in os.walk(top):
            for fn in files:
                if fn.lower().endswith((".wav", ".flac")):
                    idx.setdefault(fn, os.path.join(dirpath, fn))
    return idx


def audit_split(corpus_dir, split, audio_index):
    sd = os.path.join(corpus_dir, split)
    text = read_two_column(os.path.join(sd, "text"))
    if not text:
        return None
    wav = read_two_column(os.path.join(sd, "wav.scp"))
    seg = read_segments(os.path.join(sd, "segments"))
    utt2spk = read_two_column(os.path.join(sd, "utt2spk"))
    utt2dur = {}
    for k, v in read_two_column(os.path.join(sd, "utt2dur")).items():
        try:
            utt2dur[k] = float(v)
        except ValueError:
            pass

    resolved, unresolved, path_as_given = set(), 0, 0
    for rec, entry in wav.items():
        cand = entry.split()[0] if entry else ""
        if cand and not cand.endswith("|") and os.path.exists(cand):
            resolved.add(rec); path_as_given += 1
        elif audio_index.get(os.path.basename(cand) if cand else rec + ".wav") \
                or audio_index.get(rec + ".wav"):
            resolved.add(rec)
        else:
            unresolved += 1

    layout = "segmented" if seg else "one-file-per-utterance"
    stats = collections.Counter()
    recordings = set()
    mono_speakers, stereo_channels = set(), set()
    families = collections.Counter()
    seconds = 0.0

    for utt, transcript in text.items():
        if layout == "segmented":
            if utt not in seg:
                stats["no_segment"] += 1
                continue
            rec = seg[utt][0]
        else:
            rec = utt
        if rec not in wav:
            stats["not_in_wav_scp"] += 1
            continue
        if rec not in resolved:
            stats["audio_file_missing"] += 1
            continue
        if not transcript.strip():
            stats["empty_transcript"] += 1
            continue

        stats["verifiable"] += 1
        family, spk, base = classify(rec)
        families[family] += 1
        recordings.add(base)
        if family == "monologue" and spk:
            mono_speakers.add(spk)
        elif family == "stereo_channel" and spk:
            stereo_channels.add(spk)
        if layout == "segmented":
            seconds += max(0.0, seg[utt][2] - seg[utt][1])
        elif utt in utt2dur:
            seconds += utt2dur[utt]

    bands = collections.Counter(m.group(1) for m in
                                (RE_BAND.search(r) for r in wav) if m)
    dup_band = len(wav) - len({RE_BAND.sub('', r) for r in wav})

    return {
        "corpus": os.path.basename(corpus_dir.rstrip("/")),
        "split": split,
        "layout": layout,
        "utts_in_text": len(text),
        "segments": len(seg),
        "recordings_in_wav_scp": len(wav),
        "recordings_resolved": len(resolved),
        "recordings_path_as_given": path_as_given,
        "recordings_unresolved": unresolved,
        "verifiable_utts": stats["verifiable"],
        "unique_recordings": len(recordings),
        "unique_monologue_speakers": len(mono_speakers),
        "unique_stereo_channels": len(stereo_channels),
        "utts_with_no_speaker_label": families["vendor"] + families["unknown"],
        "fam_monologue": families["monologue"],
        "fam_stereo_channel": families["stereo_channel"],
        "fam_vendor": families["vendor"],
        "fam_unknown": families["unknown"],
        "hours": round(seconds / 3600.0, 2),
        "lost_no_segment": stats["no_segment"],
        "lost_not_in_wav_scp": stats["not_in_wav_scp"],
        "lost_audio_missing": stats["audio_file_missing"],
        "lost_empty_transcript": stats["empty_transcript"],
        "utt2spk_degenerate": sum(1 for u, s in utt2spk.items() if u == s),
        "utt2spk_rows": len(utt2spk),
        "band_WB": bands.get("WB", 0),
        "band_NB": bands.get("NB", 0),
        "band_duplicate_recordings": dup_band,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="directory holding the corpora")
    ap.add_argument("--prefix", default="SPRING_INX",
                    help="only audit corpora whose name starts with this")
    ap.add_argument("--splits", default="train,dev,eval")
    ap.add_argument("--out", default="dataset_audit.csv")
    args = ap.parse_args()

    corpora = sorted(d for d in os.listdir(args.root)
                     if d.startswith(args.prefix)
                     and os.path.isdir(os.path.join(args.root, d)))
    if not corpora:
        sys.exit(f"no corpora starting with {args.prefix!r} under {args.root}")

    rows = []
    for name in corpora:
        cdir = os.path.join(args.root, name)
        audio_index = index_audio(cdir)
        if not audio_index:                       # cleaned dirs point at the raw one
            raw = re.sub(r'_(cleaned|final)$', '', name)
            audio_index = index_audio(os.path.join(args.root, raw))
        for split in args.splits.split(","):
            if os.path.isdir(os.path.join(cdir, split)):
                r = audit_split(cdir, split, audio_index)
                if r:
                    rows.append(r)
                    print(f"[ok] {name}/{split}: {r['verifiable_utts']:,} verifiable",
                          file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Totals cover the ORIGINAL corpora only. *_cleaned / *_final are derived
    # copies of the same audio; adding them would double-count.
    raw = [r for r in rows if not re.search(r'_(cleaned|final)$', r["corpus"])]
    print()
    print(f"{'corpus':<32}{'split':<7}{'verifiable':>12}{'recs':>8}"
          f"{'mono_spk':>9}{'chan':>6}{'hours':>9}")
    print("-" * 83)
    for r in rows:
        tag = "  (derived copy)" if r not in raw else ""
        print(f"{r['corpus']:<32}{r['split']:<7}{r['verifiable_utts']:>12,}"
              f"{r['unique_recordings']:>8,}{r['unique_monologue_speakers']:>9,}"
              f"{r['unique_stereo_channels']:>6,}{r['hours']:>9.1f}{tag}")
    print("-" * 83)
    print(f"{'TOTAL (original corpora only)':<39}"
          f"{sum(r['verifiable_utts'] for r in raw):>12,}{'':>8}{'':>9}{'':>6}"
          f"{sum(r['hours'] for r in raw):>9.1f}")
    print("Speaker columns are per-split distinct counts and are NOT summable "
          "across splits.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
