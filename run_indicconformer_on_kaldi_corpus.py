"""
run_indicconformer_on_kaldi_corpus.py
──────────────────────
Runs IndicConformer inference + WER/CER on a Kaldi-format dataset.

Expected folder structure:
    BASE_FOLDER/
      train/
        text          →  utt_id transcript words here
        wav.scp       →  rec_id /path/to/recording.wav   (or a pipe command ending with |)
        utt2spk       →  utt_id spk_id
        spk2utt       →  spk_id utt_id1 utt_id2 ...
        segments      →  utt_id rec_id start_sec end_sec  (OPTIONAL — only if audio is segmented)
      dev/
        (same files)
      eval/
        (same files)

Outputs (in OUTPUT_DIR):
    train_results.csv / dev_results.csv / eval_results.csv
    train_manifest.json / dev_manifest.json / eval_manifest.json   (NeMo format)
    summary.txt
"""

# ── cuDNN fix — MUST be before any torch import ─────────────────────────────────
import os
_torch_lib = "/path/to/env/lib/python3.10/site-packages/torch/lib"
os.environ["LD_LIBRARY_PATH"] = _torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# ────────────────────────────────────────────────────────────────────────────────

import csv
import json
import re
import subprocess
import tempfile
import unicodedata
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict

import torchaudio
import torch
import jiwer
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these before running
# ══════════════════════════════════════════════════════════════════════════════

BASE_FOLDER  = "/workspace/datasets/debaditya/SPRING_INX_Assamese_R3_cleaned"          # <── change this
SPLITS       = ["train", "dev", "eval"]              # subfolder names to process
OUTPUT_DIR   = "/workspace/datasets/debaditya/kaldi_results"

NEMO_MODEL   = "/workspace/debaditya/nemo/NeMo/indicconformer_600m_multi.nemo"
LANG         = "as"
DECODER      = "rnnt"       # "ctc" (fast) or "rnnt" (accurate)
BATCH_SIZE   = 64
NUM_GPUS     = 2
TARGET_SR    = 16000

# ══════════════════════════════════════════════════════════════════════════════


# ── Kaldi file parsers ───────────────────────────────────────────────────────────

def parse_text(path: Path) -> dict[str, str]:
    """
    text file format:  utt_id word1 word2 word3
    Returns {utt_id: transcript}
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            transcript = parts[1] if len(parts) > 1 else ""
            result[utt_id] = transcript
    return result


def parse_wav_scp(path: Path) -> dict[str, str]:
    """
    wav.scp format:  rec_id /path/to/file.wav
                 or  rec_id sox /path/to/file.wav -t wav - |    (pipe command)
    Returns {rec_id: path_or_command}
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def parse_utt2spk(path: Path) -> dict[str, str]:
    """
    utt2spk format:  utt_id spk_id
    Returns {utt_id: spk_id}
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                result[parts[0]] = parts[1]
    return result


def parse_segments(path: Path) -> dict[str, tuple[str, float, float]]:
    """
    segments format:  utt_id rec_id start_sec end_sec
    Returns {utt_id: (rec_id, start_sec, end_sec)}
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 4:
                utt_id = parts[0]
                rec_id = parts[1]
                start  = float(parts[2])
                end    = float(parts[3])
                result[utt_id] = (rec_id, start, end)
    return result


def load_kaldi_split(split_dir: Path) -> list[dict]:
    """
    Parse all Kaldi files in split_dir and return a list of utterance dicts.
    Each dict has: utt_id, rec_id, audio_source, start, end, transcript, spk_id
    audio_source is either a file path or a shell pipe command.
    """
    required = ["text", "wav.scp", "utt2spk"]
    for f in required:
        if not (split_dir / f).exists():
            raise FileNotFoundError(
                f"Required Kaldi file missing: {split_dir / f}"
            )

    print(f"  Parsing text       ...", end=" ")
    text_map   = parse_text(split_dir / "text")
    print(f"{len(text_map):,} utterances")

    print(f"  Parsing wav.scp    ...", end=" ")
    wav_map    = parse_wav_scp(split_dir / "wav.scp")
    print(f"{len(wav_map):,} recordings")

    print(f"  Parsing utt2spk    ...", end=" ")
    utt2spk    = parse_utt2spk(split_dir / "utt2spk")
    print(f"{len(utt2spk):,} utterances")

    # segments is optional
    seg_path   = split_dir / "segments"
    segments   = None
    if seg_path.exists():
        print(f"  Parsing segments   ...", end=" ")
        segments = parse_segments(seg_path)
        print(f"{len(segments):,} segments")
    else:
        print(f"  segments file      : not found — treating wav.scp keys as utt IDs")

    # Build utterance list
    utterances = []
    missing    = []

    for utt_id, transcript in text_map.items():
        spk_id = utt2spk.get(utt_id, "unknown")

        if segments is not None:
            # Segmented: utt maps to a time slice of a recording
            if utt_id not in segments:
                missing.append(f"segment missing for utt: {utt_id}")
                continue
            rec_id, start, end = segments[utt_id]
            if rec_id not in wav_map:
                missing.append(f"wav.scp missing rec: {rec_id}")
                continue
            audio_source = wav_map[rec_id]
        else:
            # Non-segmented: utt_id == rec_id, look it up directly in wav.scp
            if utt_id not in wav_map:
                missing.append(f"wav.scp missing utt: {utt_id}")
                continue
            audio_source = wav_map[utt_id]
            rec_id = utt_id
            start  = None
            end    = None

        utterances.append({
            "utt_id":       utt_id,
            "rec_id":       rec_id,
            "spk_id":       spk_id,
            "audio_source": audio_source,   # file path or pipe command
            "start":        start,          # float or None
            "end":          end,            # float or None
            "transcript":   transcript,
        })

    if missing:
        print(f"  [WARN] {len(missing):,} utterances skipped (e.g. {missing[:2]})")

    print(f"  Loaded {len(utterances):,} utterances for this split")
    return utterances


# ── Audio extraction ─────────────────────────────────────────────────────────────

def is_pipe_command(audio_source: str) -> bool:
    """Kaldi pipe commands end with ' |'"""
    return audio_source.strip().endswith("|")


def load_audio_from_source(audio_source: str,
                            start: float | None,
                            end:   float | None) -> tuple[torch.Tensor, int]:
    """
    Load audio from either:
      - A file path (with optional start/end for segment extraction)
      - A Kaldi pipe command (e.g. 'sox file.sph -t wav - |')

    Returns (waveform_tensor, sample_rate)
    """
    if is_pipe_command(audio_source):
        # Execute pipe command, capture WAV bytes from stdout
        cmd = audio_source.rstrip().rstrip("|").strip()
        result = subprocess.run(
            cmd, shell=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Pipe command failed: {cmd}\n{result.stderr.decode()}")
        import io
        wav, sr = torchaudio.load(io.BytesIO(result.stdout))
    else:
        # Direct file path
        audio_path = audio_source.strip()
        if start is not None and end is not None:
            # Read only the needed segment — efficient, no full file load
            info         = torchaudio.info(audio_path)
            sr           = info.sample_rate
            frame_offset = int(start * sr)
            num_frames   = int((end   * sr) - frame_offset)
            wav, sr      = torchaudio.load(
                audio_path,
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        else:
            wav, sr = torchaudio.load(audio_path)

    return wav, sr


def prepare_audio_batch(batch: list[dict], tmpdir: str) -> list[str]:
    """
    For each utterance in batch:
      - Load audio (handle pipes and segments)
      - Convert to mono 16kHz
      - Save as temp WAV file NeMo can read
    Returns list of temp file paths (same order as batch).
    """
    paths = []
    for item in batch:
        try:
            wav, sr = load_audio_from_source(
                item["audio_source"],
                item["start"],
                item["end"],
            )

            # Stereo → mono
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=TARGET_SR
                )
                wav = resampler(wav)

            # Save to temp WAV
            tmp_path = os.path.join(tmpdir, item["utt_id"].replace("/", "_") + ".wav")
            torchaudio.save(tmp_path, wav, TARGET_SR,
                            encoding="PCM_S", bits_per_sample=16)
            paths.append(tmp_path)

        except Exception as e:
            # On error, append None — inference will skip this index
            tqdm.write(f"  [AUDIO ERROR] {item['utt_id']}: {e}")
            paths.append(None)

    return paths


# ── Text normalisation ───────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r'[\u0964\u0965\u0c3d]', '', text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


# ── Metrics ──────────────────────────────────────────────────────────────────────

# ── Metrics ──────────────────────────────────────────────────────────────────────

def compute_metrics(ref_raw: str, hyp_raw: str) -> tuple[float, float]:
    ref = normalize(ref_raw)
    hyp = normalize(hyp_raw)

    # Both empty — perfect by definition
    if ref == "" and hyp == "":
        return 0.0, 0.0

    # Empty reference — WER/CER undefined, mark as -1 so we can filter later
    # These rows will be excluded from corpus-level stats
    if ref == "":
        return -1.0, -1.0

    # Non-empty ref, empty hypothesis — 100% deletion
    if hyp == "":
        return 100.0, 100.0

    utt_wer = round(jiwer.wer(ref, hyp) * 100, 4)
    utt_cer = round(jiwer.cer(ref, hyp) * 100, 4)
    return utt_wer, utt_cer


# ── Per-GPU worker ───────────────────────────────────────────────────────────────

def run_on_gpu(gpu_id: int,
               utterances: list[dict],
               output_csv: str,
               split_name: str):
    """
    Process a subset of utterances on one GPU.
    Handles audio loading, resampling, inference, WER/CER computation.
    """
    import os
    os.environ["LD_LIBRARY_PATH"] = (
        "/path/to/env/lib/python3.10/site-packages/torch/lib"
        + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    import torch
    import nemo.collections.asr as nemo_asr
    import torchaudio
    import csv
    import tempfile
    import jiwer
    from pathlib import Path
    from tqdm import tqdm

    device = torch.device(f"cuda:{gpu_id}")

    print(f"\n[GPU {gpu_id} | {split_name}] Loading model ...")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        restore_path=NEMO_MODEL,
        map_location=device,
    )
    model.freeze()
    model = model.to(device)
    model.eval()
    print(f"[GPU {gpu_id} | {split_name}] Ready | {len(utterances):,} utterances\n")

    fieldnames = [
        "utt_id", "spk_id", "rec_id",
        "start", "end",
        "reference", "hypothesis",
        "ref_normalized", "hyp_normalized",
        "WER", "CER",
    ]

    csv_file = open(output_csv, "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    total_wer, total_cer, n_processed = 0.0, 0.0, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for batch_start in tqdm(
            range(0, len(utterances), BATCH_SIZE),
            desc=f"GPU{gpu_id}/{split_name}",
            unit="batch",
            position=gpu_id,
            leave=True,
        ):
            batch = utterances[batch_start : batch_start + BATCH_SIZE]

            # ── Prepare audio (load + resample + segment) ──────────────────────
            prepped_paths = prepare_audio_batch(batch, tmpdir)

            valid_idx   = [i for i, p in enumerate(prepped_paths) if p is not None]
            valid_paths = [prepped_paths[i] for i in valid_idx]

            # ── Inference ──────────────────────────────────────────────────────
            hyp_map = {}
            if valid_paths:
                try:
                    model.cur_decoder = DECODER
                    results = model.transcribe(
                        valid_paths,
                        batch_size=BATCH_SIZE,
                        language_id=LANG,
                    )
                    if isinstance(results, tuple):
                        results = results[0]
                    for i, pred in zip(valid_idx, results):
                        hyp_map[i] = pred if pred is not None else ""
                except Exception as e:
                    tqdm.write(f"  [INFERENCE ERROR] GPU{gpu_id} batch {batch_start}: {e}")

            # Failed audio
            for i in range(len(batch)):
                if i not in hyp_map:
                    hyp_map[i] = "[AUDIO_FAILED]"

            # ── Metrics + write ────────────────────────────────────────────────
            for i, item in enumerate(batch):
                hyp     = hyp_map.get(i, "[INFERENCE_FAILED]")
                ref_raw = item["transcript"]

                utt_wer, utt_cer = compute_metrics(ref_raw, hyp)
                total_wer   += utt_wer
                total_cer   += utt_cer
                n_processed += 1

                writer.writerow({
                    "utt_id":          item["utt_id"],
                    "spk_id":          item["spk_id"],
                    "rec_id":          item["rec_id"],
                    "start":           item["start"] if item["start"] is not None else "",
                    "end":             item["end"]   if item["end"]   is not None else "",
                    "reference":       ref_raw,
                    "hypothesis":      hyp,
                    "ref_normalized":  normalize(ref_raw),
                    "hyp_normalized":  normalize(hyp),
                    "WER":             utt_wer,
                    "CER":             utt_cer,
                })

            csv_file.flush()

            # Clean up temp audio files after each batch
            for p in prepped_paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    csv_file.close()
    avg_wer = total_wer / n_processed if n_processed else 0
    avg_cer = total_cer / n_processed if n_processed else 0
    print(f"\n[GPU {gpu_id} | {split_name}] Done | {n_processed:,} utts | "
          f"Avg WER: {avg_wer:.2f}%  Avg CER: {avg_cer:.2f}%")


# ── Merge CSVs ───────────────────────────────────────────────────────────────────

def merge_csvs(partial_paths: list[str], out_path: str):
    fieldnames = [
        "utt_id", "spk_id", "rec_id",
        "start", "end",
        "reference", "hypothesis",
        "ref_normalized", "hyp_normalized",
        "WER", "CER",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for path in partial_paths:
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    writer.writerow(row)


# ── Per-split summary ─────────────────────────────────────────────────────────────

# def compute_split_summary(csv_path: str, split_name: str) -> dict:
#     all_refs, all_hyps = [], []
#     total_wer = total_cer = n = 0

#     with open(csv_path, encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             all_refs.append(row["ref_normalized"])
#             all_hyps.append(row["hyp_normalized"])
#             total_wer += float(row["WER"])
#             total_cer += float(row["CER"])
#             n += 1

#     if n == 0:
#         return {"split": split_name, "n": 0,
#                 "avg_wer": 0, "avg_cer": 0,
#                 "corpus_wer": 0, "corpus_cer": 0}

#     corpus_wer = round(jiwer.wer(all_refs, all_hyps) * 100, 4)
#     corpus_cer = round(jiwer.cer(all_refs, all_hyps) * 100, 4)

#     return {
#         "split":      split_name,
#         "n":          n,
#         "avg_wer":    round(total_wer / n, 4),
#         "avg_cer":    round(total_cer / n, 4),
#         "corpus_wer": corpus_wer,
#         "corpus_cer": corpus_cer,
#     }

# ── Per-split summary ─────────────────────────────────────────────────────────────

def compute_split_summary(csv_path: str, split_name: str) -> dict:
    all_refs, all_hyps = [], []
    total_wer = total_cer = n = n_skipped = 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref = row["ref_normalized"].strip()
            hyp = row["hyp_normalized"].strip()
            wer_val = float(row["WER"])
            cer_val = float(row["CER"])

            # Skip rows with empty reference (WER undefined)
            # These are marked as -1.0 by compute_metrics
            if ref == "" or wer_val < 0:
                n_skipped += 1
                continue

            # Skip rows where inference explicitly failed
            if "[INFERENCE_FAILED]" in hyp or "[AUDIO_FAILED]" in hyp:
                n_skipped += 1
                continue

            all_refs.append(ref)
            all_hyps.append(hyp)
            total_wer += wer_val
            total_cer += cer_val
            n += 1

    if n_skipped > 0:
        print(f"  [{split_name}] Skipped {n_skipped:,} rows from summary "
              f"(empty reference or failed inference)")

    if n == 0:
        print(f"  [{split_name}] No valid rows for summary computation.")
        return {"split": split_name, "n": 0,
                "avg_wer": 0, "avg_cer": 0,
                "corpus_wer": 0, "corpus_cer": 0}

    corpus_wer = round(jiwer.wer(all_refs, all_hyps) * 100, 4)
    corpus_cer = round(jiwer.cer(all_refs, all_hyps) * 100, 4)

    return {
        "split":      split_name,
        "n":          n,
        "n_skipped":  n_skipped,
        "avg_wer":    round(total_wer / n, 4),
        "avg_cer":    round(total_cer / n, 4),
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
    }

def write_nemo_manifest(csv_path: str, manifest_path: str):
    """Write a NeMo manifest from a results CSV."""
    with open(csv_path, encoding="utf-8") as f_in, \
         open(manifest_path, "w", encoding="utf-8") as f_out:
        for row in csv.DictReader(f_in):
            # Re-derive audio path from utt_id — we store it in audio source
            # Use ref_normalized as the NeMo text field
            entry = {
                "audio_filepath": "",        # see note below
                "text":           row["ref_normalized"],
                "utt_id":         row["utt_id"],
                "duration":       0.0,       # computed below if possible
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Process one split ────────────────────────────────────────────────────────────

def process_split(split_name: str, out_dir: Path) -> dict:
    split_dir = Path(BASE_FOLDER) / split_name
    if not split_dir.exists():
        print(f"\n[SKIP] Split folder not found: {split_dir}")
        return {}

    print(f"\n{'═'*65}")
    print(f"  SPLIT: {split_name.upper()}")
    print(f"{'═'*65}")
    print(f"  Folder: {split_dir}")

    utterances = load_kaldi_split(split_dir)
    if not utterances:
        print(f"  [SKIP] No utterances loaded for {split_name}")
        return {}

    num_gpus   = min(NUM_GPUS, torch.cuda.device_count(), len(utterances))
    final_csv  = str(out_dir / f"{split_name}_results.csv")

    if num_gpus == 1:
        run_on_gpu(0, utterances, final_csv, split_name)
    else:
        # Split utterances across GPUs
        chunk      = len(utterances) // num_gpus
        splits     = []
        for i in range(num_gpus):
            start = i * chunk
            end   = start + chunk if i < num_gpus - 1 else len(utterances)
            splits.append(utterances[start:end])

        partial_csvs = [
            str(out_dir / f"{split_name}_gpu{i}.csv")
            for i in range(num_gpus)
        ]

        print(f"\n  Launching {num_gpus} GPU workers ...")
        for i, s in enumerate(splits):
            print(f"    GPU {i} → {len(s):,} utterances")

        procs = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=run_on_gpu,
                args=(gpu_id, splits[gpu_id], partial_csvs[gpu_id], split_name),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        merge_csvs(partial_csvs, final_csv)
        for path in partial_csvs:
            if os.path.exists(path):
                os.remove(path)

    # Write NeMo manifest
    manifest_path = str(out_dir / f"{split_name}_manifest.json")
    _write_manifest_from_csv(final_csv, manifest_path, split_dir)
    print(f"  Manifest written → {manifest_path}")

    return compute_split_summary(final_csv, split_name)


def _write_manifest_from_csv(csv_path: str, manifest_path: str, split_dir: Path):
    """
    Write NeMo manifest from results CSV.
    Re-reads wav.scp and segments to recover audio paths and durations.
    """
    # Reload wav.scp and segments for audio path resolution
    wav_map  = parse_wav_scp(split_dir / "wav.scp")
    seg_path = split_dir / "segments"
    segments = parse_segments(seg_path) if seg_path.exists() else None

    with open(csv_path, encoding="utf-8") as f_in, \
         open(manifest_path, "w", encoding="utf-8") as f_out:
        for row in csv.DictReader(f_in):
            utt_id = row["utt_id"]

            if segments and utt_id in segments:
                rec_id, start, end = segments[utt_id]
                audio_source = wav_map.get(rec_id, "")
                duration = end - start
            else:
                audio_source = wav_map.get(utt_id, "")
                start = None
                # Get duration from file if it's a plain path
                try:
                    if audio_source and not is_pipe_command(audio_source):
                        info = torchaudio.info(audio_source.strip())
                        duration = info.num_frames / info.sample_rate
                    else:
                        duration = 0.0
                except Exception:
                    duration = 0.0

            entry = {
                "audio_filepath": audio_source.strip() if not is_pipe_command(audio_source) else "",
                "text":           row["ref_normalized"],
                "utt_id":         utt_id,
                "duration":       round(duration, 3),
            }
            if start is not None:
                entry["offset"] = start   # NeMo supports offset for segment loading

            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    mp.set_start_method("spawn", force=True)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  IndicConformer — Kaldi Format Evaluation")
    print(f"{'═'*65}")
    print(f"  Base folder : {BASE_FOLDER}")
    print(f"  Splits      : {SPLITS}")
    print(f"  Decoder     : {DECODER.upper()}")
    print(f"  Language    : {LANG}")
    print(f"  GPUs        : {NUM_GPUS}")
    print(f"  Output      : {out_dir.resolve()}/")

    # Check GPU availability
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        print("\n[ERROR] No CUDA GPUs found.")
        return
    for i in range(n_gpu):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    all_summaries = []
    for split_name in SPLITS:
        summary = process_split(split_name, out_dir)
        if summary:
            all_summaries.append(summary)

    # ── Print final summary ──────────────────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print(f"  FINAL SUMMARY — {DECODER.upper()} decoder | Language: {LANG}")
    print(f"{'═'*65}")
    print(f"  {'Split':<8} {'Utterances':>12} {'Corpus WER':>12} {'Corpus CER':>12} {'Avg Utt WER':>13}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*13}")
    for s in all_summaries:
        print(f"  {s['split']:<8} {s['n']:>12,} "
              f"{s['corpus_wer']:>11.2f}% "
              f"{s['corpus_cer']:>11.2f}% "
              f"{s['avg_wer']:>12.2f}%")
    print(f"{'═'*65}")
    print(f"\n  ► Report corpus-level WER — standard benchmark metric.")
    print(f"  ► All results in: {out_dir.resolve()}/\n")

    # Save summary to text file
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Decoder: {DECODER.upper()} | Language: {LANG}\n\n")
        f.write(f"{'Split':<8} {'Utterances':>12} {'Corpus WER':>12} {'Corpus CER':>12} {'Avg Utt WER':>13}\n")
        f.write(f"{'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*13}\n")
        for s in all_summaries:
            f.write(f"{s['split']:<8} {s['n']:>12,} "
                    f"{s['corpus_wer']:>11.2f}% "
                    f"{s['corpus_cer']:>11.2f}% "
                    f"{s['avg_wer']:>12.2f}%\n")
    print(f"  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()