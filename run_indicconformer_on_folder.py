# ── Fix cuDNN library loading BEFORE any torch import ──────────────────────────
import os
_torch_lib = "/path/to/env/lib/python3.10/site-packages/torch/lib"
os.environ["LD_LIBRARY_PATH"] = _torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# ────────────────────────────────────────────────────────────────────────────────

import re
import csv
import unicodedata
import multiprocessing as mp
from pathlib import Path

import jiwer
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────────
MAIN_FOLDER  = "/workspace/datasets/debaditya/IIIT_Hyderabad_Telugu_cstd_100hrs"
#/workspace/datasets/debaditya/IIIT_Hyderabad_Telugu_cstd_100hrs/audio_16k
AUDIO_SUBDIR = "audio_16k"
TRANS_SUBDIR = "transcripts"
OUTPUT_CSV   = "telugu_wer_cer_results.csv"

LANG         = "te"
DECODER      = "ctc"    # ctc = fast (~25-40 min); swap to "rnnt" for max accuracy
BATCH_SIZE   = 64       # safe for H100 80GB; try 128 if no OOM
NEMO_MODEL   = "/workspace/debaditya/nemo/NeMo/indicconformer_600m_multi.nemo"
NUM_GPUS     = 2        # set to 1 if you only want one GPU
# ────────────────────────────────────────────────────────────────────────────────


# ── Text normalisation (Telugu) ─────────────────────────────────────────────────
def normalize(text: str) -> str:
    """
    Normalise Telugu text before metric computation.
    Reference and hypothesis go through the same pipeline
    so formatting differences don't count as errors.
    """
    # 1. Unicode NFC — consistent codepoint representation
    text = unicodedata.normalize("NFC", text)
    # 2. Strip leading/trailing whitespace
    text = text.strip()
    # 3. Remove Telugu punctuation: । ॥ ఽ and all ASCII punctuation
    text = re.sub(r'[\u0964\u0965\u0c3d]', '', text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    # 4. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Lowercase (safe for Unicode — doesn't touch Telugu glyphs)
    text = text.lower()
    return text


# ── Discover pairs ───────────────────────────────────────────────────────────────
def discover_pairs(main_folder: str) -> list[dict]:
    """
    Scan Main_folder/Audio/*.wav and Main_folder/transcripts/*.txt,
    match by filename stem, return list of dicts.
    """
    root      = Path(main_folder)
    audio_dir = root / AUDIO_SUBDIR
    trans_dir = root / TRANS_SUBDIR

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")
    if not trans_dir.exists():
        raise FileNotFoundError(f"Transcripts folder not found: {trans_dir}")

    audio_files = {f.stem: f for f in sorted(audio_dir.glob("*.wav"))}
    trans_files = {f.stem: f for f in sorted(trans_dir.glob("*.txt"))}

    matched  = sorted(set(audio_files) & set(trans_files))
    no_trans = sorted(set(audio_files) - set(trans_files))
    no_audio = sorted(set(trans_files) - set(audio_files))

    print(f"  Audio files found      : {len(audio_files):,}")
    print(f"  Transcript files found : {len(trans_files):,}")
    print(f"  Matched pairs          : {len(matched):,}")
    if no_trans:
        print(f"  [WARN] No transcript for {len(no_trans):,} audio files "
              f"(e.g. {no_trans[:3]})")
    if no_audio:
        print(f"  [WARN] No audio for {len(no_audio):,} transcript files "
              f"(e.g. {no_audio[:3]})")

    return [
        {
            "stem":            stem,
            "audio_path":      str(audio_files[stem]),
            "transcript_path": str(trans_files[stem]),
        }
        for stem in matched
    ]


# ── Metrics ──────────────────────────────────────────────────────────────────────
def compute_metrics(ref_raw: str, hyp_raw: str) -> tuple[float, float]:
    ref = normalize(ref_raw)
    hyp = normalize(hyp_raw)
    if ref == "" and hyp == "":
        return 0.0, 0.0
    utt_wer = round(jiwer.wer(ref, hyp) * 100, 4)
    utt_cer = round(jiwer.cer(ref, hyp) * 100, 4)
    return utt_wer, utt_cer


# ── Per-GPU worker ───────────────────────────────────────────────────────────────
def run_on_gpu(gpu_id: int, pairs: list[dict], output_csv: str):
    """
    Load the model on gpu_id, transcribe all pairs assigned to this GPU,
    compute WER/CER per utterance, write results to output_csv.
    Called in a separate process per GPU.
    """
    # Re-apply cuDNN fix inside the child process
    import os
    _torch_lib = "/path/to/env/lib/python3.10/site-packages/torch/lib"
    os.environ["LD_LIBRARY_PATH"] = _torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    import torch
    import nemo.collections.asr as nemo_asr
    import csv
    import jiwer
    from pathlib import Path
    from tqdm import tqdm

    device = torch.device(f"cuda:{gpu_id}")

    print(f"\n[GPU {gpu_id}] Loading model ...")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        restore_path=NEMO_MODEL,
        map_location=device,
    )
    model.freeze()
    model = model.to(device)
    model.eval()
    print(f"[GPU {gpu_id}] Model ready on {next(model.parameters()).device}")
    print(f"[GPU {gpu_id}] Processing {len(pairs):,} utterances ...\n")

    fieldnames = [
        "utterance_id", "audio_path",
        "reference", "hypothesis",
        "ref_normalized", "hyp_normalized",
        "WER", "CER",
    ]

    csv_file = open(output_csv, "w", newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    total_wer, total_cer, n_processed = 0.0, 0.0, 0

    for batch_start in tqdm(
        range(0, len(pairs), BATCH_SIZE),
        desc=f"GPU {gpu_id}",
        unit="batch",
        position=gpu_id,
        leave=True,
    ):
        batch       = pairs[batch_start : batch_start + BATCH_SIZE]
        audio_paths = [item["audio_path"] for item in batch]

        # ── Inference ──
        try:
            model.cur_decoder = DECODER
            results = model.transcribe(
                audio_paths,
                batch_size=BATCH_SIZE,
                language_id=LANG,
            )
            # NeMo may return (texts, logprobs) tuple
            if isinstance(results, tuple):
                results = results[0]
            preds = [r if r is not None else "" for r in results]
        except Exception as e:
            tqdm.write(f"  [GPU {gpu_id} ERROR] batch {batch_start}: {e}")
            preds = ["[INFERENCE_FAILED]"] * len(batch)

        # ── Metrics + write ──
        for item, hyp in zip(batch, preds):
            ref_raw = Path(item["transcript_path"]).read_text(
                          encoding="utf-8").strip()

            utt_wer, utt_cer = compute_metrics(ref_raw, hyp)
            total_wer   += utt_wer
            total_cer   += utt_cer
            n_processed += 1

            writer.writerow({
                "utterance_id":   item["stem"],
                "audio_path":     item["audio_path"],
                "reference":      ref_raw,
                "hypothesis":     hyp,
                "ref_normalized": normalize(ref_raw),
                "hyp_normalized": normalize(hyp),
                "WER":            utt_wer,
                "CER":            utt_cer,
            })

        # Flush every batch — safe even if job is killed mid-run
        csv_file.flush()

    csv_file.close()

    avg_wer = total_wer / n_processed if n_processed else 0
    avg_cer = total_cer / n_processed if n_processed else 0
    print(f"\n[GPU {gpu_id}] Finished {n_processed:,} utterances | "
          f"Avg WER: {avg_wer:.2f}%  Avg CER: {avg_cer:.2f}%")


# ── Merge CSVs ───────────────────────────────────────────────────────────────────
def merge_csvs(csv_paths: list[str], output_path: str):
    fieldnames = [
        "utterance_id", "audio_path",
        "reference", "hypothesis",
        "ref_normalized", "hyp_normalized",
        "WER", "CER",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for path in csv_paths:
            if not os.path.exists(path):
                print(f"  [WARN] Missing partial CSV: {path}")
                continue
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    writer.writerow(row)
    print(f"\nMerged {len(csv_paths)} CSVs → {output_path}")


# ── Final corpus-level summary ───────────────────────────────────────────────────
def print_summary(csv_path: str):
    all_refs, all_hyps = [], []
    total_wer, total_cer, n = 0.0, 0.0, 0

    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_refs.append(row["ref_normalized"])
            all_hyps.append(row["hyp_normalized"])
            total_wer += float(row["WER"])
            total_cer += float(row["CER"])
            n += 1

    corpus_wer = round(jiwer.wer(all_refs, all_hyps) * 100, 4)
    corpus_cer = round(jiwer.cer(all_refs, all_hyps) * 100, 4)

    print("\n" + "═" * 60)
    print(f"  SUMMARY — Telugu ({DECODER.upper()}) | {n:,} utterances")
    print("═" * 60)
    print(f"  Avg utterance WER  : {total_wer / n:.2f}%")
    print(f"  Avg utterance CER  : {total_cer / n:.2f}%")
    print(f"  Corpus-level WER   : {corpus_wer:.2f}%")
    print(f"  Corpus-level CER   : {corpus_cer:.2f}%")
    print("═" * 60)
    print("\n  ► Report corpus-level WER — it is the standard benchmark number.")
    print(f"  ► Full results saved to: {csv_path}\n")


# ── Main ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # required for CUDA multiprocessing

    print(f"Scanning {MAIN_FOLDER} ...")
    pairs = discover_pairs(MAIN_FOLDER)

    if not pairs:
        print("No matched pairs found. Check MAIN_FOLDER, AUDIO_SUBDIR, TRANS_SUBDIR.")
        exit(1)

    if NUM_GPUS == 1:
        # ── Single GPU mode ──────────────────────────────────────────────────────
        run_on_gpu(gpu_id=0, pairs=pairs, output_csv=OUTPUT_CSV)

    else:
        # ── Multi-GPU mode (default: 2 H100s) ────────────────────────────────────
        # Split pairs as evenly as possible across GPUs
        chunk_size = len(pairs) // NUM_GPUS
        splits     = []
        for i in range(NUM_GPUS):
            start = i * chunk_size
            end   = start + chunk_size if i < NUM_GPUS - 1 else len(pairs)
            splits.append(pairs[start:end])

        partial_csvs = [f"results_gpu{i}.csv" for i in range(NUM_GPUS)]

        print(f"\nLaunching {NUM_GPUS} GPU workers ...")
        for i, split in enumerate(splits):
            print(f"  GPU {i} → {len(split):,} utterances")

        procs = []
        for gpu_id in range(NUM_GPUS):
            p = mp.Process(
                target=run_on_gpu,
                args=(gpu_id, splits[gpu_id], partial_csvs[gpu_id]),
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # Merge partial CSVs from both GPUs into one final file
        merge_csvs(partial_csvs, OUTPUT_CSV)

        # Clean up partial files
        for path in partial_csvs:
            if os.path.exists(path):
                os.remove(path)

    # Print final summary from the merged CSV
    print_summary(OUTPUT_CSV)