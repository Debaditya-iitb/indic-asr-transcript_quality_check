"""
resample_to_16k.py
──────────────────
GPU-accelerated batch resampling of audio files to 16kHz mono.
Splits work across all available GPUs automatically.

Usage:
    # Convert in-place (overwrites originals)
    python3 resample_to_16k.py --input /path/to/Audio

    # Convert to a separate output folder (keeps originals)
    python3 resample_to_16k.py --input /path/to/Audio --output /path/to/Audio_16k

    # Use only 1 GPU
    python3 resample_to_16k.py --input /path/to/Audio --gpus 1

    # Dry run — check how many files need conversion without doing anything
    python3 resample_to_16k.py --input /path/to/Audio --dry-run
"""

# ── cuDNN fix — must be before any torch import ─────────────────────────────────
import os
_torch_lib = "/path/to/env/lib/python3.10/site-packages/torch/lib"
os.environ["LD_LIBRARY_PATH"] = _torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
# ────────────────────────────────────────────────────────────────────────────────

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

TARGET_SR    = 16000
AUDIO_EXTS   = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}
BATCH_SIZE   = 64     # files processed per GPU batch — tune if OOM


# ── Helpers ──────────────────────────────────────────────────────────────────────

def scan_files(input_dir: Path) -> list[Path]:
    """Recursively find all audio files under input_dir."""
    files = []
    for ext in AUDIO_EXTS:
        files.extend(input_dir.rglob(f"*{ext}"))
    return sorted(files)


def needs_conversion(path: Path) -> bool:
    """Return True if file is not already 16kHz mono."""
    try:
        info = torchaudio.info(str(path))
        return info.sample_rate != TARGET_SR or info.num_channels != 1
    except Exception:
        return True  # if we can't read it, try to convert


def get_output_path(input_path: Path, input_dir: Path,
                    output_dir: Path | None) -> Path:
    """
    If output_dir given  → mirror the input subfolder structure under output_dir
    If output_dir is None → overwrite in-place
    """
    if output_dir is None:
        return input_path
    rel = input_path.relative_to(input_dir)
    out = output_dir / rel
    out = out.with_suffix(".wav")   # always write as WAV
    return out


# ── Per-GPU worker ───────────────────────────────────────────────────────────────

def worker(gpu_id: int,
           file_paths: list[Path],
           input_dir: Path,
           output_dir: Path | None,
           skip_if_done: bool,
           result_queue: mp.Queue):
    """
    Run on a single GPU: load → move to GPU → resample → mono → save.
    Reports (done, skipped, errors) back via result_queue.
    """
    # Re-apply cuDNN fix in child process
    os.environ["LD_LIBRARY_PATH"] = (
        "/path/to/env/lib/python3.10/site-packages/torch/lib"
        + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    device    = torch.device(f"cuda:{gpu_id}")
    done_cnt  = 0
    skip_cnt  = 0
    err_cnt   = 0

    # Cache resamplers per source sample rate so we don't rebuild them each file
    resamplers: dict[int, torchaudio.transforms.Resample] = {}

    for path in tqdm(file_paths,
                     desc=f"GPU {gpu_id}",
                     unit="file",
                     position=gpu_id,
                     leave=True):

        out_path = get_output_path(path, input_dir, output_dir)

        # Skip already-converted files if requested
        if skip_if_done and out_path.exists() and not needs_conversion(out_path):
            skip_cnt += 1
            continue

        try:
            # ── Load (CPU) ───────────────────────────────────────────────────────
            wav, sr = torchaudio.load(str(path))   # shape: (C, T)

            # ── Move to GPU ──────────────────────────────────────────────────────
            wav = wav.to(device)

            # ── Stereo → mono ────────────────────────────────────────────────────
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            # ── Resample if needed ───────────────────────────────────────────────
            if sr != TARGET_SR:
                if sr not in resamplers:
                    resamplers[sr] = torchaudio.transforms.Resample(
                        orig_freq=sr,
                        new_freq=TARGET_SR,
                        resampling_method="sinc_interp_hann",
                    ).to(device)
                wav = resamplers[sr](wav)

            # ── Move back to CPU for saving ──────────────────────────────────────
            wav = wav.cpu()

            # ── Ensure output directory exists ───────────────────────────────────
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # ── Save as 16kHz mono WAV ───────────────────────────────────────────
            torchaudio.save(
                str(out_path),
                wav,
                TARGET_SR,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            done_cnt += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] GPU {gpu_id} | {path.name}: {e}")
            err_cnt += 1

    result_queue.put((gpu_id, done_cnt, skip_cnt, err_cnt))


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated batch resampling to 16kHz mono WAV"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input folder containing audio files (searched recursively)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output folder (default: overwrite input files in-place)"
    )
    parser.add_argument(
        "--gpus", "-g", type=int, default=None,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report how many files need conversion, then exit"
    )
    parser.add_argument(
        "--skip-if-done", action="store_true", default=True,
        help="Skip files that are already 16kHz mono at the output path (default: True)"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Reconvert all files even if already 16kHz mono"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve() if args.output else None
    skip_done  = not args.no_skip

    if not input_dir.exists():
        print(f"[ERROR] Input folder not found: {input_dir}")
        exit(1)

    # ── Scan ────────────────────────────────────────────────────────────────────
    print(f"\nScanning {input_dir} ...")
    all_files = scan_files(input_dir)
    print(f"  Total audio files found : {len(all_files):,}")

    if not all_files:
        print("No audio files found. Check --input path and supported extensions.")
        print(f"Supported: {AUDIO_EXTS}")
        exit(0)

    # Quick sample check (first 10 files) to report what we'll encounter
    print("\nSample rate check (first 10 files):")
    rate_counts: dict[int, int] = {}
    needs_conv  = 0
    for f in all_files[:min(10, len(all_files))]:
        try:
            info = torchaudio.info(str(f))
            rate_counts[info.sample_rate] = rate_counts.get(info.sample_rate, 0) + 1
            if info.sample_rate != TARGET_SR or info.num_channels != 1:
                needs_conv += 1
            print(f"  {f.name:50s} | sr={info.sample_rate:6d} | ch={info.num_channels}")
        except Exception as e:
            print(f"  {f.name:50s} | [READ ERROR] {e}")

    if args.dry_run:
        print(f"\n[DRY RUN] Exiting without converting.")
        print(f"  Files that need conversion (sample): {needs_conv}/{min(10, len(all_files))}")
        print(f"  Total files that would be processed: {len(all_files):,}")
        return

    # ── GPU setup ────────────────────────────────────────────────────────────────
    n_available = torch.cuda.device_count()
    if n_available == 0:
        print("\n[ERROR] No CUDA GPUs found. Cannot run GPU resampling.")
        exit(1)

    num_gpus = min(args.gpus or n_available, n_available, len(all_files))
    print(f"\nGPUs available: {n_available}  |  Using: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput folder : {output_dir}")
    else:
        print(f"\nMode          : IN-PLACE (overwriting originals)")

    # ── Split files across GPUs ──────────────────────────────────────────────────
    chunk = len(all_files) // num_gpus
    splits = []
    for i in range(num_gpus):
        start = i * chunk
        end   = start + chunk if i < num_gpus - 1 else len(all_files)
        splits.append(all_files[start:end])

    print(f"\nSplitting {len(all_files):,} files across {num_gpus} GPU(s):")
    for i, s in enumerate(splits):
        print(f"  GPU {i} → {len(s):,} files")

    # ── Launch workers ───────────────────────────────────────────────────────────
    print(f"\nStarting conversion to {TARGET_SR}Hz mono WAV ...\n")
    t_start = time.time()

    result_queue = mp.Queue()
    procs = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker,
            args=(
                gpu_id,
                splits[gpu_id],
                input_dir,
                output_dir,
                skip_done,
                result_queue,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # ── Collect results ──────────────────────────────────────────────────────────
    total_done = total_skip = total_err = 0
    while not result_queue.empty():
        gpu_id, done, skip, err = result_queue.get()
        print(f"  GPU {gpu_id}: converted={done:,}  skipped={skip:,}  errors={err}")
        total_done += done
        total_skip += skip
        total_err  += err

    elapsed = time.time() - t_start
    mins, secs = divmod(int(elapsed), 60)

    print("\n" + "═" * 55)
    print("  CONVERSION COMPLETE")
    print("═" * 55)
    print(f"  Converted  : {total_done:,} files")
    print(f"  Skipped    : {total_skip:,} files (already 16kHz mono)")
    print(f"  Errors     : {total_err:,} files")
    print(f"  Time taken : {mins}m {secs}s")
    print(f"  Output     : {output_dir if output_dir else input_dir} (in-place)")
    print("═" * 55)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()