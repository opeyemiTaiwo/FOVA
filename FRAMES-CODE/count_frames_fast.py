"""
Fast frame counter for UCF-Crime FRAMES directory.

Speed tactics (the ones that actually matter for filesystem I/O):
  - os.scandir() instead of os.listdir() — fewer syscalls per entry
  - ThreadPoolExecutor across crime-type folders — parallel disk reads
  - String suffix check instead of os.path.splitext() — avoids extra calls
  - Optional JSON cache — skip the scan entirely on subsequent runs

GPUs cannot help with this; the work is filesystem syscalls, not arithmetic.
"""
import os
import json
import time
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

FRAMES_DIR  = r"C:\Opeyemi\PROMPTS\FRAMES"
OUTPUT_DIR  = r"C:\Opeyemi\PROMPTS\RESULTS\FRAME-CAL"
CACHE_FILE  = OUTPUT_DIR + r"\frame_counts_cache.json"
TXT_FILE    = OUTPUT_DIR + r"\frame_counts_summary.txt"
USE_CACHE   = True              # set False to force a fresh scan
MAX_WORKERS = 8                 # parallel folder scans (SSDs benefit; HDDs don't)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def _is_image(name_lower: str) -> bool:
    return name_lower.endswith(IMAGE_SUFFIXES)


def _count_in_dir_recursive(path: str) -> int:
    """Walk one directory tree and return the number of image files."""
    total = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    name = entry.name
                    if name.startswith('.'):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        if _is_image(name.lower()):
                            total += 1
        except (PermissionError, FileNotFoundError):
            pass
    return total


def _count_videos_in_crime_folder(crime_dir: str):
    """Return (n_videos, n_frames, per_video_list) for one crime-type folder."""
    per_video = []
    total_frames = 0

    sub_dirs = []
    flat_files = []
    try:
        with os.scandir(crime_dir) as it:
            for entry in it:
                if entry.name.startswith('.'):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    sub_dirs.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    if _is_image(entry.name.lower()):
                        flat_files.append(entry.name)
    except (PermissionError, FileNotFoundError):
        return 0, 0, []

    if sub_dirs:
        # Sub-folder layout: <CrimeType>/<VideoID>/<frames>
        for video_dir in sub_dirs:
            n = _count_in_dir_recursive(video_dir)
            if n > 0:
                video_id = os.path.basename(video_dir)
                per_video.append((video_id, n))
                total_frames += n
    else:
        # Flat layout: group by VideoID prefix
        groups = defaultdict(int)
        for fname in flat_files:
            base = os.path.splitext(fname)[0]
            if "_frame_" in base:
                vid_id = base.split("_frame_")[0]
            else:
                vid_id = re.sub(r"_?\d+$", "", base) or base
            groups[vid_id] += 1
        for video_id, n in groups.items():
            per_video.append((video_id, n))
            total_frames += n

    return len(per_video), total_frames, per_video


def count_frames(frames_dir: str, use_cache: bool = USE_CACHE):
    if not os.path.isdir(frames_dir):
        print(f"ERROR: directory not found: {frames_dir}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if use_cache and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cached = json.load(f)
            if cached.get("frames_dir") == frames_dir:
                print(f"[cache] loaded from {CACHE_FILE}")
                _print_report(cached, from_cache=True)
                return cached
        except Exception:
            pass

    t0 = time.time()
    crime_dirs = []
    with os.scandir(frames_dir) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False) and not entry.name.startswith('.'):
                crime_dirs.append(entry.path)
    crime_dirs.sort()

    print(f"Scanning {len(crime_dirs)} crime folders in parallel "
          f"({MAX_WORKERS} threads)...")

    crime_results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_crime = {
            pool.submit(_count_videos_in_crime_folder, cd): os.path.basename(cd)
            for cd in crime_dirs
        }
        for fut in as_completed(future_to_crime):
            crime = future_to_crime[fut]
            try:
                n_vids, n_frames, per_video = fut.result()
                crime_results[crime] = {
                    "n_videos": n_vids,
                    "n_frames": n_frames,
                    "per_video": per_video,
                }
            except Exception as e:
                print(f"  ERROR scanning {crime}: {e}")

    elapsed = time.time() - t0

    report = {
        "frames_dir": frames_dir,
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "by_crime": crime_results,
        "grand_total_videos": sum(r["n_videos"] for r in crime_results.values()),
        "grand_total_frames": sum(r["n_frames"] for r in crime_results.values()),
    }

    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[cache] saved to {CACHE_FILE}")
    except Exception as e:
        print(f"  WARN: could not write cache: {e}")

    _print_report(report)
    return report


def _print_report(report: dict, from_cache: bool = False):
    by_crime = report["by_crime"]
    print(f"\nFrames directory: {report['frames_dir']}")
    if not from_cache:
        print(f"Scan time: {report['elapsed_seconds']}s")
    print("=" * 70)
    print(f"{'Crime Type':<20} {'Videos':>8} {'Frames':>12} {'Avg/video':>12}")
    print("-" * 70)
    for crime in sorted(by_crime.keys()):
        r = by_crime[crime]
        n_vids = r["n_videos"]
        n_frames = r["n_frames"]
        avg = n_frames / n_vids if n_vids else 0
        print(f"{crime:<20} {n_vids:>8} {n_frames:>12,} {avg:>12,.1f}")
    print("-" * 70)
    gv = report["grand_total_videos"]
    gf = report["grand_total_frames"]
    overall_avg = gf / gv if gv else 0
    print(f"{'TOTAL':<20} {gv:>8} {gf:>12,} {overall_avg:>12,.1f}")
    print("=" * 70)

    all_videos = []
    for crime, r in by_crime.items():
        for vid, n in r["per_video"]:
            all_videos.append((crime, vid, n))
    if all_videos:
        all_videos.sort(key=lambda x: x[2], reverse=True)
        print("\nTop 5 longest videos:")
        for crime, vid, n in all_videos[:5]:
            print(f"  {n:>7,}  {crime}/{vid}")
        print("\nTop 5 shortest videos:")
        for crime, vid, n in all_videos[-5:]:
            print(f"  {n:>7,}  {crime}/{vid}")


if __name__ == "__main__":
    report = count_frames(FRAMES_DIR)

    # Also save the human-readable table to a .txt file
    if report:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_report(report)
        with open(TXT_FILE, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print(f"\n[txt] table saved to {TXT_FILE}")
