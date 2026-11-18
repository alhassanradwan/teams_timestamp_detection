#!/usr/bin/env python
"""Phase 0 benchmark: is "<= 10 s of processing per 1 hour of video" reachable on this machine?

For one recording it measures:
  1. resolution, frame rate, codec          (ffprobe)
  2. keyframe interval                      (ffprobe -skip_frame nokey)
  3. full GPU decode speed                  (ffmpeg -hwaccel cuda ... -f null)
  4. keyframe-only GPU decode time          (ffmpeg -skip_frame nokey -hwaccel cuda ...)
  5. audio extraction + Silero VAD time     (ffmpeg -> 16 kHz mono float32 -> batched VAD)
and then projects the fast pipeline's cost per hour of video.

Usage:
    python scripts/benchmark.py meeting.mp4
    python scripts/benchmark.py meeting.mp4 --max-seconds 600 --json bench.json
    python scripts/benchmark.py meeting.mp4 --ffmpeg C:/ffmpeg/bin/ffmpeg.exe --ffprobe C:/ffmpeg/bin/ffprobe.exe
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is not installed - run: pip install -r requirements.txt")

TARGET_S_PER_HOUR = 10.0
PASS_D_OVERHEAD_S = 1.0          # rough cost of building segments + writing the CSV (Pass D)
FULL_DECODE_SHARES = (0.05, 0.10, 0.20, 0.30)
SAMPLE_RATE = 16000
VAD_CHUNK = 512                  # Silero VAD v5 takes exactly 512 samples (32 ms) per step at 16 kHz
VAD_THRESHOLD = 0.5
VAD_MIN_GAP_S = 0.8              # pauses shorter than this are ignored (project brief)
VAD_MIN_SPEECH_S = 0.25


# --------------------------------------------------------------------------- helpers

def log(msg: str = "") -> None:
    print(msg, flush=True)


def row(label: str, value: str) -> None:
    log(f"  {label:<34} {value}")


def die(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def last_lines(text: str, n: int = 4) -> str:
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
    return "\n    ".join(lines[-n:])


def parse_rate(text: str | None) -> float | None:
    """'30000/1001' -> 29.97, '25' -> 25.0, '0/0' or junk -> None."""
    if not text:
        return None
    try:
        if "/" in text:
            num, den = (float(x) for x in text.split("/", 1))
            return num / den if num > 0 and den > 0 else None
        val = float(text)
        return val if val > 0 else None
    except ValueError:
        return None


def per_hour(seconds: float, media_s: float) -> float:
    return seconds / media_s * 3600.0


def tool_path(name: str) -> str:
    found = shutil.which(name)
    if not found:
        die(f"'{name}' not found. Install FFmpeg (see README) or pass --{Path(name).stem} <path>.")
    return found


# --------------------------------------------------------------------------- 1. video info

def probe(ffprobe: str, video: str) -> dict:
    entries = ("format=duration,format_name:"
               "stream=index,codec_type,codec_name,profile,width,height,pix_fmt,"
               "r_frame_rate,avg_frame_rate,nb_frames,sample_rate,channels")
    p = subprocess.run([ffprobe, "-v", "error", "-show_entries", entries, "-of", "json", video],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        die(f"ffprobe could not read the video:\n    {last_lines(p.stderr)}")
    data = json.loads(p.stdout or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        die("no video stream found")
    duration = float(data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        die("could not read the video duration")
    return {
        "container": data.get("format", {}).get("format_name"),
        "duration_s": duration,
        "video_codec": v.get("codec_name"),
        "video_profile": v.get("profile"),
        "width": v.get("width"),
        "height": v.get("height"),
        "pix_fmt": v.get("pix_fmt"),
        "r_frame_rate": parse_rate(v.get("r_frame_rate")),
        "avg_frame_rate": parse_rate(v.get("avg_frame_rate")),
        "audio_codec": a.get("codec_name") if a else None,
        "audio_sample_rate": int(a["sample_rate"]) if a and a.get("sample_rate") else None,
        "audio_channels": a.get("channels") if a else None,
    }


# --------------------------------------------------------------------------- 2. keyframes

def keyframe_interval(ffprobe: str, video: str, max_seconds: float | None) -> dict:
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
           "-show_entries", "frame=pts_time,best_effort_timestamp_time", "-of", "csv=p=0"]
    if max_seconds:
        cmd += ["-read_intervals", f"%+{max_seconds}"]
    cmd.append(video)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return {"ok": False, "error": last_lines(p.stderr), "ffprobe_wall_s": wall}

    times = []
    for line in p.stdout.splitlines():
        for field in line.split(","):          # first numeric field (pts_time, else best-effort)
            try:
                times.append(float(field))
                break
            except ValueError:
                continue
    times.sort()
    gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
    out = {"ok": True, "count": len(times), "ffprobe_wall_s": wall}
    if gaps:
        out.update(mean_s=statistics.fmean(gaps), median_s=statistics.median(gaps),
                   min_s=min(gaps), max_s=max(gaps))
    return out


# --------------------------------------------------------------------------- 3 + 4. decoding

def decode_cmd(ffmpeg: str, video: str, use_cuda: bool, max_seconds: float | None,
               keyframes_only: bool) -> list[str]:
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostats", "-progress", "pipe:1"]
    if use_cuda:
        # Frames stay in GPU memory: we measure pure NVDEC decode speed.
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", video, "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-"]
    return cmd


def run_decode(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    frames = re.findall(r"^frame=(\d+)", p.stdout, re.M)
    speeds = re.findall(r"^speed=\s*([\d.]+)x", p.stdout, re.M)
    return {
        "ok": p.returncode == 0,
        "wall_s": wall,
        "frames": int(frames[-1]) if frames else None,
        "ffmpeg_speed_x": float(speeds[-1]) if speeds else None,
        "error": last_lines(p.stderr) if p.returncode != 0 else "",
    }


def decode_with_fallback(ffmpeg: str, video: str, use_cuda: bool, max_seconds: float | None,
                         keyframes_only: bool, allow_cpu_fallback: bool) -> dict:
    res = run_decode(decode_cmd(ffmpeg, video, use_cuda, max_seconds, keyframes_only))
    res["device"] = "cuda" if use_cuda else "cpu"
    if not res["ok"] and use_cuda and allow_cpu_fallback:
        log(f"  ! GPU decode failed, retrying on CPU:\n    {res['error']}")
        cpu = run_decode(decode_cmd(ffmpeg, video, False, max_seconds, keyframes_only))
        cpu["device"] = "cpu"
        cpu["cuda_error"] = res["error"]
        return cpu
    return res


def ffmpeg_has_cuda(ffmpeg: str) -> bool:
    p = subprocess.run([ffmpeg, "-hide_banner", "-hwaccels"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return "cuda" in p.stdout.split()


def gpu_name() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.stdout.strip().splitlines()[0] if p.returncode == 0 and p.stdout.strip() else None


# --------------------------------------------------------------------------- 5. audio + VAD

def extract_audio(ffmpeg: str, video: str, max_seconds: float | None) -> tuple[np.ndarray | None, float, str]:
    cmd = [ffmpeg, "-hide_banner", "-v", "error", "-nostats"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", video, "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        return None, wall, last_lines(p.stderr.decode("utf-8", errors="replace"))
    return np.frombuffer(p.stdout, dtype=np.float32), wall, ""


def _sync(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _vad_batched(torch, model, audio: np.ndarray, device: str, segment_s: float) -> tuple[np.ndarray, dict]:
    """Silero VAD is a recurrent model that steps through audio 32 ms at a time.
    Stepping through 1 h one chunk at a time = 112,500 sequential calls (slow).
    Instead we cut the audio into `segment_s` pieces and run them side by side as one batch:
    1 h with 60 s pieces = 60 rows x 1,875 steps. The model state restarts at each piece
    boundary, which only affects the first fraction of a second of each piece."""
    steps = max(1, int(segment_s * SAMPLE_RATE) // VAD_CHUNK)
    n_chunks = max(1, math.ceil(len(audio) / VAD_CHUNK))
    rows = math.ceil(n_chunks / steps)
    padded = np.zeros(rows * steps * VAD_CHUNK, dtype=np.float32)
    padded[: len(audio)] = audio
    # [rows, steps, chunk] -> [steps, rows, chunk]: x[i] is one contiguous batch for step i
    x = torch.from_numpy(padded).view(rows, steps, VAD_CHUNK).transpose(0, 1).contiguous().to(device)
    model = model.to(device)

    with torch.inference_mode():
        t0 = time.perf_counter()
        model.reset_states()
        for i in range(min(3, steps)):        # TorchScript optimises during the first calls
            model(x[i], SAMPLE_RATE)
        _sync(torch, device)
        warmup = time.perf_counter() - t0

        model.reset_states()
        probs = torch.empty(steps, rows, device=device)
        t0 = time.perf_counter()
        for i in range(steps):
            probs[i] = model(x[i], SAMPLE_RATE).view(-1)
        flat = probs.transpose(0, 1).reshape(-1).cpu().numpy()   # back to time order (syncs GPU)
        infer = time.perf_counter() - t0

    return flat[:n_chunks], {"warmup_s": warmup, "infer_s": infer, "batch_rows": rows, "steps": steps}


def speech_regions(probs: np.ndarray) -> list[tuple[float, float]]:
    step = VAD_CHUNK / SAMPLE_RATE
    on = np.concatenate([[0], (probs >= VAD_THRESHOLD).astype(np.int8), [0]])
    d = np.diff(on)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    merged: list[list[float]] = []
    for s, e in zip(starts * step, ends * step):
        if merged and s - merged[-1][1] < VAD_MIN_GAP_S:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= VAD_MIN_SPEECH_S]


def run_vad(audio: np.ndarray, device_pref: str, segment_s: float) -> dict:
    res: dict = {}
    t0 = time.perf_counter()
    try:
        import torch
    except ImportError:
        return {"ok": False, "error": "torch not installed (see README)"}
    res["torch_import_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        from silero_vad import load_silero_vad
        model = load_silero_vad(onnx=False)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"silero-vad: {type(e).__name__}: {e}", **res}
    res["model_load_s"] = time.perf_counter() - t0

    device = device_pref
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        probs, timings = _vad_batched(torch, model, audio, device, segment_s)
    except Exception as e:  # noqa: BLE001
        if device == "cpu":
            return {"ok": False, "error": f"VAD failed: {type(e).__name__}: {e}", **res}
        res["cuda_error"] = f"{type(e).__name__}: {e}"
        log(f"  ! VAD on CUDA failed ({res['cuda_error']}), retrying on CPU")
        device = "cpu"
        probs, timings = _vad_batched(torch, load_silero_vad(onnx=False), audio, device, segment_s)

    t0 = time.perf_counter()
    regions = speech_regions(probs)
    timings["postprocess_s"] = time.perf_counter() - t0
    res.update(timings)
    res.update(ok=True, device=device, speech_regions=len(regions),
               speech_total_s=sum(e - s for s, e in regions))
    return res


# --------------------------------------------------------------------------- 6. projection

def project(media_s: float, full: dict, key: dict, audio_wall: float | None, vad: dict | None) -> dict:
    audio_h = 0.0
    if audio_wall is not None and vad and vad.get("ok"):
        audio_h = per_hour(audio_wall + vad["infer_s"] + vad["postprocess_s"], media_s)
    key_h = per_hour(key["wall_s"], media_s) if key["ok"] else math.nan
    full_h = per_hour(full["wall_s"], media_s) if full["ok"] else math.nan

    budget = TARGET_S_PER_HOUR - audio_h - key_h - PASS_D_OVERHEAD_S
    afford = max(0.0, budget / full_h) if full_h and not math.isnan(full_h) else math.nan
    estimates = {p: audio_h + key_h + p * full_h + PASS_D_OVERHEAD_S for p in FULL_DECODE_SHARES}

    if full.get("device") != "cuda" or key.get("device") != "cuda":
        verdict = "NOT MEANINGFUL - decoding ran on the CPU, run this on the NVIDIA machine"
    elif math.isnan(budget) or math.isnan(afford):
        verdict = "UNKNOWN - a measurement failed"
    elif budget <= 0:
        verdict = "NO - audio + keyframe passes alone already exceed 10 s/h"
    elif afford >= 0.20:
        verdict = "YES - comfortable"
    elif afford >= 0.05:
        verdict = "MAYBE - depends on how much of the video is 'unclear' and needs full decoding"
    else:
        verdict = "UNLIKELY - almost no budget left for full-decoding unclear stretches"
    return {"audio_vad_s_per_h": audio_h, "keyframes_s_per_h": key_h, "full_decode_s_per_h": full_h,
            "budget_for_full_decode_s_per_h": budget, "max_full_decode_share": afford,
            "estimates_s_per_h": {f"{int(p*100)}%": v for p, v in estimates.items()},
            "verdict": verdict}


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="Teams recording (MP4)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only benchmark the first N seconds (default: whole video); results are scaled to 1 h")
    ap.add_argument("--hwaccel", choices=["auto", "cuda", "none"], default="auto",
                    help="video decoding: auto = CUDA if available, else CPU (default: auto)")
    ap.add_argument("--vad-device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="device for Silero VAD (default: auto)")
    ap.add_argument("--vad-segment-s", type=float, default=60.0,
                    help="audio piece length for batched VAD (default: 60)")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="path to ffmpeg (default: from PATH)")
    ap.add_argument("--ffprobe", default="ffprobe", help="path to ffprobe (default: from PATH)")
    ap.add_argument("--json", type=Path, default=None, help="also save all results to this JSON file")
    args = ap.parse_args()

    video = str(Path(args.video))
    if not Path(video).is_file():
        die(f"video not found: {video}")
    ffmpeg = args.ffmpeg if Path(args.ffmpeg).is_file() else tool_path(args.ffmpeg)
    ffprobe = args.ffprobe if Path(args.ffprobe).is_file() else tool_path(args.ffprobe)
    t_start = time.perf_counter()
    results: dict = {"video": video, "max_seconds": args.max_seconds}

    # ---- 1. video info
    info = probe(ffprobe, video)
    results["info"] = info
    media_s = min(info["duration_s"], args.max_seconds) if args.max_seconds else info["duration_s"]
    results["media_s_measured"] = media_s

    log(f"\nBenchmark: {video}")
    log(f"  GPU: {gpu_name() or 'none found (nvidia-smi missing)'}")
    log("\n[1] Video")
    row("Container / codec", f"{info['container']} / {info['video_codec']} {info['video_profile'] or ''}")
    row("Resolution", f"{info['width']} x {info['height']}  ({info['pix_fmt']})")
    row("Duration", f"{info['duration_s']:.1f} s ({info['duration_s']/60:.1f} min)"
        + (f" - measuring first {media_s:.0f} s" if args.max_seconds else ""))
    r, avg = info["r_frame_rate"], info["avg_frame_rate"]
    row("Frame rate (r / avg)", f"{r or 0:.3f} / {avg or 0:.3f} fps")
    if r and avg and abs(r - avg) / r > 0.01:
        row("", "-> r and avg differ: probably VARIABLE frame rate (use real PTS, never index/fps)")
    row("Audio", f"{info['audio_codec']}, {info['audio_sample_rate']} Hz, {info['audio_channels']} ch"
        if info["audio_codec"] else "none")

    # ---- 2. keyframe interval
    log("\n[2] Keyframe interval (ffprobe -skip_frame nokey)")
    kf = keyframe_interval(ffprobe, video, args.max_seconds)
    results["keyframe_interval"] = kf
    if not kf["ok"]:
        row("FAILED", kf["error"])
    elif kf["count"] < 2:
        row("Keyframes found", f"{kf['count']} - too few to compute an interval")
    else:
        fps = avg or r
        row("Keyframes found", f"{kf['count']}  (~{kf['count'] / media_s * 3600:.0f} per hour)")
        row("Interval median / mean", f"{kf['median_s']:.2f} s / {kf['mean_s']:.2f} s"
            + (f"  (~{kf['median_s'] * fps:.0f} frames per GOP)" if fps else ""))
        row("Interval min / max", f"{kf['min_s']:.2f} s / {kf['max_s']:.2f} s")
    row("(ffprobe time, CPU, info only)", f"{kf['ffprobe_wall_s']:.2f} s")

    # ---- 3 + 4. decode speeds
    want_cuda = args.hwaccel == "cuda" or (args.hwaccel == "auto" and gpu_name() and ffmpeg_has_cuda(ffmpeg))
    allow_fallback = args.hwaccel == "auto"
    log(f"\n[3] Full decode, every frame ({'GPU / NVDEC' if want_cuda else 'CPU'})")
    full = decode_with_fallback(ffmpeg, video, bool(want_cuda), args.max_seconds, False, allow_fallback)
    results["full_decode"] = full
    if full["ok"]:
        row("Device", full["device"])
        row("Frames decoded", f"{full['frames']}")
        row("Wall time", f"{full['wall_s']:.2f} s  ->  {per_hour(full['wall_s'], media_s):.1f} s per hour")
        row("Speed", f"{media_s / full['wall_s']:.1f}x realtime  (ffmpeg reports speed={full['ffmpeg_speed_x']}x)"
            + (f", {full['frames'] / full['wall_s']:.0f} fps" if full["frames"] else ""))
    else:
        row("FAILED", full["error"])

    log(f"\n[4] Keyframe-only decode ({'GPU / NVDEC' if want_cuda else 'CPU'})")
    key = decode_with_fallback(ffmpeg, video, bool(want_cuda), args.max_seconds, True, allow_fallback)
    results["keyframe_decode"] = key
    if key["ok"]:
        row("Device", key["device"])
        row("Keyframes decoded", f"{key['frames']}")
        row("Wall time", f"{key['wall_s']:.2f} s  ->  {per_hour(key['wall_s'], media_s):.1f} s per hour")
        if key["frames"]:
            row("Per keyframe", f"{key['wall_s'] / key['frames'] * 1000:.1f} ms")
    else:
        row("FAILED", key["error"])

    # ---- 5. audio + VAD
    log("\n[5] Audio extraction + Silero VAD")
    audio_wall, vad = None, None
    if not info["audio_codec"]:
        row("Skipped", "video has no audio stream")
    else:
        audio, audio_wall, err = extract_audio(ffmpeg, video, args.max_seconds)
        results["audio_extract_s"] = audio_wall
        if audio is None:
            row("Audio extraction FAILED", err)
        else:
            row("Extraction (16 kHz mono)", f"{audio_wall:.2f} s  ->  {per_hour(audio_wall, media_s):.1f} s per hour")
            vad = run_vad(audio, args.vad_device, args.vad_segment_s)
            results["vad"] = vad
            if not vad.get("ok"):
                row("VAD FAILED", vad.get("error", "?"))
            else:
                row("Device", f"{vad['device']}  (batch of {vad['batch_rows']} x {vad['steps']} steps)")
                row("VAD inference", f"{vad['infer_s']:.2f} s  ->  {per_hour(vad['infer_s'], media_s):.1f} s per hour")
                row("Speech found (rough)", f"{vad['speech_regions']} regions, {vad['speech_total_s']:.0f} s "
                    f"({vad['speech_total_s'] / media_s * 100:.0f}% of audio)")
                row("Startup: torch import", f"{vad['torch_import_s']:.2f} s  (one-time, not per hour)")
                row("Startup: model load + warm-up", f"{vad['model_load_s'] + vad['warmup_s']:.2f} s  (one-time)")

    # ---- 6. projection
    proj = project(media_s, full, key, audio_wall, vad)
    results["projection"] = proj
    log("\n[6] Projection for 1 hour of video (fast pipeline, startup excluded)")
    row("Pass A  audio + VAD", f"{proj['audio_vad_s_per_h']:.1f} s")
    row("Pass B  keyframes (decode only)", f"{proj['keyframes_s_per_h']:.1f} s")
    row("Pass D  segments + CSV (guess)", f"{PASS_D_OVERHEAD_S:.1f} s")
    row("Full decode of 100% (baseline)", f"{proj['full_decode_s_per_h']:.1f} s")
    row("Left for Pass C (target 10 s)", f"{proj['budget_for_full_decode_s_per_h']:.1f} s")
    if not math.isnan(proj["max_full_decode_share"]):
        row("-> max share you can fully decode", f"{proj['max_full_decode_share'] * 100:.1f}% of the video")
    for share, est in proj["estimates_s_per_h"].items():
        mark = "ok" if est <= TARGET_S_PER_HOUR else "too slow"
        row(f"If {share} of video is 'unclear'", f"{est:.1f} s per hour  ({mark})")
    log(f"\n  VERDICT: 10 s/h reachable?  {proj['verdict']}")
    log("  Notes: keyframe time does not yet include copying frames to RAM or the ring check;"
        "\n         startup (torch import, model load) is reported above but excluded from the per-hour numbers.")

    total = time.perf_counter() - t_start
    results["benchmark_total_s"] = total
    log(f"\n(benchmark itself took {total:.1f} s)")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        log(f"Results saved to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
