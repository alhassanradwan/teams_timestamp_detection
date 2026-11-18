"""Phase 0 benchmark: can this machine process 1 hour of Teams recording in about 10 seconds?

Run it on the GPU machine with one real recording:

    uv run python scripts/benchmark.py meeting.mp4
    uv run python scripts/benchmark.py meeting.mp4 --max-seconds 600    # only the first 10 minutes
    uv run python scripts/benchmark.py meeting.mp4 --device cpu         # no NVIDIA GPU (laptop)
    uv run python scripts/benchmark.py meeting.mp4 --gpu-memory-gb 2    # GPU memory limit

It measures five things and prints each result as it goes:
    [1] Video info             resolution, frame rate, length
    [2] Keyframes              how far apart they are
    [3] Full decode            every frame: time (and VRAM on GPU)
    [4] Keyframe-only decode   only the keyframes: time
    [5] Audio + VAD            extract the audio, find where people speak
Then [6] estimates the processing time per hour of video for three approaches.

Needs: ffmpeg and ffprobe on PATH (with CUDA support), numpy, torch, silero-vad.
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

TARGET_SECONDS_PER_HOUR = 10.0

SAMPLE_RATE = 16000           # Silero VAD works on 16 kHz mono audio
VAD_CHUNK = 512               # Silero VAD reads 512 samples (= 32 ms) per step
VAD_PIECE_SECONDS = 60        # audio is cut into 60 s pieces that run side by side
VAD_THRESHOLD = 0.5           # speech probability above this counts as speech
MIN_PAUSE_SECONDS = 0.8       # shorter pauses do not split speech
MIN_SPEECH_SECONDS = 0.25     # shorter bits of "speech" are ignored
SPEECH_MARGIN_SECONDS = 0.5   # extra video kept before and after each speech region


# ------------------------------------------------------------------ helpers

def run(cmd):
    """Run a command and return its result, with stdout/stderr as text."""
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def gpu_memory_used_mb():
    """GPU memory in use right now (all programs), in MB. None if nvidia-smi is missing."""
    try:
        result = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        return int(result.stdout.split()[0])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def per_hour(seconds, media_seconds):
    """Scale a time measured on `media_seconds` of video up to 1 hour of video."""
    return seconds / media_seconds * 3600


def within_limit(used_mb, limit_gb):
    """Text saying whether `used_mb` stays under the GPU memory limit."""
    if used_mb is None:
        return "unknown"
    if used_mb <= limit_gb * 1024:
        return f"within the {limit_gb} GB limit"
    return f"OVER the {limit_gb} GB limit"


# ------------------------------------------------------------------ [1] video info

def video_info(video):
    result = run(["ffprobe", "-v", "error", "-of", "json", "-show_entries",
                  "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate", video])
    if result.returncode != 0:
        sys.exit(f"ffprobe could not read the video:\n{result.stderr}")

    data = json.loads(result.stdout)
    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    if video_stream is None or not has_audio:
        sys.exit("The file needs both a video and an audio stream.")

    num, den = video_stream["avg_frame_rate"].split("/")
    return {
        "codec": video_stream["codec_name"],
        "width": video_stream["width"],
        "height": video_stream["height"],
        "fps": float(num) / float(den) if float(den) else 0.0,
        "duration": float(data["format"]["duration"]),
    }


# ------------------------------------------------------------------ [2] keyframes

def keyframe_times(video, max_seconds):
    """Times (in seconds) of all keyframes, found with ffprobe -skip_frame nokey."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
           "-show_entries", "frame=pts_time", "-of", "csv=p=0"]
    if max_seconds:
        cmd += ["-read_intervals", f"%+{max_seconds}"]
    result = run(cmd + [video])

    times = []
    for line in result.stdout.split():
        try:
            times.append(float(line.strip(",")))
        except ValueError:
            pass  # a frame without a timestamp ("N/A")
    return sorted(times)


# ------------------------------------------------------------------ [3] + [4] decoding

def time_decode(video, max_seconds, keyframes_only, device):
    """Decode the video on the CPU or GPU and throw the frames away.
    Returns (seconds it took, extra VRAM used in MB - None on CPU)."""
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostats"]
    if device == "gpu":
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]   # decode on GPU, keep frames there
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", video, "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-"]

    vram_before = gpu_memory_used_mb()
    vram_peak = vram_before
    with tempfile.TemporaryFile() as error_file:
        start = time.perf_counter()
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=error_file)
        while True:  # check the VRAM every 0.5 s until FFmpeg is done
            used = gpu_memory_used_mb()
            if used is not None:
                vram_peak = max(vram_peak, used)
            try:
                process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
        seconds = time.perf_counter() - start
        error_file.seek(0)
        errors = error_file.read().decode("utf-8", errors="replace")

    if process.returncode != 0:
        sys.exit(f"Decoding on {device.upper()} failed:\n{errors}")
    extra_vram = vram_peak - vram_before if vram_before is not None else None
    return seconds, extra_vram


# ------------------------------------------------------------------ [5] audio + VAD

def extract_audio(video, max_seconds):
    """Extract the audio as 16 kHz mono float32. Returns (samples, seconds it took)."""
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostats"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-i", video, "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True)
    seconds = time.perf_counter() - start
    if result.returncode != 0:
        sys.exit(f"Audio extraction failed:\n{result.stderr.decode('utf-8', errors='replace')}")
    return np.frombuffer(result.stdout, dtype=np.float32), seconds


def vad_probabilities(model, audio, device):
    """Speech probability for every 32 ms of audio. Returns (probabilities, seconds it took).

    Silero VAD reads the audio 32 ms at a time and remembers what it heard before, so
    1 hour = 112,500 small steps one after another, which is slow. Instead we cut the audio
    into 60 s pieces and feed all pieces at once as one batch: 1 hour = 60 pieces side by
    side, only 1,875 steps. (Each piece starts with an empty memory; that only affects its
    first fraction of a second.)
    """
    import torch

    steps = int(VAD_PIECE_SECONDS * SAMPLE_RATE) // VAD_CHUNK   # steps per piece (1,875)
    n_chunks = max(1, math.ceil(len(audio) / VAD_CHUNK))
    n_pieces = math.ceil(n_chunks / steps)

    padded = np.zeros(n_pieces * steps * VAD_CHUNK, dtype=np.float32)
    padded[:len(audio)] = audio
    # [pieces, steps, chunk] -> [steps, pieces, chunk], so batch[i] = step i of every piece
    batch = torch.from_numpy(padded).view(n_pieces, steps, VAD_CHUNK).transpose(0, 1).contiguous()
    batch = batch.to(device)
    model.to(device)

    with torch.inference_mode():
        model.reset_states()
        for i in range(min(3, steps)):   # warm-up: the first calls are slow
            model(batch[i], SAMPLE_RATE)
        model.reset_states()

        start = time.perf_counter()
        probs = torch.empty(steps, n_pieces, device=device)
        for i in range(steps):
            probs[i] = model(batch[i], SAMPLE_RATE).view(-1)
        probs = probs.transpose(0, 1).reshape(-1).cpu().numpy()   # back to time order
        seconds = time.perf_counter() - start

    return probs[:n_chunks], seconds


def speech_regions(probs):
    """Turn per-32-ms probabilities into a list of (start, end) speech regions in seconds."""
    step = VAD_CHUNK / SAMPLE_RATE
    regions = []
    start = None
    for i, p in enumerate(probs):
        if p >= VAD_THRESHOLD and start is None:
            start = i * step
        elif p < VAD_THRESHOLD and start is not None:
            regions.append([start, i * step])
            start = None
    if start is not None:
        regions.append([start, len(probs) * step])

    merged = []
    for start, end in regions:          # join regions separated by a short pause
        if merged and start - merged[-1][1] < MIN_PAUSE_SECONDS:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= MIN_SPEECH_SECONDS]


def skippable_seconds(regions, media_seconds, keyframe_gap):
    """How much video we never need to decode.
    = every silence (after adding the margin around speech), minus one keyframe gap,
    because decoding has to restart at the keyframe before the next speech."""
    padded = []
    for start, end in regions:
        start = max(0.0, start - SPEECH_MARGIN_SECONDS)
        end = min(media_seconds, end + SPEECH_MARGIN_SECONDS)
        if padded and start <= padded[-1][1]:
            padded[-1][1] = max(padded[-1][1], end)
        else:
            padded.append([start, end])

    # silences: from 0 to the first region, between regions, and from the last region to the end
    edges = [0.0] + [t for region in padded for t in region] + [media_seconds]
    silences = [edges[i + 1] - edges[i] for i in range(0, len(edges), 2)]
    return sum(max(0.0, silence - keyframe_gap) for silence in silences)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description="Phase 0 benchmark (details at the top of this file).")
    parser.add_argument("video", help="path to the Teams recording (MP4)")
    parser.add_argument("--max-seconds", type=float, help="only use the first N seconds of the video")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                        help="decode and run VAD on the CPU or the GPU (default: gpu)")
    parser.add_argument("--gpu-memory-gb", type=float, default=2.0,
                        help="max GPU memory to use, only with --device gpu (default: 2)")
    args = parser.parse_args()
    video, max_seconds, device = args.video, args.max_seconds, args.device

    if device == "gpu" and gpu_memory_used_mb() is None:
        sys.exit("GPU selected but no NVIDIA GPU was found (nvidia-smi is missing). Use --device cpu.")

    # [1] Video info
    info = video_info(video)
    media = min(info["duration"], max_seconds) if max_seconds else info["duration"]
    print(f"\n[1] Video: {video}")
    print(f"    {info['width']}x{info['height']} {info['codec']}, {info['fps']:.2f} fps, "
          f"{info['duration'] / 60:.1f} min")
    if max_seconds:
        print(f"    measuring the first {media:.0f} s only - all times below are scaled to 1 hour")
    if device == "gpu":
        print(f"    GPU memory already in use before we start: {gpu_memory_used_mb()} MB "
              f"(our limit: {args.gpu_memory_gb} GB)")

    # [2] Keyframes
    print("\n[2] Keyframes")
    times = keyframe_times(video, max_seconds)
    gaps = [b - a for a, b in zip(times, times[1:])]
    keyframe_gap = statistics.median(gaps) if gaps else media
    print(f"    {len(times)} keyframes, typical gap {keyframe_gap:.2f} s "
          f"(shortest {min(gaps, default=0):.2f} s, longest {max(gaps, default=0):.2f} s)")

    # [3] Full decode
    print(f"\n[3] Full decode on {device.upper()} (every frame)")
    full_s, full_vram = time_decode(video, max_seconds, keyframes_only=False, device=device)
    print(f"    {full_s:.1f} s = {media / full_s:.0f}x real time -> {per_hour(full_s, media):.1f} s per hour")
    if device == "gpu":
        print(f"    extra VRAM used: {full_vram} MB ({within_limit(full_vram, args.gpu_memory_gb)})")

    # [4] Keyframe-only decode
    print(f"\n[4] Keyframe-only decode on {device.upper()}")
    key_s, _ = time_decode(video, max_seconds, keyframes_only=True, device=device)
    print(f"    {key_s:.1f} s -> {per_hour(key_s, media):.1f} s per hour "
          f"({key_s / max(1, len(times)) * 1000:.1f} ms per keyframe)")

    # [5] Audio + VAD
    print("\n[5] Audio + VAD")
    audio, audio_s = extract_audio(video, max_seconds)
    print(f"    audio extraction:  {audio_s:.1f} s -> {per_hour(audio_s, media):.1f} s per hour")

    vram_before_torch = gpu_memory_used_mb()
    start = time.perf_counter()
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad()
    print(f"    startup (import torch + load VAD model, once per run): {time.perf_counter() - start:.1f} s")

    probs, vad_s = vad_probabilities(model, audio, "cpu")
    print(f"    VAD on CPU:        {vad_s:.1f} s -> {per_hour(vad_s, media):.1f} s per hour")
    if device == "gpu" and torch.cuda.is_available():
        # PyTorch's built-in cap: it stops with an out-of-memory error instead of going over the limit
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_memory_gb / total_gb))
        try:
            _, vad_gpu_s = vad_probabilities(model, audio, "cuda")
            vram = gpu_memory_used_mb() - vram_before_torch if vram_before_torch is not None else None
            print(f"    VAD on GPU:        {vad_gpu_s:.1f} s -> {per_hour(vad_gpu_s, media):.1f} s per hour "
                  f"(VRAM incl. CUDA startup: {vram} MB, {within_limit(vram, args.gpu_memory_gb)})")
            vad_s = min(vad_s, vad_gpu_s)
        except Exception as error:
            print(f"    VAD on GPU failed, using the CPU result ({type(error).__name__}: {error})")

    regions = speech_regions(probs)
    speech_s = sum(end - start for start, end in regions)
    skip_s = skippable_seconds(regions, media, keyframe_gap)
    print(f"    speech: {speech_s / media:.0%} of the meeting ({len(regions)} speech regions)")
    print(f"    video we can skip: {skip_s / media:.0%} "
          f"(silences with a {SPEECH_MARGIN_SECONDS} s margin, minus one keyframe gap each)")

    # [6] Estimates
    audio_h = per_hour(audio_s + vad_s, media)
    full_h = per_hour(full_s, media)
    plan_a = full_h
    plan_b = audio_h + full_h * (1 - skip_s / media)
    plan_c = audio_h + per_hour(key_s, media)

    print("\n[6] Estimated processing time for 1 hour of video (startup not included)")
    for name, value in [("A. Decode every frame", plan_a),
                        ("B. Audio first, skip silence", plan_b),
                        ("C. Audio + keyframes only (best case)", plan_c)]:
        status = "OK" if value <= TARGET_SECONDS_PER_HOUR else "too slow"
        print(f"    {name:<40} {value:6.1f} s   {status}")

    print("\n    Verdict: ", end="")
    if plan_a <= TARGET_SECONDS_PER_HOUR:
        print("YES - even decoding every frame is fast enough.")
    elif plan_b <= TARGET_SECONDS_PER_HOUR:
        print("YES - with 'audio first, skip silence'.")
    elif plan_c <= TARGET_SECONDS_PER_HOUR:
        print(f"MAYBE - only with the keyframe approach. It leaves "
              f"{TARGET_SECONDS_PER_HOUR - plan_c:.1f} s per hour for fully decoding unclear parts.")
    else:
        print("NO - even the best case is over 10 s per hour on this machine.")

    if device == "cpu":
        print("\n    Note: CPU run - the times are NOT the GPU machine's. "
              "Keyframe gap, speech % and skippable % are still correct.")


if __name__ == "__main__":
    main()
