#!/usr/bin/env python
"""Phase 0: check that this machine has everything the pipeline needs.

Checks: OS + Python, NVIDIA GPU (nvidia-smi), FFmpeg/ffprobe with CUDA (NVDEC)
decoding, and the Python packages (numpy, torch with CUDA, silero-vad).

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --ffmpeg C:/ffmpeg/bin/ffmpeg.exe --ffprobe C:/ffmpeg/bin/ffprobe.exe

Exit code is 0 when there are no FAIL lines.
"""
from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MIN_VRAM_GB = 6.0
MIN_FFMPEG_MAJOR = 6  # needed for "-fps_mode"

results: list[tuple[str, str, str]] = []  # (status, check, detail)


def report(status: str, check: str, detail: str = "") -> None:
    results.append((status, check, detail))
    print(f"[{status:<4}] {check:<28} {detail}", flush=True)


def run(cmd: list[str], timeout: float = 120) -> tuple[int | None, str, str]:
    """Run a command; return (returncode, stdout, stderr). returncode None = could not start."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return None, "", "not found"
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout:.0f} s"


def last_lines(text: str, n: int = 3) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " | ".join(lines[-n:])


# --------------------------------------------------------------------------- checks

def check_system() -> None:
    print("\n== System ==")
    report("OK", "OS", platform.platform())
    v = sys.version_info
    status = "OK" if (3, 10) <= (v.major, v.minor) <= (3, 12) else "WARN"
    report(status, "Python", f"{platform.python_version()}  ({sys.executable})")
    in_venv = sys.prefix != sys.base_prefix
    report("OK" if in_venv else "WARN", "Virtual environment",
           sys.prefix if in_venv else "not in a venv (recommended: .venv)")


def check_gpu() -> None:
    print("\n== NVIDIA GPU ==")
    if not shutil.which("nvidia-smi"):
        report("FAIL", "nvidia-smi", "not found - NVIDIA driver missing or not on PATH")
        return
    rc, out, err = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                        "--format=csv,noheader,nounits"])
    if rc != 0 or not out.strip():
        report("FAIL", "nvidia-smi", last_lines(err or out))
        return
    for i, line in enumerate(out.strip().splitlines()):
        name, total, used, driver = [x.strip() for x in line.split(",")]
        total_gb = float(total) / 1024
        status = "OK" if total_gb >= MIN_VRAM_GB - 0.5 else "WARN"
        report(status, f"GPU {i}", f"{name}, {total_gb:.1f} GB VRAM ({float(used)/1024:.1f} GB used), "
                                   f"driver {driver}")
    rc, out, _ = run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    report("OK", "Driver CUDA version", m.group(1) if m else "unknown")


def check_ffmpeg(ffmpeg: str, ffprobe: str) -> None:
    print("\n== FFmpeg ==")
    rc, out, err = run([ffmpeg, "-hide_banner", "-version"])
    if rc is None:
        report("FAIL", "ffmpeg", f"'{ffmpeg}' not found - install FFmpeg or pass --ffmpeg")
        return
    first = out.splitlines()[0] if out else "?"
    m = re.search(r"version\s+n?(\d+)\.", first)
    if m:
        major = int(m.group(1))
        status = "OK" if major >= MIN_FFMPEG_MAJOR else "FAIL"
        report(status, "ffmpeg version", f"{first}  (need >= {MIN_FFMPEG_MAJOR})")
    else:
        report("WARN", "ffmpeg version", f"{first}  (could not parse; need >= {MIN_FFMPEG_MAJOR})")

    rc, out, _ = run([ffprobe, "-hide_banner", "-version"])
    report("OK" if rc == 0 else "FAIL", "ffprobe",
           out.splitlines()[0] if rc == 0 and out else f"'{ffprobe}' not found - pass --ffprobe")

    rc, out, _ = run([ffmpeg, "-hide_banner", "-hwaccels"])
    has_cuda = "cuda" in out.split()
    report("OK" if has_cuda else "FAIL", "hwaccel 'cuda' listed",
           "yes" if has_cuda else "no - this FFmpeg build has no NVDEC support")

    rc, out, _ = run([ffmpeg, "-hide_banner", "-decoders"])
    cuvid = sorted(set(re.findall(r"\b(\w+_cuvid)\b", out)))
    report("OK" if cuvid else "WARN", "cuvid decoders", ", ".join(cuvid) or "none (optional)")

    if has_cuda:
        functional_decode_test(ffmpeg, ffprobe)


def functional_decode_test(ffmpeg: str, ffprobe: str) -> None:
    """Encode a 2 s 1080p H.264 clip, then really decode it on the GPU."""
    with tempfile.TemporaryDirectory() as tmp:
        clip = str(Path(tmp) / "test_1080p.mp4")
        rc, _, err = run([ffmpeg, "-hide_banner", "-v", "error", "-f", "lavfi",
                          "-i", "testsrc2=size=1920x1080:rate=30", "-t", "2",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "30", clip])
        if rc != 0:
            report("WARN", "GPU decode test", f"could not create test clip (libx264?): {last_lines(err)}")
            return

        rc, _, err = run([ffmpeg, "-hide_banner", "-v", "error",
                          "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                          "-i", clip, "-f", "null", "-"])
        if rc == 0 and not err.strip():
            report("OK", "GPU decode test", "decoded a 1080p H.264 clip with NVDEC")
        else:
            report("FAIL", "GPU decode test", last_lines(err) or f"exit code {rc}")

        rc, out, err = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                            "-show_entries", "frame=pts_time", "-of", "csv=p=0", clip])
        n_key = len([ln for ln in out.splitlines() if ln.strip()])
        report("OK" if rc == 0 and n_key == 2 else "WARN", "ffprobe -skip_frame nokey",
               f"{n_key} keyframes found in test clip (expected 2)" if rc == 0 else last_lines(err))


def check_python_packages() -> None:
    print("\n== Python packages ==")
    try:
        import numpy
        report("OK", "numpy", numpy.__version__)
    except ImportError:
        report("FAIL", "numpy", "not installed (pip install -r requirements.txt)")

    try:
        import torch
    except ImportError:
        report("FAIL", "torch", "not installed (see README: install the CUDA build first)")
        return
    built_cuda = torch.version.cuda
    report("OK" if built_cuda else "FAIL", "torch", f"{torch.__version__}  "
           + (f"(built for CUDA {built_cuda})" if built_cuda else "(CPU-only build - reinstall CUDA build)"))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        try:
            x = torch.ones(1024, 1024, device="cuda")
            float((x @ x).sum())
            report("OK", "torch.cuda", f"{props.name}, {props.total_memory / 1024**3:.1f} GB - test op ran")
        except Exception as e:  # noqa: BLE001
            report("FAIL", "torch.cuda", f"available but test op failed: {e}")
    else:
        report("FAIL", "torch.cuda", "torch.cuda.is_available() is False")

    try:
        from importlib.metadata import version
        from silero_vad import load_silero_vad
        load_silero_vad(onnx=False)
        report("OK", "silero-vad", f"{version('silero-vad')} - model loads (offline)")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "silero-vad", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ffmpeg", default="ffmpeg", help="path to ffmpeg (default: from PATH)")
    ap.add_argument("--ffprobe", default="ffprobe", help="path to ffprobe (default: from PATH)")
    args = ap.parse_args()

    check_system()
    check_gpu()
    check_ffmpeg(args.ffmpeg, args.ffprobe)
    check_python_packages()

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")
    print(f"\n== Summary: {n_fail} FAIL, {n_warn} WARN ==")
    if n_fail:
        print("Fix the FAIL lines above before running scripts/benchmark.py.")
    else:
        print("Environment looks ready. Next: python scripts/benchmark.py <video.mp4>")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
