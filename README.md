# Teams Speaker Timeline

Give it a Microsoft Teams meeting recording (MP4) and it produces a CSV of **who was speaking and when**.
It detects the purple ring Teams draws around whoever is talking, and uses the audio to skip the moments
when nobody speaks.

**Status:** Phase 0: set up the GPU machine and measure how fast it is.

---

## Phase 0: set up the GPU machine and run the benchmark

Do all of this **on the machine with the NVIDIA GPU**, not on the development PC.

### 1. Install the tools (once)

```powershell
winget install Python.Python.3.11
winget install astral-sh.uv
winget install Gyan.FFmpeg
```

Then **close and reopen VS Code / the terminal** so the new programs are found.

You also need a recent NVIDIA driver. Check it with `nvidia-smi` in step 3.

### 2. Get the code and install the Python packages

```powershell
git clone https://github.com/alhassanradwan/teams_timestamp_detection.git
cd teams_timestamp_detection
uv sync
```

`uv sync` creates `.venv` and installs numpy, PyTorch (CUDA build) and Silero VAD.
The CUDA build of PyTorch comes from PyTorch's own server, which is set up in `pyproject.toml`.

### 3. Check the machine

Run each command and compare with "what you should see".

| # | Command | What you should see |
|---|---------|---------------------|
| 1 | `nvidia-smi` | Your GPU's name. Memory usage, e.g. `1500MiB / 4096MiB`: at least **~2 GB free** |
| 2 | `ffmpeg -hide_banner -hwaccels` | A list that includes **`cuda`** |
| 3 | `ffmpeg -hide_banner -v error -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 3 -c:v libx264 test.mp4` | Nothing (creates `test.mp4`, a 3 s test video) |
| 4 | `ffmpeg -hide_banner -v error -hwaccel cuda -hwaccel_output_format cuda -i test.mp4 -f null -` | **Nothing.** Any text means GPU decoding does not work |
| 5 | `uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `True` and your GPU's name |
| 6 | `uv run python -c "from silero_vad import load_silero_vad; load_silero_vad(); print('VAD OK')"` | `VAD OK` |

Delete `test.mp4` afterwards. If a check fails, send me the command and its output.

### 4. Run the benchmark

```powershell
uv run python scripts/benchmark.py "C:\path\to\meeting.mp4"
```

For a quick first try, use only the first 10 minutes (results are still scaled to 1 hour):

```powershell
uv run python scripts/benchmark.py "C:\path\to\meeting.mp4" --max-seconds 600
```

Options:

- `--device cpu` decodes and runs VAD on the CPU (for a PC without an NVIDIA GPU). The default is `--device gpu`.
- `--gpu-memory-gb 2` is the most GPU memory to use (only with `--device gpu`). The benchmark shows whether
  it stayed within the limit.

#### Running on the CPU (development PC)

```powershell
uv run python scripts/benchmark.py "C:\path\to\meeting.mp4" --device cpu --max-seconds 600
```

The times will be much slower than on the GPU machine. Use this run to check the keyframe gap and the
% of speech, not the speed.

It measures and prints:

| Step | What it measures | Why we need it |
|------|------------------|----------------|
| [1] Video | resolution, fps, length | confirms the input is what we expect (1080p, 30 fps) |
| [2] Keyframes | how far apart keyframes are | the video can only be decoded starting at a keyframe, so this limits how precisely we can skip |
| [3] Full GPU decode | time + VRAM to decode every frame | cost of the simple approach |
| [4] Keyframe-only decode | time to decode only keyframes | cost of the fastest approach |
| [5] Audio + VAD | audio extraction time, VAD time (CPU and GPU), % speech, % video we can skip | how much the "audio first, skip silence" idea saves |
| [6] Estimate | seconds per hour of video for approaches A, B, C | is the 10 s/hour target reachable? |

The three approaches in step [6]:

- **A. Decode every frame:** simplest, slowest.
- **B. Audio first, skip silence:** decode only where someone speaks (plus a 0.5 s margin).
- **C. Audio + keyframes only:** the best case; the real fast pipeline lands between B and C.

### 5. Send me the output

Copy the whole output into the chat. If you can, run it on **2–3 different recordings**, because the keyframe gaps
and the amount of silence differ between meetings. The numbers decide which approach we build.
