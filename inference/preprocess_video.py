"""Extract frames from a surgical video so they can be fed to ATLAS.

Frames are decoded with ffmpeg (`module load ffmpeg/8.1`, or the
imageio-ffmpeg binary as a fallback),
optionally resampled to a lower frame rate and cropped to the endoscope field of
view, and written as JPGs (same format as the ATLAS-120k images) together with a
`meta.json` that the inference script reads.

Example:
    python inference/preprocess_video.py \
        --video materials/videos/my_video.mp4 \
        --out_dir /path/to/output/my_video/frames \
        --fps 30 --crop none
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_exe():
    """System ffmpeg if available, else the binary bundled with imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found (module load ffmpeg/8.1, or pip install imageio-ffmpeg)")


def probe_video(video_path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(video_path))
        meta = next(reader)
        reader.close()
        return {
            "width": int(meta["size"][0]),
            "height": int(meta["size"][1]),
            "fps": float(meta["fps"]),
            "nb_frames": int(round(meta["duration"] * meta["fps"])),
            "duration": float(meta["duration"]),
        }
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "json", str(video_path),
    ]
    stream = json.loads(subprocess.check_output(cmd))["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den),
        "nb_frames": int(stream.get("nb_frames", 0) or 0),
        "duration": float(stream.get("duration", 0) or 0),
    }


def read_gray_samples(video_path: Path, info: dict, num_samples: int = 20) -> np.ndarray:
    """Decode a few evenly spaced frames as grayscale (H, W) arrays."""
    w, h = info["width"], info["height"]
    samples = []
    for t in np.linspace(0.05, 0.95, num_samples) * info["duration"]:
        cmd = [
            ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        if len(raw) == w * h:
            samples.append(np.frombuffer(raw, np.uint8).reshape(h, w))
    return np.stack(samples)


def detect_crop(video_path: Path, info: dict, thresh: int = 15, margin: int = 4):
    """Bounding box of the non-black area (endoscope field of view)."""
    max_img = read_gray_samples(video_path, info).max(axis=0)
    rows = np.where((max_img > thresh).mean(axis=1) > 0.02)[0]
    cols = np.where((max_img > thresh).mean(axis=0) > 0.02)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    y0, y1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, info["height"])
    x0, x1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, info["width"])
    # Even sizes keep yuv420 encoders happy when writing the overlay video
    return int(x0), int(y0), int((x1 - x0) // 2 * 2), int((y1 - y0) // 2 * 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True, help="Folder for frame_XXXXXX.jpg + meta.json")
    parser.add_argument("--fps", type=float, default=0, help="Output frame rate (0 = keep native)")
    parser.add_argument("--crop", choices=["none", "auto"], default="none",
                        help="'auto' crops away the black border around the endoscope view")
    parser.add_argument("--max_side", type=int, default=0,
                        help="Downscale so the longest side is at most this (0 = keep). "
                             "The model works at 256 px on the short side anyway.")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=0, help="Seconds to extract (0 = until the end)")
    parser.add_argument("--jpg_quality", type=int, default=2, help="ffmpeg -q:v (2 = high quality)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.glob("frame_*.jpg")):
        if not args.overwrite:
            raise SystemExit(f"{args.out_dir} already contains frames (use --overwrite)")
        for f in args.out_dir.glob("frame_*.jpg"):
            f.unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(args.video)
    print(f"Video: {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
          f"{info['duration']:.1f}s, {info['nb_frames']} frames")

    filters = []
    crop = None
    if args.crop == "auto":
        crop = detect_crop(args.video, info)
        if crop is not None:
            x, y, w, h = crop
            filters.append(f"crop={w}:{h}:{x}:{y}")
            print(f"Auto crop: x={x} y={y} w={w} h={h}")
        else:
            print("Auto crop: no border detected, keeping full frame")
    out_fps = args.fps if args.fps > 0 else info["fps"]
    if args.fps > 0:
        filters.append(f"fps={args.fps}")
    if args.max_side > 0:
        s = args.max_side
        filters.append(f"scale='if(gt(iw,ih),min({s},iw),-2)':'if(gt(iw,ih),-2,min({s},ih))'")

    cmd = [ffmpeg_exe(), "-v", "error", "-stats"]
    if args.start > 0:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", str(args.video)]
    if args.duration > 0:
        cmd += ["-t", str(args.duration)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-an", "-q:v", str(args.jpg_quality), "-start_number", "0",
            str(args.out_dir / "frame_%06d.jpg")]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    frames = sorted(args.out_dir.glob("frame_*.jpg"))
    meta = {
        "video": str(args.video.resolve()),
        "source": info,
        "fps": out_fps,
        "start": args.start,
        "crop_xywh": crop,
        "num_frames": len(frames),
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {len(frames)} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
