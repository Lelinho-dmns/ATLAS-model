"""Summarise the outputs of infer_video.py and log them to Weights & Biases.

Since the video has no ground-truth annotations, the quantitative results are
descriptive: how much of the frame each class covers over time, how often each
class is present, the model's confidence and the procedure recognition. The
qualitative results are a set of representative frames with the predicted
organs coloured and labelled.

Writes (in --pred_dir):
    representative_frames/   original | prediction figures, one per selected frame
    plots/                   class coverage, presence, confidence and procedure plots
    class_summary.csv        per-class statistics over the whole video
    summary.json             overall statistics
and logs all of it (plus a downscaled overlay video) to a W&B run.
"""

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

PRESENCE_THRESH = 0.005  # a class counts as present in a frame above 0.5% of the pixels


def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], np.array(rows[1:], dtype=np.float64)


def select_frames(confidence, num_frames):
    """Split the video into equal segments and take the most confident frame of each."""
    bounds = np.linspace(0, len(confidence), num_frames + 1).astype(int)
    return [int(s + np.argmax(confidence[s:e])) for s, e in zip(bounds[:-1], bounds[1:]) if e > s]


def draw_representative(img, mask, classes, title, out_path, alpha=0.5):
    palette = np.array([c["color"] for c in classes], dtype=np.float64)
    a = (mask > 0)[..., None] * alpha
    overlay = (img * (1 - a) + palette[mask] * a).astype(np.uint8)

    h, w = mask.shape
    fig, axes = plt.subplots(1, 2, figsize=(16, 16 * h / (2 * w) + 1.2))
    axes[0].imshow(img)
    axes[0].set_title("Frame")
    axes[1].imshow(overlay)
    axes[1].set_title("ATLAS prediction")
    for ax in axes:
        ax.axis("off")

    ids, counts = np.unique(mask, return_counts=True)
    handles = []
    for cls_id, count in sorted(zip(ids, counts), key=lambda x: -x[1]):
        frac = count / mask.size
        if cls_id == 0 or frac < PRESENCE_THRESH:
            continue
        color = palette[cls_id] / 255
        name = classes[cls_id]["name"]
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f"{name} ({frac:.0%})"))
        # Name the class at the centre of its largest connected region
        labelled, n = ndimage.label(mask == cls_id)
        if n == 0:
            continue
        largest = 1 + np.argmax(ndimage.sum(np.ones_like(mask), labelled, range(1, n + 1)))
        cy, cx = ndimage.center_of_mass(labelled == largest)
        axes[1].text(cx, cy, name, color="white", fontsize=10, ha="center", va="center",
                     bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2))

    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def make_plots(plot_dir, times, fractions, names, confidence, class_summary, proc):
    plot_dir.mkdir(exist_ok=True)
    paths = {}

    # Coverage over time of the classes with the largest mean area (1 s moving average)
    order = np.argsort(-fractions.mean(0))
    top = [i for i in order if i != 0][:10]
    k = max(1, int(round(1 / max(times[1] - times[0], 1e-6)))) if len(times) > 1 else 1
    smooth = lambda x: np.convolve(x, np.ones(k) / k, mode="same")  # noqa: E731
    fig, ax = plt.subplots(figsize=(14, 5))
    for i in top:
        ax.plot(times, smooth(fractions[:, i]) * 100, label=names[i])
    ax.set(xlabel="time (s)", ylabel="% of frame", title="Class coverage over time (top 10, 1 s moving average)")
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    paths["class_coverage_over_time"] = plot_dir / "class_coverage_over_time.png"
    fig.savefig(paths["class_coverage_over_time"], dpi=110)
    plt.close(fig)

    # Mean coverage and presence per class
    rows = [r for r in class_summary if r["id"] != 0]
    rows.sort(key=lambda r: -r["mean_coverage"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    labels = [r["name"] for r in rows]
    axes[0].barh(labels, [r["mean_coverage"] * 100 for r in rows], color="tab:blue")
    axes[0].set(xlabel="mean % of frame", title="Mean coverage")
    axes[1].barh(labels, [r["presence"] * 100 for r in rows], color="tab:orange")
    axes[1].set(xlabel="% of frames", title=f"Presence (> {PRESENCE_THRESH:.1%} of the frame)")
    axes[0].invert_yaxis()
    fig.tight_layout()
    paths["class_coverage_presence"] = plot_dir / "class_coverage_presence.png"
    fig.savefig(paths["class_coverage_presence"], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, confidence, lw=0.6, alpha=0.5, label="per frame")
    ax.plot(times, smooth(confidence), lw=1.5, label="1 s moving average")
    ax.set(xlabel="time (s)", ylabel="mean per-pixel confidence", title="Prediction confidence")
    ax.legend()
    fig.tight_layout()
    paths["confidence_over_time"] = plot_dir / "confidence_over_time.png"
    fig.savefig(paths["confidence_over_time"], dpi=110)
    plt.close(fig)

    if proc is not None:
        items = list(proc["mean_probabilities"].items())
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([n for n, _ in items], [p for _, p in items], color="tab:green")
        ax.invert_yaxis()
        ax.set(xlabel="mean probability over frames", title="Procedure recognition (closed set of 14 ATLAS procedures)")
        fig.tight_layout()
        paths["procedure_probabilities"] = plot_dir / "procedure_probabilities.png"
        fig.savefig(paths["procedure_probabilities"], dpi=110)
        plt.close(fig)

    return paths


def make_preview_video(src, dst, width):
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = "ffmpeg"
    cmd = [exe, "-v", "error", "-y", "-i", str(src), "-vf", f"scale={width}:-2",
           "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", "-an", str(dst)]
    return subprocess.run(cmd).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--num_frames", type=int, default=12, help="Number of representative frames")
    parser.add_argument("--wandb_project", default="ATLAS-inference")
    parser.add_argument("--wandb_entity", default=None, help="Default: your default W&B entity")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--video_width", type=int, default=960, help="Width of the overlay preview uploaded to W&B")
    args = parser.parse_args()

    classes = json.loads((args.pred_dir / "classes.json").read_text())
    meta = json.loads((args.frames_dir / "meta.json").read_text())
    run_info = json.loads((args.pred_dir / "run_info.json").read_text())
    proc_path = args.pred_dir / "procedure.json"
    proc = json.loads(proc_path.read_text()) if proc_path.exists() else None

    header, data = load_csv(args.pred_dir / "class_stats.csv")
    frame_ids, times, confidence = data[:, 0].astype(int), data[:, 1], data[:, 2]
    fractions = data[:, 3:]
    names = header[3:]

    # ---------------- quantitative ----------------
    present = fractions > PRESENCE_THRESH
    class_summary = []
    for i, name in enumerate(names):
        class_summary.append({
            "id": i,
            "name": name,
            "mean_coverage": float(fractions[:, i].mean()),
            "presence": float(present[:, i].mean()),
            "max_coverage": float(fractions[:, i].max()),
            "seconds_present": float(present[:, i].sum() / meta["fps"]),
        })
    with open(args.pred_dir / "class_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(class_summary[0]))
        w.writeheader()
        w.writerows(class_summary)

    summary = {
        "num_frames": int(len(frame_ids)),
        "duration_s": float(len(frame_ids) / meta["fps"]),
        "mean_confidence": float(confidence.mean()),
        "std_confidence": float(confidence.std()),
        "mean_classes_per_frame": float(present[:, 1:].sum(1).mean()),
        "num_classes_seen": int(present[:, 1:].any(0).sum()),
        "inference_frames_per_s": run_info["frames_per_s"],
    }
    if proc is not None:
        top_name, top_p = next(iter(proc["mean_probabilities"].items()))
        summary["procedure_top1"] = top_name
        summary["procedure_top1_prob"] = top_p
    with open(args.pred_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    plot_paths = make_plots(args.pred_dir / "plots", times, fractions, names, confidence, class_summary, proc)

    # ---------------- qualitative ----------------
    rep_dir = args.pred_dir / "representative_frames"
    rep_dir.mkdir(exist_ok=True)
    selected = select_frames(confidence, args.num_frames)
    rep_items = []
    for row in selected:
        fid = frame_ids[row]
        stem = f"frame_{fid:06d}"
        img = np.array(Image.open(args.frames_dir / f"{stem}.jpg").convert("RGB"))
        mask = np.array(Image.open(args.pred_dir / "masks" / f"{stem}.png"))
        t = meta.get("start", 0) + times[row]
        title = f"{stem}  -  t = {int(t // 60)}:{t % 60:04.1f}  -  confidence {confidence[row]:.2f}"
        out = rep_dir / f"{stem}.png"
        draw_representative(img, mask, classes, title, out)
        rep_items.append((fid, t, img, mask, out))
    print(f"Saved {len(rep_items)} representative frames to {rep_dir}")

    preview = args.pred_dir / "overlay_preview.mp4"
    has_video = (args.pred_dir / "overlay.mp4").exists() and make_preview_video(
        args.pred_dir / "overlay.mp4", preview, args.video_width)

    # ---------------- W&B ----------------
    if args.wandb_mode == "disabled":
        return
    import wandb

    os.environ.setdefault("WANDB_DIR", str(args.pred_dir))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"{Path(meta['video']).stem}_{Path(run_info['ckpt']).stem}",
        job_type="inference",
        mode=args.wandb_mode,
        config={"video": meta, "inference": run_info, "num_representative_frames": len(rep_items)},
    )

    run.define_metric("time_s")
    run.define_metric("frame/*", step_metric="time_s")
    proc_header, proc_data = (None, None)
    if (args.pred_dir / "procedure_per_frame.csv").exists():
        proc_header, proc_data = load_csv(args.pred_dir / "procedure_per_frame.csv")
    top = [i for i in np.argsort(-fractions.mean(0)) if i != 0][:15]
    for row in range(len(frame_ids)):
        log = {"time_s": times[row], "frame/confidence": confidence[row]}
        log |= {f"frame/coverage/{names[i]}": fractions[row, i] for i in top}
        if proc_data is not None:
            log |= {f"frame/procedure/{n}": proc_data[row, j + 1] for j, n in enumerate(proc_header[1:])}
        run.log(log)

    class_labels = {c["id"]: c["name"] for c in classes}
    run.log({
        "class_summary": wandb.Table(dataframe=None, columns=list(class_summary[0]),
                                     data=[list(r.values()) for r in class_summary]),
        "representative_frames": [
            wandb.Image(img, caption=f"frame {fid} (t={t:.1f}s)",
                        masks={"prediction": {"mask_data": mask, "class_labels": class_labels}})
            for fid, t, img, mask, _ in rep_items
        ],
        "representative_figures": [wandb.Image(str(p), caption=p.stem) for *_, p in rep_items],
        **{f"plots/{k}": wandb.Image(str(p)) for k, p in plot_paths.items()},
        **({"overlay_video": wandb.Video(str(preview), format="mp4")} if has_video else {}),
    })
    if proc is not None:
        run.log({"procedure_probabilities": wandb.Table(
            columns=["procedure", "mean_probability"], data=list(proc["mean_probabilities"].items()))})
    run.summary.update(summary)
    run.finish()


if __name__ == "__main__":
    main()
