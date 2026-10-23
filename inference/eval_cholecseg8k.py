"""Quantitative evaluation of an ATLAS checkpoint on CholecSeg8k.

CholecSeg8k (Hong et al., 2020) has 13 classes, ATLAS 30. Both are mapped to a
shared label space (EVAL_CLASSES below). CholecSeg8k classes without an ATLAS
counterpart (connective tissue, blood) are ignored, and ATLAS predictions of
classes that do not exist in CholecSeg8k count as errors ("Other ATLAS class").

Images are processed exactly as in infer_video.py (sliding windows, same
resizing as the repo's validation). Reported:
    - per-class IoU / Dice, mIoU (with and without background), pixel accuracy
    - per-video mIoU
    - confusion matrix in the shared label space
    - procedure recognition accuracy (every CholecSeg8k video is a cholecystectomy)
and logged to a W&B project. No qualitative images are logged.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from infer_video import CLASS_NAMES, PROCEDURE_NAMES, REPO_ROOT, build_model, predict, scaled_size  # noqa: E402

# Shared label space: (name, CholecSeg8k watershed-mask values, ATLAS class ids)
EVAL_CLASSES = [
    ("Background", [50], [0]),
    ("Abdominal wall", [11], [7]),
    ("Liver", [21], [10]),
    ("Gastrointestinal tract", [13], [5, 6, 15]),
    ("Fat", [12], [9]),
    ("Instruments", [31, 32], [1]),  # grasper, L-hook electrocautery
    ("Cystic duct", [25], [11]),
    ("Gallbladder", [22], [12]),
    ("Hepatic vein", [33], [2]),
    ("Liver ligament", [5], [13]),
]
IGNORED_GT = {23: "Connective tissue", 24: "Blood", 0: "Unlabelled", 255: "Unlabelled"}
IGNORE = 255
OTHER = len(EVAL_CLASSES)  # prediction of an ATLAS class absent in CholecSeg8k
EVAL_NAMES = [c[0] for c in EVAL_CLASSES]


def gt_lut():
    lut = np.full(256, IGNORE, dtype=np.uint8)
    for i, (_, values, _) in enumerate(EVAL_CLASSES):
        lut[values] = i
    return lut


def pred_lut(num_atlas_classes):
    lut = torch.full((num_atlas_classes,), OTHER, dtype=torch.long)
    for i, (_, _, atlas_ids) in enumerate(EVAL_CLASSES):
        lut[atlas_ids] = i
    return lut


class CholecSeg8k(Dataset):
    """videoXX/videoXX_NNNNN/frame_N_endo.png + frame_N_endo_watershed_mask.png"""

    def __init__(self, root, img_size, videos=None):
        self.img_size = img_size
        self.items = []
        for img in sorted(Path(root).rglob("*_endo.png"), key=self._sort_key):
            mask = img.with_name(img.stem + "_watershed_mask.png")
            video = img.parent.parent.name
            if mask.exists() and (videos is None or video in videos):
                self.items.append((img, mask, video, img.parent.name))
        self.lut = gt_lut()
        unknown = set()
        for _, m, _, _ in self.items[:: max(1, len(self.items) // 50)]:
            mask = np.array(Image.open(m))
            vals = np.unique(mask[..., 0] if mask.ndim == 3 else mask)
            unknown |= {int(v) for v in vals if self.lut[v] == IGNORE and v not in IGNORED_GT}
        if unknown:
            print(f"WARNING: unmapped mask values {sorted(unknown)} will be ignored")

    @staticmethod
    def _sort_key(p):
        return p.parent.parent.name, p.parent.name, int(p.stem.split("_")[1])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, _, _ = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        new_h, new_w = scaled_size((img.height, img.width), self.img_size)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        gt = self.lut[mask]
        return torch.from_numpy(np.array(resized)).permute(2, 0, 1), torch.from_numpy(gt), idx


def metrics_from_confusion(conf):
    """conf: (num_gt, num_pred) pixel counts; IoU/Dice for each GT class."""
    n = conf.shape[0]
    tp = np.diag(conf[:, :n])
    fp = conf[:, :n].sum(0) - tp
    fn = conf.sum(1) - tp
    present = conf.sum(1) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(present, tp / (tp + fp + fn), np.nan)
        dice = np.where(present, 2 * tp / (2 * tp + fp + fn), np.nan)
    return iou, dice, tp.sum() / max(conf.sum(), 1)


def plot_confusion(conf, path):
    norm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(conf.shape[1]), EVAL_NAMES + ["Other ATLAS class"], rotation=45, ha="right")
    ax.set_yticks(range(conf.shape[0]), EVAL_NAMES)
    ax.set(xlabel="ATLAS prediction", ylabel="CholecSeg8k ground truth",
           title="Confusion matrix (row-normalised, pixels)")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=Path("/work/tesi_eprato01/datasets/cholecseg8k"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=Path("/work/tesi_eprato01/checkpoints/ATLAS/vit-s/atlas_vits.ckpt"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/ATLAS_DINOv3_vits.yaml")
    parser.add_argument("--clip_length", type=int, default=1,
                        help="1 = per-frame; >1 = query propagation over consecutive frames of each 80-frame clip")
    parser.add_argument("--videos", nargs="*", default=None, help="Subset of videos, e.g. video01 video12")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--wandb_project", default="ATLAS-CholecSeg8k")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    network, img_size = build_model(args.config, args.ckpt, device)
    num_atlas = network.class_head.out_features - 1
    plut = pred_lut(num_atlas).to(device)

    dataset = CholecSeg8k(args.data_root, img_size, set(args.videos) if args.videos else None)
    if len(dataset) == 0:
        raise SystemExit(f"No CholecSeg8k frames found in {args.data_root}")
    videos = sorted({v for _, _, v, _ in dataset.items})
    print(f"{len(dataset)} frames from {len(videos)} videos")

    temporal = args.clip_length > 1
    loader = DataLoader(dataset, batch_size=1 if temporal else args.batch_size,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    amp = device.type == "cuda" and not args.no_amp

    n_gt, n_pred = len(EVAL_CLASSES), len(EVAL_CLASSES) + 1
    conf_total = torch.zeros(n_gt, n_pred, dtype=torch.long, device=device)
    conf_video = defaultdict(lambda: torch.zeros(n_gt, n_pred, dtype=torch.long, device=device))
    proc_correct = defaultdict(list)
    proc_probs = defaultdict(list)
    chole = PROCEDURE_NAMES.index("cholecystectomy")
    prev_query_embed, prev_clip, t_in_clip = None, None, 0

    t_start = time.time()
    for imgs, gts, idxs in tqdm(loader, desc="CholecSeg8k", mininterval=30):
        imgs, gts = imgs.to(device, non_blocking=True), gts.to(device, non_blocking=True)
        if temporal:
            clip = dataset.items[int(idxs[0])][3]
            t_in_clip = 0 if clip != prev_clip else t_in_clip + 1
            if t_in_clip % args.clip_length == 0:
                prev_query_embed = None
            prev_clip = clip
            time_indices = torch.full((1,), t_in_clip % args.clip_length, device=device, dtype=torch.long)
        else:
            time_indices = torch.zeros(imgs.shape[0], device=device, dtype=torch.long)

        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            scores, proc, query_embed = predict(network, imgs, img_size, time_indices, prev_query_embed)
        if temporal:
            prev_query_embed = query_embed

        preds = plut[F.interpolate(scores, gts.shape[-2:], mode="bilinear").argmax(1)]
        for b, idx in enumerate(idxs.tolist()):
            video = dataset.items[idx][2]
            valid = gts[b] != IGNORE
            cm = torch.bincount(gts[b][valid].long() * n_pred + preds[b][valid],
                                minlength=n_gt * n_pred).view(n_gt, n_pred)
            conf_total += cm
            conf_video[video] += cm
            if proc is not None:
                p = proc[b].softmax(-1)
                proc_probs[video].append(p.cpu())
                proc_correct[video].append(int(p.argmax()) == chole)
    elapsed = time.time() - t_start

    # ---------------- metrics ----------------
    conf = conf_total.cpu().numpy()
    iou, dice, pix_acc = metrics_from_confusion(conf)
    per_class = [{"class": n, "iou": float(iou[i]), "dice": float(dice[i]),
                  "gt_pixel_share": float(conf[i].sum() / conf.sum())} for i, n in enumerate(EVAL_NAMES)]
    per_video = []
    for v in videos:
        v_iou, _, v_acc = metrics_from_confusion(conf_video[v].cpu().numpy())
        mean_proc = torch.stack(proc_probs[v]).mean(0) if proc_probs[v] else None
        per_video.append({
            "video": v,
            "frames": len(proc_correct[v]),
            "miou": float(np.nanmean(v_iou)),
            "miou_no_bg": float(np.nanmean(v_iou[1:])),
            "pixel_acc": float(v_acc),
            "procedure_frame_acc": float(np.mean(proc_correct[v])) if proc_correct[v] else float("nan"),
            "procedure_pred": PROCEDURE_NAMES[int(mean_proc.argmax())] if mean_proc is not None else "",
        })

    all_correct = [c for v in videos for c in proc_correct[v]]
    summary = {
        "miou": float(np.nanmean(iou)),
        "miou_no_bg": float(np.nanmean(iou[1:])),
        "mdice": float(np.nanmean(dice)),
        "mdice_no_bg": float(np.nanmean(dice[1:])),
        "pixel_acc": float(pix_acc),
        "other_atlas_class_rate": float(conf[:, OTHER].sum() / conf.sum()),
        "procedure_frame_acc": float(np.mean(all_correct)) if all_correct else float("nan"),
        "procedure_video_acc": float(np.mean([r["procedure_pred"] == "cholecystectomy" for r in per_video])),
        "num_frames": len(dataset),
        "num_videos": len(videos),
        "frames_per_s": len(dataset) / elapsed,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    for name, rows in [("per_class.csv", per_class), ("per_video.csv", per_video)]:
        with open(args.out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savetxt(args.out_dir / "confusion_matrix.csv", conf, fmt="%d", delimiter=",",
               header=",".join(EVAL_NAMES + ["Other ATLAS class"]))
    plot_confusion(conf, args.out_dir / "confusion_matrix.png")

    print(json.dumps(summary, indent=2))
    print("\nPer-class IoU:")
    for r in per_class:
        print(f"  {r['class']:24s} IoU {r['iou']:.3f}  Dice {r['dice']:.3f}  (GT share {r['gt_pixel_share']:.1%})")

    # ---------------- W&B ----------------
    if args.wandb_mode == "disabled":
        return
    import wandb

    os.environ.setdefault("WANDB_DIR", str(args.out_dir))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"{Path(args.ckpt).stem}_clip{args.clip_length}",
        job_type="evaluation",
        mode=args.wandb_mode,
        config={
            "ckpt": str(args.ckpt), "config": str(args.config), "clip_length": args.clip_length,
            "amp": amp, "videos": videos, "eval_classes": {n: {"cholecseg8k": g, "atlas": [CLASS_NAMES[a] for a in at]}
                                                          for n, g, at in EVAL_CLASSES},
            "ignored_gt": list(IGNORED_GT.values()),
        },
    )
    class_table = wandb.Table(columns=list(per_class[0]), data=[list(r.values()) for r in per_class])
    run.log({
        "per_class": class_table,
        "per_class_iou": wandb.plot.bar(class_table, "class", "iou", title="IoU per class"),
        "per_video": wandb.Table(columns=list(per_video[0]), data=[list(r.values()) for r in per_video]),
        "confusion_matrix": wandb.Image(str(args.out_dir / "confusion_matrix.png")),
        "confusion_matrix_pixels": wandb.Table(
            columns=["ground_truth"] + EVAL_NAMES + ["Other ATLAS class"],
            data=[[EVAL_NAMES[i]] + conf[i].tolist() for i in range(n_gt)]),
    })
    run.summary.update(summary)
    run.summary.update({f"iou/{r['class']}": r["iou"] for r in per_class})
    run.finish()


if __name__ == "__main__":
    main()
