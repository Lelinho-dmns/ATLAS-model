"""Run a finetuned ATLAS checkpoint on frames extracted by preprocess_video.py.

The preprocessing mirrors the validation path of the training code
(`MaskClassificationSemantic.eval_step`): each frame is resized (PIL bilinear) so
that its short side matches the model input (256 px), split into overlapping
256x256 windows along the long side, and the per-pixel class scores of the
windows are averaged and resized back to the frame resolution.

The DINOv3 backbone architecture is rebuilt locally from the checkpoint shapes,
so no access to the gated facebook/dinov3 HuggingFace repo is needed: all
weights come from the ATLAS checkpoint.

Outputs (in --out_dir):
    masks/frame_XXXXXX.png   uint8 class ids (0..29, see classes.json)
    overlay.mp4              colour overlay video (unless --no_video)
    legend.png               colour legend of the classes
    classes.json             id -> name and colour
    class_stats.csv          per-frame mean confidence and fraction of pixels of each class
    procedure.json           procedure recognition (mean over frames + per-frame argmax)
    procedure_per_frame.csv  per-frame procedure probabilities
    run_info.json            settings and timing
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import models.vit  # noqa: E402
from models.atlas import ATLAS  # noqa: E402
from models.vit import ViT  # noqa: E402

# Copied from datasets/atlas_semantic.py (importing it would pull in the whole data module)
CLASS_NAMES = [
    "Background", "Tools/camera", "Vein", "Artery", "Nerve", "Small intestine",
    "Colon/rectum", "Abdominal wall", "Diaphragm", "Fat", "Liver",
    "Bile/lymph Duct", "Gallbladder", "Hepatic ligament", "Cystic plate",
    "Stomach", "Spleen", "Uterus", "Ovary", "Oviduct", "Prostate", "Urethra",
    "Ligated plexus", "Seminal vesicles", "Non anatomical", "Bladder", "Lung",
    "Airway (bronchus/trachea)", "Esophagus", "Pericardium",
]
PROCEDURE_NAMES = [
    "adrenalectomy", "appendectomy", "cholecystectomy", "colectomy",
    "esophagectomy", "gastric_surgery", "gastrojejunostomy", "hemicolectomy",
    "lar", "liver_resection", "rarp", "rectopexy", "sigmoidcolectomy",
    "splenectomy",
]


def make_palette(num_classes):
    """Distinct colours per class; background is black (not drawn in the overlay)."""
    import matplotlib.pyplot as plt

    cmap = [plt.get_cmap("tab20")(i) for i in range(20)] + [plt.get_cmap("tab20b")(i) for i in range(20)]
    palette = np.array([c[:3] for c in cmap[: num_classes - 1]]) * 255
    return np.concatenate([np.zeros((1, 3)), palette]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def dinov3_config_from_state_dict(sd, prefix="network.encoder.backbone."):
    from transformers import DINOv3ViTConfig

    hidden = sd[prefix + "norm.weight"].shape[0]
    num_layers = 1 + max(
        int(k[len(prefix + "blocks."):].split(".")[0]) for k in sd if k.startswith(prefix + "blocks.")
    )
    gated = any(".mlp.gate_proj." in k for k in sd)
    return DINOv3ViTConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden // 64,  # DINOv3 uses 64-dim heads for all sizes
        intermediate_size=sd[prefix + "blocks.0.mlp.up_proj.weight"].shape[0],
        num_register_tokens=sd[prefix + "patch_embed.register_tokens"].shape[1],
        patch_size=sd[prefix + "patch_embed.patch_embeddings.weight"].shape[-1],
        use_gated_mlp=gated,
        key_bias=(prefix + "blocks.0.attention.k_proj.bias") in sd,
    )


class _OfflineAutoModel:
    """Stands in for transformers.AutoModel inside models/vit.py: builds the
    architecture from a config instead of downloading pretrained weights."""

    def __init__(self, config):
        self.config = config

    def from_pretrained(self, *args, **kwargs):
        from transformers import DINOv3ViTModel

        return DINOv3ViTModel(self.config)


def build_model(config_path, ckpt_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    net_args = dict(cfg["model"]["init_args"]["network"]["init_args"])
    enc_args = dict(net_args.pop("encoder")["init_args"])
    enc_args["ckpt_path"] = None
    img_size = tuple(cfg["data"]["init_args"]["img_size"])
    enc_args["img_size"] = img_size
    net_args["num_classes"] = cfg["data"]["init_args"]["num_classes"]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("._orig_mod", ""): v for k, v in sd.items()}

    models.vit.AutoModel = _OfflineAutoModel(dinov3_config_from_state_dict(sd))
    network = ATLAS(encoder=ViT(**enc_args), **net_args)

    net_sd = {k[len("network."):]: v for k, v in sd.items() if k.startswith("network.")}
    missing, unexpected = network.load_state_dict(net_sd, strict=False)
    # rope periods are recomputed from the config, everything else must match
    missing = [k for k in missing if "rope_embeddings" not in k]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match config.\nMissing: {missing}\nUnexpected: {unexpected}")
    print(f"Loaded {len(net_sd)} tensors from {ckpt_path}")

    return network.to(device).eval(), img_size


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def scaled_size(size, img_size):
    """Same as LightningModule.scale_img_size_semantic."""
    factor = max(img_size[0] / size[0], img_size[1] / size[1])
    return [round(s * factor) for s in size]


class FrameDataset(Dataset):
    def __init__(self, frame_paths, img_size):
        self.frame_paths = frame_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        new_h, new_w = scaled_size((img.height, img.width), self.img_size)
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        return torch.from_numpy(np.array(resized)).permute(2, 0, 1), idx


def window_starts(long_side, win):
    """Same windowing as LightningModule.window_imgs_semantic."""
    num_crops = math.ceil(long_side / win)
    overlap = num_crops * win - long_side
    overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0
    return [int(j * (win - overlap_per_crop)) for j in range(num_crops)]


def to_crops(imgs, win):
    """(B, 3, H, W) resized frames -> (B * n, 3, win, win) crops along the long side."""
    h, w = imgs.shape[-2:]
    vertical = h > w
    starts = window_starts(max(h, w), win)
    crops = [imgs[:, :, s:s + win, :] if vertical else imgs[:, :, :, s:s + win] for s in starts]
    return torch.stack(crops, dim=1).flatten(0, 1), starts, vertical


def merge_crops(crop_logits, batch_size, starts, vertical, resized_hw, win):
    """Average overlapping crop scores back onto the resized frame."""
    n = len(starts)
    crop_logits = crop_logits.view(batch_size, n, *crop_logits.shape[1:])
    sums = crop_logits.new_zeros(batch_size, crop_logits.shape[2], *resized_hw)
    counts = crop_logits.new_zeros(1, 1, *resized_hw)
    for j, s in enumerate(starts):
        if vertical:
            sums[:, :, s:s + win, :] += crop_logits[:, j]
            counts[:, :, s:s + win, :] += 1
        else:
            sums[:, :, :, s:s + win] += crop_logits[:, j]
            counts[:, :, :, s:s + win] += 1
    return sums / counts


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(network, imgs, img_size, time_indices, prev_query_embed):
    """One forward pass on a batch of resized uint8 frames.

    Returns per-pixel class scores at the resized resolution, procedure logits
    per frame and the query embeddings (for temporal query propagation).
    """
    win = min(img_size)
    crops, starts, vertical = to_crops(imgs, win)
    n = len(starts)
    crop_time = time_indices.repeat_interleave(n)
    crop_prev = prev_query_embed.repeat_interleave(n, dim=0) if prev_query_embed is not None else None

    mask_logits_l, class_logits_l, proc_logits_l, query_embed = network(
        crops.float() / 255.0,
        time_indices=crop_time,
        prev_query_embed=crop_prev,
        return_query_embedding=True,
    )
    mask_logits = F.interpolate(mask_logits_l[-1].float(), img_size, mode="bilinear")
    crop_scores = torch.einsum(
        "bqhw, bqc -> bchw",
        mask_logits.sigmoid(),
        class_logits_l[-1].float().softmax(dim=-1)[..., :-1],
    )
    scores = merge_crops(crop_scores, imgs.shape[0], starts, vertical, imgs.shape[-2:], win)

    bsz = imgs.shape[0]
    proc = proc_logits_l[-1].float().view(bsz, n, -1).mean(1) if proc_logits_l else None
    query_embed = query_embed.view(bsz, n, *query_embed.shape[1:]).mean(1)
    return scores, proc, query_embed


def ffmpeg_exe():
    """Prefer the imageio-ffmpeg binary (has libx264) over the system ffmpeg."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class VideoWriter:
    def __init__(self, path, width, height, fps, crf):
        exe = ffmpeg_exe()
        encoders = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
        if "libx264" in encoders:
            codec = ["-c:v", "libx264", "-crf", str(crf)]
        else:  # e.g. the cluster ffmpeg module is built without x264
            codec = ["-c:v", "mpeg4", "-q:v", "3"]
        cmd = [
            exe, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            *codec, "-pix_fmt", "yuv420p", str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def save_legend(path, palette):
    row_h, w = 26, 330
    legend = Image.new("RGB", (w, row_h * len(CLASS_NAMES)), "white")
    draw = ImageDraw.Draw(legend)
    for i, name in enumerate(CLASS_NAMES):
        draw.rectangle([6, i * row_h + 4, 30, i * row_h + row_h - 4], fill=tuple(int(c) for c in palette[i]))
        draw.text((40, i * row_h + 7), f"{i:2d} {name}", fill="black")
    legend.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames_dir", type=Path, required=True, help="Output of preprocess_video.py")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=Path("/work/tesi_eprato01/checkpoints/ATLAS/vit-s/atlas_vits.ckpt"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/ATLAS_DINOv3_vits.yaml",
                        help="Must match the checkpoint variant (vits/vitb/vitl)")
    parser.add_argument("--clip_length", type=int, default=1,
                        help="1 = every frame independently (as in the repo's validation). "
                             ">1 = propagate queries across consecutive frames in clips of this "
                             "length, as in training (clip_length 5 in the configs).")
    parser.add_argument("--batch_size", type=int, default=16, help="Frames per forward pass (clip_length 1 only)")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--no_amp", action="store_true", help="Disable fp16 autocast")
    parser.add_argument("--no_video", action="store_true", help="Skip the overlay video")
    parser.add_argument("--no_masks", action="store_true", help="Skip saving PNG masks")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity")
    parser.add_argument("--crf", type=int, default=20, help="x264 quality of the overlay video")
    args = parser.parse_args()

    frame_paths = sorted(args.frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit(f"No frame_*.jpg in {args.frames_dir}; run preprocess_video.py first")
    meta_path = args.frames_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fps = meta.get("fps", 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    torch.backends.cuda.matmul.allow_tf32 = True
    network, img_size = build_model(args.config, args.ckpt, device)
    num_classes = network.class_head.out_features - 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.out_dir / "masks"
    if not args.no_masks:
        mask_dir.mkdir(exist_ok=True)
    palette = make_palette(num_classes)
    save_legend(args.out_dir / "legend.png", palette)
    with open(args.out_dir / "classes.json", "w") as f:
        json.dump([{"id": i, "name": n, "color": palette[i].tolist()} for i, n in enumerate(CLASS_NAMES)], f, indent=2)

    first = Image.open(frame_paths[0])
    orig_w, orig_h = first.size
    writer = None if args.no_video else VideoWriter(args.out_dir / "overlay.mp4", orig_w, orig_h, fps, args.crf)

    temporal = args.clip_length > 1
    loader = DataLoader(
        FrameDataset(frame_paths, img_size),
        batch_size=1 if temporal else args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    palette_t = torch.from_numpy(palette).to(device)
    amp = device.type == "cuda" and not args.no_amp

    stats_f = open(args.out_dir / "class_stats.csv", "w", newline="")
    stats = csv.writer(stats_f)
    stats.writerow(["frame", "time_s", "confidence"] + CLASS_NAMES)
    proc_probs = []
    prev_query_embed = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.time()

    for imgs, idxs in tqdm(loader, desc="Inference", unit="batch", mininterval=30):
        imgs = imgs.to(device, non_blocking=True)
        if temporal:
            t = int(idxs[0]) % args.clip_length
            if t == 0:
                prev_query_embed = None
            time_indices = torch.full((imgs.shape[0],), t, device=device, dtype=torch.long)
        else:
            time_indices = torch.zeros(imgs.shape[0], device=device, dtype=torch.long)

        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            scores, proc, query_embed = predict(network, imgs, img_size, time_indices, prev_query_embed)
        if temporal:
            prev_query_embed = query_embed
        if proc is not None:
            proc_probs.append(proc.softmax(-1).cpu())

        for b, idx in enumerate(idxs.tolist()):
            # Back to the frame resolution, as revert_window_logits_semantic does
            full = F.interpolate(scores[b:b + 1], (orig_h, orig_w), mode="bilinear")[0]
            # Scores are summed over the queries, so normalise them per pixel to get a [0, 1] confidence
            conf, pred = (full / full.sum(0, keepdim=True).clamp_min(1e-6)).max(0)

            counts = torch.bincount(pred.flatten(), minlength=num_classes).float() / pred.numel()
            stats.writerow([idx, f"{idx / fps:.3f}", f"{conf.mean().item():.4f}"]
                           + [f"{c:.5f}" for c in counts.tolist()])

            pred_np = pred.to(torch.uint8).cpu().numpy()
            if not args.no_masks:
                Image.fromarray(pred_np).save(mask_dir / f"{frame_paths[idx].stem}.png")
            if writer is not None:
                frame = torch.from_numpy(np.array(Image.open(frame_paths[idx]).convert("RGB"))).to(device)
                color = palette_t[pred].float()
                a = (pred > 0).float()[..., None] * args.alpha
                blended = (frame.float() * (1 - a) + color * a).round().to(torch.uint8)
                writer.write(blended.cpu().numpy())

    stats_f.close()
    if writer is not None:
        writer.close()
    elapsed = time.time() - t_start

    with open(args.out_dir / "run_info.json", "w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "config": str(args.config),
            "clip_length": args.clip_length,
            "amp": amp,
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "num_frames": len(frame_paths),
            "frame_size": [orig_w, orig_h],
            "fps": fps,
            "elapsed_s": elapsed,
            "frames_per_s": len(frame_paths) / elapsed,
        }, f, indent=2)
    print(f"{len(frame_paths)} frames in {elapsed:.1f}s ({len(frame_paths) / elapsed:.1f} frames/s, incl. I/O)")

    if proc_probs:
        per_frame = torch.cat(proc_probs)
        mean = per_frame.mean(0)
        ranking = sorted(zip(PROCEDURE_NAMES, mean.tolist()), key=lambda x: -x[1])
        with open(args.out_dir / "procedure.json", "w") as f:
            json.dump({
                "mean_probabilities": dict(ranking),
                "per_frame_argmax": [PROCEDURE_NAMES[i] for i in per_frame.argmax(1).tolist()],
            }, f, indent=2)
        with open(args.out_dir / "procedure_per_frame.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame"] + PROCEDURE_NAMES)
            for i, p in enumerate(per_frame.tolist()):
                w.writerow([i] + [f"{x:.4f}" for x in p])
        print("Procedure recognition (mean over frames):")
        for name, p in ranking[:3]:
            print(f"  {name:20s} {p:.3f}")

    print(f"Done: {len(frame_paths)} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
