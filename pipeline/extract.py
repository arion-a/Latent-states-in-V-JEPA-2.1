"""Extract final-layer mean-pooled 768-d embeddings via frozen V-JEPA 2.1 ViT-B/16 EMA encoder.
Usage: python extract.py [--manifest PATH] [--repo PATH] [--checkpoint PATH] [--out PATH]
Strict-loads ema_encoder, eval mode, no grad, float32, RGB/255 + ImageNet normalize,
selects source frame indices 0,10,...,150 (16 frames), asserts [1,4608,768], mean-pools."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio

FRAME_INDICES = list(range(0, 151, 10))  # 0,10,...,150 -> 16 frames
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def build_encoder(repo_path, checkpoint_path, device):
    sys.path.insert(0, str(repo_path))
    from app.vjepa_2_1.models import vision_transformer as vit_encoder
    from src.hub.backbones import _clean_backbone_key

    encoder = vit_encoder.vit_base(
        patch_size=16,
        img_size=(384, 384),
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = _clean_backbone_key(ckpt["ema_encoder"])
    missing, unexpected = encoder.load_state_dict(state, strict=True)
    assert not missing and not unexpected, f"missing={missing} unexpected={unexpected}"

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)
    return encoder


def load_clip_tensor(path, device):
    frames = np.stack(iio.mimread(path, format="FFMPEG"))  # [151,H,W,3] uint8
    assert frames.shape[0] == 151, f"expected 151 frames, got {frames.shape[0]} in {path}"
    sel = frames[FRAME_INDICES]  # [16,H,W,3]
    x = torch.from_numpy(sel).float() / 255.0  # [16,H,W,3]
    x = x.permute(3, 0, 1, 2).unsqueeze(0)  # [1,3,16,H,W]
    x = (x - MEAN) / STD
    return x.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="/workspace/gravity/manifest.json")
    ap.add_argument("--repo", type=str, default="/workspace/gravity/repo")
    ap.add_argument("--checkpoint", type=str, default="/workspace/gravity/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt")
    ap.add_argument("--out", type=str, default="/workspace/gravity/embeddings.json")
    ap.add_argument("--scenes", type=str, default=None, help="comma-separated scene IDs to restrict to")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    checkpoint_sha256 = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    print(f"checkpoint sha256: {checkpoint_sha256}", flush=True)

    t_load0 = time.time()
    encoder = build_encoder(Path(args.repo), args.checkpoint, device)
    print(f"encoder strict-loaded and moved to {device} in {time.time()-t_load0:.2f}s", flush=True)

    manifest = json.loads(Path(args.manifest).read_text())
    if args.scenes is not None:
        wanted = {int(x) for x in args.scenes.split(",")}
        manifest = [r for r in manifest if r["scene"] in wanted]

    results = {}
    out_path = Path(args.out)
    if out_path.exists():
        results = json.loads(out_path.read_text())

    times = []
    for i, row in enumerate(manifest):
        if row["name"] in results:
            continue
        t0 = time.time()
        x = load_clip_tensor(row["path"], device)
        with torch.no_grad():
            out = encoder(x)  # [1, N, D]
        assert out.shape == (1, 4608, 768), f"unexpected shape {out.shape} for {row['name']}"
        emb = out.mean(dim=1).squeeze(0).float().cpu().numpy()  # [768]
        assert np.isfinite(emb).all(), f"non-finite embedding for {row['name']}"
        dt = time.time() - t0
        times.append(dt)
        results[row["name"]] = dict(
            scene=row["scene"], condition=row["condition"], gravity=row["gravity"],
            embedding=emb.astype(np.float64).tolist(),
            checkpoint_sha256=checkpoint_sha256, extract_seconds=dt,
        )
        print(f"[{i+1}/{len(manifest)}] {row['name']}: {dt:.2f}s, shape {out.shape}, "
              f"mean={emb.mean():.4f} std={emb.std():.4f}", flush=True)
        out_path.write_text(json.dumps(results))

    if times:
        print(f"\nDone. {len(times)} new embeddings. "
              f"per-clip time: mean={np.mean(times):.2f}s min={np.min(times):.2f}s max={np.max(times):.2f}s", flush=True)
    print(f"Total embeddings in {out_path}: {len(results)}", flush=True)


if __name__ == "__main__":
    main()
