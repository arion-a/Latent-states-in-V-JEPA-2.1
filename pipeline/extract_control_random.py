"""Control baseline 1: SAME V-JEPA 2.1 ViT-B/16 architecture, RANDOM (untrained) weights.
No checkpoint is loaded. This isolates architecture+pooling+cosine-geometry effects from
anything actually learned during pretraining. If this baseline shows similar A/D/R to the
real trained model, that would mean the H1-H3 signal is mostly a generic property of the
architecture, not evidence of learned physics understanding.
Usage: python extract_control_random.py [--manifest PATH] [--repo PATH] [--out PATH] [--seed 0]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as iio

FRAME_INDICES = list(range(0, 151, 10))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def build_random_encoder(repo_path, device, seed):
    sys.path.insert(0, str(repo_path))
    from app.vjepa_2_1.models import vision_transformer as vit_encoder

    torch.manual_seed(seed)
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
    # NOTE: no checkpoint load -- this is the default random PyTorch initialization.
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)
    return encoder


def load_clip_tensor(path, device):
    frames = np.stack(iio.mimread(path, format="FFMPEG"))
    assert frames.shape[0] == 151
    sel = frames[FRAME_INDICES]
    x = torch.from_numpy(sel).float() / 255.0
    x = x.permute(3, 0, 1, 2).unsqueeze(0)
    x = (x - MEAN) / STD
    return x.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="/workspace/gravity/manifest.json")
    ap.add_argument("--repo", type=str, default="/workspace/gravity/repo")
    ap.add_argument("--out", type=str, default="/workspace/gravity/embeddings_control_random.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, seed: {args.seed}", flush=True)

    encoder = build_random_encoder(Path(args.repo), device, args.seed)
    print("random-weight encoder built (NO checkpoint loaded)", flush=True)

    manifest = json.loads(Path(args.manifest).read_text())
    results = {}
    times = []
    for i, row in enumerate(manifest):
        t0 = time.time()
        x = load_clip_tensor(row["path"], device)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (1, 4608, 768), f"unexpected shape {out.shape}"
        emb = out.mean(dim=1).squeeze(0).float().cpu().numpy()
        assert np.isfinite(emb).all()
        dt = time.time() - t0
        times.append(dt)
        results[row["name"]] = dict(scene=row["scene"], condition=row["condition"],
                                     gravity=row["gravity"], embedding=emb.astype(np.float64).tolist())
        if (i + 1) % 20 == 0 or i == len(manifest) - 1:
            print(f"[{i+1}/{len(manifest)}] {row['name']}: {dt:.2f}s", flush=True)

    Path(args.out).write_text(json.dumps(results))
    print(f"\nDone. {len(results)} embeddings (RANDOM/untrained baseline). "
          f"mean time={np.mean(times):.2f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
