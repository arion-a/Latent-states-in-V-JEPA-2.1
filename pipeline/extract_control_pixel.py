"""Control baseline 2: trivial non-learned pixel-statistics feature. No model, no training,
no architecture at all -- just per-frame mean+std RGB over the same 16 sampled frames,
flattened into one vector per clip. This is the floor: if even this dumb feature shows
similar A/D/R to the real V-JEPA2 embeddings, the H1-H3 signal says nothing about learned
representations at all -- it would just reflect that gravity changes pixel statistics in
a consistent way, which is true by construction (the ball moves to a different place).
Usage: python extract_control_pixel.py [--manifest PATH] [--out PATH]"""
import argparse
import json
from pathlib import Path

import numpy as np
import imageio.v2 as iio

FRAME_INDICES = list(range(0, 151, 10))


def features_for_clip(path):
    frames = np.stack(iio.mimread(path, format="FFMPEG"))  # [151,H,W,3]
    assert frames.shape[0] == 151
    sel = frames[FRAME_INDICES].astype(np.float64) / 255.0  # [16,H,W,3]
    per_frame_mean = sel.mean(axis=(1, 2))  # [16,3]
    per_frame_std = sel.std(axis=(1, 2))    # [16,3]
    feat = np.concatenate([per_frame_mean.flatten(), per_frame_std.flatten()])  # [96]
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="/workspace/gravity/manifest.json")
    ap.add_argument("--out", type=str, default="/workspace/gravity/embeddings_control_pixel.json")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    results = {}
    for i, row in enumerate(manifest):
        feat = features_for_clip(row["path"])
        assert np.isfinite(feat).all()
        results[row["name"]] = dict(scene=row["scene"], condition=row["condition"],
                                     gravity=row["gravity"], embedding=feat.tolist())
        if (i + 1) % 20 == 0 or i == len(manifest) - 1:
            print(f"[{i+1}/{len(manifest)}] {row['name']}: feature dim {feat.shape[0]}", flush=True)

    Path(args.out).write_text(json.dumps(results))
    print(f"\nDone. {len(results)} embeddings (PIXEL-STATISTICS baseline, dim=96) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
