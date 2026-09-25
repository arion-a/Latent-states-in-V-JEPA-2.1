"""Render clips per scene_plan.json (design_revision ballistic-launch-v3).
Usage: python render_all.py [--scenes 0,1,2] [--out DIR] [--plan PATH]
Renders baseline (g=9.8, orange) + low-gravity (g=4.9, orange) for every requested
scene, plus a color clip (g=9.8, blue) for scenes in color_control_scene_ids.
Lossless FFV1-in-MKV, verified by decode round-trip. Writes manifest.json."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as iio

WORLD = (-100, 100, 0, 200)
RES = 384
SCALE = RES / (WORLD[1] - WORLD[0])
RADIUS_M = 5.0
BG = (32, 32, 32)
GRID = (44, 44, 44)
ORANGE = (235, 100, 60)
BLUE = (60, 140, 235)
N_FRAMES = 151
FPS = 30


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def project(x, y):
    return SCALE * (x - WORLD[0]), SCALE * (WORLD[3] - y)


def render_frames(x0, y0, vx, vy0, g, color):
    bg = Image.new("RGB", (RES, RES), BG)
    draw = ImageDraw.Draw(bg)
    for p in range(0, RES, 38):
        draw.line((p, 0, p, RES - 1), fill=GRID)
        draw.line((0, p, RES - 1, p), fill=GRID)

    frames = []
    for k in range(N_FRAMES):
        t = k / FPS
        x = x0 + vx * t
        y = y0 + vy0 * t - 0.5 * g * t * t
        assert WORLD[0] + RADIUS_M < x < WORLD[1] - RADIUS_M, f"x out of bounds: {x}"
        assert WORLD[2] + RADIUS_M < y < WORLD[3] - RADIUS_M, f"y out of bounds: {y}"
        u, v = project(x, y)
        r_px = RADIUS_M * SCALE
        im = bg.copy()
        ImageDraw.Draw(im).ellipse((u - r_px, v - r_px, u + r_px, v + r_px), fill=color)
        frames.append(np.asarray(im))
    return np.stack(frames)


def write_clip(frames, path):
    iio.mimwrite(path, frames, format="FFMPEG", fps=FPS, codec="ffv1", pixelformat="bgr0", macro_block_size=1)
    decoded = np.stack(iio.mimread(path, format="FFMPEG"))
    if not np.array_equal(decoded, frames):
        raise RuntimeError(f"Lossless round-trip FAILED for {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=str, default=None, help="comma-separated scene IDs; default = all")
    ap.add_argument("--out", type=str, default="/workspace/gravity/videos")
    ap.add_argument("--plan", type=str, default="/workspace/gravity/scene_plan.json")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    scenes = {s["id"]: s for s in plan["scenes"]}
    color_ids = set(plan["color_control_scene_ids"])

    wanted = sorted(scenes.keys()) if args.scenes is None else [int(x) for x in args.scenes.split(",")]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in wanted:
        s = scenes[sid]
        x0, y0, vx, vy0 = s["x0_m"], s["y0_m"], s["vx_m_per_s"], s["vy0_m_per_s"]
        conditions = [("g98", 9.8, ORANGE), ("g49", 4.9, ORANGE)]
        if sid in color_ids:
            conditions.append(("color", 9.8, BLUE))

        for cond, g, color in conditions:
            name = f"scene_{sid:02d}_{cond}"
            path = out_dir / f"{name}.mkv"
            frames = render_frames(x0, y0, vx, vy0, g, color)
            if not path.exists():
                write_clip(frames, path)
            else:
                decoded = np.stack(iio.mimread(path, format="FFMPEG"))
                if not np.array_equal(decoded, frames):
                    raise RuntimeError(f"Existing file {path} does not match expected render, refusing to reuse")
            rows.append(dict(
                name=name, scene=sid, condition=cond, gravity=g,
                x0_m=x0, y0_m=y0, vx_m_per_s=vx, vy0_m_per_s=vy0,
                path=str(path), sha256=sha256_of(path),
                n_frames=N_FRAMES, fps=FPS,
            ))
            print(f"rendered {name} ({len(rows)}/{len(wanted) * (3 if sid in color_ids else 2)} this batch)", flush=True)

    manifest_path = out_dir.parent / "manifest.json"
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        existing = [r for r in existing if r["name"] not in {r2["name"] for r2 in rows}]
    manifest_path.write_text(json.dumps(existing + rows, indent=2))
    print(f"Wrote {len(rows)} clips this run; manifest now has {len(existing) + len(rows)} entries -> {manifest_path}")


if __name__ == "__main__":
    main()
