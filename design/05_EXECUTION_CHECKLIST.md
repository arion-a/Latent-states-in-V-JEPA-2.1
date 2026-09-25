# 5. Execution checklist & guardrails

**Status: pre-flight only. Nothing has been installed, downloaded, rendered, or run.**

Documents 01–04 define *what* the experiment is. This document defines *how not to run it off the rails* — a literal pass/fail gate before each step is allowed to start the next one, tuned to the actual machine it will run on. If a gate fails, stop and fix that gate. Do not continue "for now" and fix it later — that is exactly how a small feasibility experiment quietly turns into an unverifiable one.

---

## 0. This machine, checked 2026-09-25

| Check | Result | Verdict |
|---|---|---|
| OS | Windows 10 Home, build 19045 | OK |
| CPU | Intel Core i5-7200U @ 2.50GHz — 2 cores / 4 threads | ⚠ weak, mobile-class, 2017-era, no AVX-512 |
| RAM | ~7.9 GB total | ⚠ tight for a ViT-B forward pass plus Python/torch overhead |
| GPU | none detected (`nvidia-smi` not found) | ⚠ CPU-only inference |
| `C:` free space | **12 MB free of 119 GB** | 🛑 blocking — effectively full |
| `D:` free space | 593 GB free of 918 GB | OK — everything must live here |
| `E:` free space | 1.4 GB free of 14 GB | ⚠ avoid |
| Python | 3.13.7 (`python`) available | OK |
| `torch` installed | no | expected, not yet installed |

**Two blockers, before anything else:**

1. **`C:` is full.** pip's default cache, a default virtualenv, `%TEMP%`, and `git clone` all default to `C:` unless redirected. At 12 MB free, any of those fail immediately. Every install/download in Step&nbsp;1 below must be explicitly pointed at `D:`.
2. **No GPU, weak CPU.** The design already *requires* CPU float32 (Document 1: "do not silently substitute reduced precision to accommodate an environment") — so this isn't a protocol violation. But it means per-clip inference time is currently unknown. Step&nbsp;1 turns that unknown into a measured number on one real clip before 100 clips are committed to, so a bad time estimate is discovered in minutes, not hours.

These are read-only findings from this session — no files were written or downloaded to produce this table.

---

## 1. Storage & dependency plan

All new files for this experiment — Python virtual environment, cloned `vjepa2` source, downloaded checkpoint, rendered videos, embeddings, results — go under:

```
D:\gravity-latent-experiment-runtime\
```

Nothing is written to `C:` at any point. Concretely, before any install:

- Create the virtualenv **on `D:`**, e.g. `python -m venv D:\gravity-latent-experiment-runtime\venv` — not the default `C:\Users\...` location.
- Redirect pip's cache: `pip config set global.cache-dir D:\gravity-latent-experiment-runtime\pip-cache` (or `PIP_CACHE_DIR` env var).
- Redirect temp files for the session: `TEMP`/`TMP` env vars pointed at `D:\gravity-latent-experiment-runtime\tmp` while installing/downloading.
- Clone `vjepa2` and download the checkpoint directly under `D:\gravity-latent-experiment-runtime\` (matching the earlier partial download location noted in the review package).

**Gate:** after setup, `C:` free space is unchanged (still ~12 MB) and every new file is verifiably under `D:`. If any install touched `C:`, stop and fix the redirect before continuing.

Dependencies (per `README.md`): `torch`, `torchvision`, `timm`, `einops`, `numpy`, `pillow`, `matplotlib`, `imageio`, `imageio-ffmpeg`, plus the `vjepa2` repo at commit `204698b45b3712590f06245fbfba32d3be539812` and checkpoint `vjepa2_1_vitb_dist_vitG_384.pt`.

---

## 2. Step-by-step plan, with the evaluation gate for each

This mirrors `03_WORKFLOW.md`'s six steps, but each now ends in a concrete, checkable gate instead of a prose exit condition.

### Step 1 — Environment verification
**Do:** install dependencies into the `D:`-based venv; clone the pinned `vjepa2` commit; download the checkpoint; record its SHA-256; load it strictly into the EMA encoder; run **one** real clip end-to-end (render → extract → single embedding) purely to measure wall-clock time and peak RAM.
**Gate — all must pass before Step 2:**
- [ ] `pip list` inside the venv shows every dependency, installed under `D:`.
- [ ] Checkpoint SHA-256 recorded; strict load succeeds with zero missing/unexpected keys.
- [ ] One clip produces an embedding of shape `[1, 4608, 768]`, finite values, no NaNs.
- [ ] Measured per-clip time × 100 is an *acceptable* total (your call once you see the number — this is exactly why we measure one clip first instead of guessing).
- [ ] Peak RAM during that one clip stays comfortably under ~7.9 GB (no swapping/thrashing).
- [ ] `C:` free space unchanged.

### Step 2 — Implement and check the analysis code
**Do:** write the stimulus/extraction/analysis/plotting scripts per Documents 01–02 (not the stale `run.py`); validate cosine/mean-pooling/transport math on small hand-built vectors (e.g. 3-dimensional toy vectors with a known answer) before touching real embeddings.
**Gate:**
- [ ] Toy-vector tests for mean pooling, vector subtraction sign, cosine, diagonal exclusion, matched-index lookup (`m_k`), and `d_T`/error-ratio all match hand-computed expected values.
- [ ] Scene split (`T`/`U`/`Q`) reproduced from the SHA-256 algorithm matches `scene_plan.json` exactly.
- [ ] Manifest schema (scene, condition, g, x0, y0, RGB, times, codec, SHA-256) implemented and produces valid rows on a dry run.

### Step 3 — Generate and validate all 100 clips
**Do:** render 40 baseline + 40 low-gravity + 20 color clips per the exact scene/color/timing spec.
**Gate:**
- [ ] Exactly 100 files exist, no more, no fewer; filenames map 1:1 to `scene_plan.json`.
- [ ] Every clip: 151 frames, 30 FPS, 384×384, ball stays fully in-frame for all 151 frames.
- [ ] Lossless round-trip check: decoded frames == rendered frames, for every clip (not a sample).
- [ ] Gravity-pair first frames match exactly (same start position); color-pair trajectories match the baseline exactly except RGB.
- [ ] Contact sheet generated and skimmed by eye for anything obviously wrong (wrong color, ball off-screen, frozen frame).
- [ ] Total disk used for 100 clips recorded, confirmed to fit comfortably in `D:`'s 593 GB.

### Step 4 — Extract embeddings
**Do:** run the checked Step-1 pipeline over all 100 clips.
**Gate:**
- [ ] Exactly 100 embeddings saved, keyed by scene+condition, all finite, all shape `[768]` after pooling.
- [ ] No pair (baseline/low-gravity, or baseline/color) is missing its partner.
- [ ] Checkpoint hash, code hash, dataset hash and execution log all recorded alongside the embeddings.
- [ ] Actual total wall-clock time recorded (compare against the Step-1 estimate — large deviation is itself worth noting).

### Step 5 — Run the locked analysis
**Do:** compute `A`, `D`, `R` and every supporting quantity exactly as specified in Document 02 (including the unified hypothesis/calculation/check-box version).
**Gate:**
- [ ] Integrity checks from Document 02 all pass (symmetry, disjoint splits, finite arrays, `e0`/`e1` identities within 1e-10).
- [ ] `d_T` recomputed independently from saved training vectors matches the saved `d_T` exactly.
- [ ] Every undefined-value rule (norm ≤ 1e-12, etc.) checked; any hits reported, not silently patched.
- [ ] No result triggers a re-opening of Step 2/3/4 to "fix" the numbers — a disappointing `A`, `D`, or `R` is a valid outcome, not a bug.

### Step 6 — Report and stop
**Do:** produce the 10 prescribed figures/tables and a report stating H1/H2/H3 outcomes separately, then stop — no follow-up variants without a new, separate request.
**Gate:**
- [ ] All 10 figures/tables generated with the fixed axes/bins/labels from Document 02.
- [ ] Report states each hypothesis's outcome individually before any combined claim, per the "Combined claim & statistical scope" section.
- [ ] Full bundle (code, lockfile, manifest, embeddings, figures, report, exact repro command) saved under `D:\gravity-latent-experiment-runtime\`.

---

## 3. Stop conditions — when to come back and ask, not push through

- Any Step-1 gate fails (checkpoint won't load strictly, RAM thrashes, `C:` gets touched).
- The measured per-clip time makes the 100-clip run impractical (e.g. many hours) — come back with the number before committing.
- Any integrity check in Step 5 fails — this means the *code* is wrong, not that the *hypothesis* failed; it must be fixed and re-verified on toy vectors again before re-running on real data.
- Disk, RAM, or the C: drive situation changes mid-run.

## 4. What this document does not yet decide

**Execution venue.** Documents 00–04 all defer this explicitly ("You will choose the execution environment after reviewing this package"). The hardware table in §0 is this laptop's actual profile — it is usable (meets the CPU float32 requirement) but slow and RAM-tight, with no GPU. That's a real decision point, not a formality: running here is possible but Step 1's timing benchmark should be seen *before* deciding to commit to all 100 clips on this machine versus another one.
