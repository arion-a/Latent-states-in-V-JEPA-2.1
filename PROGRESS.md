# Gravity experiment — live progress tracker

Legend: `[x]` done · `[~]` in progress right now · `[ ]` not started · `[!]` blocked

Re-open this file any time to see exactly what's happening. Results get posted directly into this file the moment each hypothesis finishes computing — not batched at the end.

---

## Phase 0 — Finalize the physics fix (vx=15, vy0=10 ballistic launch) — COMPLETE
- [x] Verified in-bounds for all 40 scenes × both g values × all 151 frames
- [x] `scene_plan.json` updated (design_revision: `ballistic-launch-v3`)
- [x] Propagated into `01_EXPERIMENT_DESIGN.md` (equations, bounds text)
- [x] Propagated into `02_MATHEMATICAL_PLAN.html` appendix

## Phase 1 — Cloud environment (RunPod RTX 3090) — COMPLETE
- [x] Pod deployed, SSH confirmed working
- [x] torch 2.8.0+cu128 pre-installed, CUDA available, 24GB VRAM, 30GB disk free
- [x] Remaining deps installed on pod (torchvision, timm, einops, pillow, numpy, imageio, imageio-ffmpeg, matplotlib)
- [x] `vjepa2` repo cloned on pod at pinned commit `204698b45b3712590f06245fbfba32d3be539812`
- [x] Checkpoint on pod, SHA-256 verified against local copy (`848a77c3...59ddf4d`, exact match)
- [x] Encoder strict-loads with zero missing/unexpected keys

## Phase 2 — Write the pipeline (locally, then smoke-test on the pod) — COMPLETE
- [x] `render_all.py` — renders clips per `scene_plan.json` (lossless FFV1)
- [x] `extract.py` — loads encoder, produces one 768-d mean-pooled embedding per clip
- [x] `analyze.py` — H1/H2/H3 math per `02_MATHEMATICAL_PLAN.md` (dummy-run tolerant: reports how much of T/U/Q is actually present)
- [x] **Dummy run**: scenes 0 (test-set) + 2 (train-set), 6 clips (baseline/low-g/color each), full render→extract→analyze on the pod GPU. Shapes correct `[1,4608,768]`, strict load clean, finite values, ~0.4s/clip extraction. Pipeline ran end to end with zero crashes.
- [x] No bugs found — dummy run passed on first attempt

**Measured timing (real number, not an estimate): ~0.4s/clip extraction on the RTX 3090 → full 100-clip extraction ≈ 40–75s. Rendering ≈ seconds. Total full run likely well under 5 minutes of compute.**

## Phase 3 — Full run (100 clips) — COMPLETE

Measured wall-clock (RTX 3090 pod): render 67.8s, extract 49.4s, analyze 0.24s. **Total ≈ 2 minutes of compute.**

- [x] All 100 clips rendered + lossless round-trip validated
- [x] All 100 embeddings extracted (100/100, all finite, all `[4608,768]` pre-pool)
- [x] **H1 result**: A = **0.634761** over all 780 pairs (full design count). A>0 → **supports H1**.
- [x] **H2 result**: D = **0.598393** over all 20 matched scenes (full design count). D>0 → **supports H2**.
- [x] **H3 result**: R = **0.600966** over 30 train / 10 test scenes (full design counts). R<1 → **supports H3**.

All three hypotheses met their directional criteria — see `04_STEP_PROMPTS.md`/`02_MATHEMATICAL_PLAN.html` §5 for what the combined claim is and is not entitled to say (descriptive only, no significance test, small designed grid not a random sample). Raw artifacts: `D:\gravity-latent-experiment-runtime\results\` (`analysis_summary.json` has full per-scene tables, `embeddings.json` has all 100 raw 768-d vectors, `manifest.json` has clip provenance/hashes).

## Phase 4 — Report
- [ ] Figures generated (10 prescribed plots/tables)
- [ ] Final report + full artifact bundle

---

## Current focus
Finishing Phase 0 (doc propagation), then moving to Phase 1 (finishing pod dependency install) and Phase 2 (writing the pipeline) in parallel.
