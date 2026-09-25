# Latent states in V-JEPA 2.1

Testing the latent states of the learned representations in V-JEPA 2.1.

A minimal, fully-controlled experiment testing whether a frozen, pretrained **V-JEPA 2.1 ViT-B/16** video encoder represents a change in simulated gravity as a consistent direction in its latent embedding space — without any fine-tuning, probes, or labels.

**Status: executed. All three hypotheses below were tested on 100 real rendered videos and 100 real extracted embeddings; nothing here is simulated or projected.**

## Result summary

| Hypothesis | Quantity | Value | Threshold | Verdict |
|---|---|---|---|---|
| **H1** — gravity vectors align across scenes | `A` (mean of 780 cosine pairs) | **0.6348** | A > 0 | ✅ supports H1 |
| **H2** — gravity change ≠ a color change | `D` (mean of 20 within-scene contrasts) | **0.5984** | D > 0 | ✅ supports H2 |
| **H3** — the direction transports to unseen scenes | `R` (RMSE ratio, 30 train / 10 held-out) | **0.6010** | R < 1 | ✅ supports H3 |

Additional H3 detail: **f = 1.0** — all 10 held-out scenes individually improved under the learned displacement, not just a couple; **H = 0.806** — mean alignment between the learned direction and each held-out scene's true gravity vector. Full raw numbers, per-scene tables, and the exact math behind every quantity are in [`design/02_MATHEMATICAL_PLAN.md`](design/02_MATHEMATICAL_PLAN.md) and [`results/`](results/).

## What this tests, precisely

For 40 synthetic scenes (a ball launched on a ballistic trajectory), each scene is rendered twice — once at Earth gravity (9.8 m/s²), once at Mars-like gravity (4.9 m/s²) — with *everything else pixel-identical*: same start position, same launch velocity, same ball color, same camera, same background. 20 of those scenes also get a third clip where only the ball's color changes (orange → blue), gravity held fixed, as a control.

Each of the 100 resulting 5-second videos is passed through the **frozen** V-JEPA 2.1 encoder (no training, no fine-tuning), and its final-layer tokens are mean-pooled into one 768-dimensional vector. Subtracting the two gravity-condition embeddings for a scene gives a **gravity intervention vector** — literally "the direction gravity pushes the model's internal representation." The three hypotheses ask: do these directions point roughly the same way across different scenes (H1), is that direction distinguishable from a mere appearance change (H2), and does the *average* direction learned from 30 scenes correctly predict the embedding shift in 10 scenes it never saw (H3)?

This is a descriptive, feasibility-level result on one synthetic renderer and one checkpoint — see **Limitations** below for exactly what it does and doesn't establish.

## Repository structure

```
design/       Full experiment design, math, methodology, and execution docs (read 00 first)
pipeline/     The actual code that ran: render_all.py, extract.py, analyze.py, report_figures.py
videos/       All 100 rendered clips (lossless FFV1-in-MKV, ~34 MB total)
results/      Raw embeddings, full cosine matrices, per-scene tables, summary JSON, and the 5 figures
PROGRESS.md   Live build log of how this was actually run, step by step
```

Read the design docs in order:
1. [`00_READ_FIRST.md`](design/00_READ_FIRST.md) — overview
2. [`01_EXPERIMENT_DESIGN.md`](design/01_EXPERIMENT_DESIGN.md) — hypotheses, scope, exact stimuli/model
3. [`02_MATHEMATICAL_PLAN.md`](design/02_MATHEMATICAL_PLAN.md) (also as a readable [`.html`](design/02_MATHEMATICAL_PLAN.html)) — every formula, defined precisely
4. [`03_WORKFLOW.md`](design/03_WORKFLOW.md) — the step-by-step execution process and gates
5. [`04_STEP_PROMPTS.md`](design/04_STEP_PROMPTS.md) — reusable prompts for each stage
6. [`05_EXECUTION_CHECKLIST.md`](design/05_EXECUTION_CHECKLIST.md) — environment/pre-flight checklist actually used

## Model

Frozen **V-JEPA 2.1 ViT-B/16 EMA encoder** (`ema_encoder` weights), official checkpoint `vjepa2_1_vitb_dist_vitG_384.pt` from [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) at commit `204698b45b3712590f06245fbfba32d3be539812`. Checkpoint SHA-256: `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`. Loaded strictly (zero missing/unexpected keys), float32, evaluation mode, no gradients. 16 frames sampled per clip (indices 0,10,…,150 of a 151-frame video), input `[1,3,16,384,384]`, output `[1,4608,768]` mean-pooled to a single 768-d vector per clip.

## Reproducing

```sh
# 1. Render all 100 clips (per design/scene_plan.json)
python pipeline/render_all.py --plan design/scene_plan.json --out videos

# 2. Extract embeddings (needs the vjepa2 repo + checkpoint above; GPU recommended)
python pipeline/extract.py --repo /path/to/vjepa2 --checkpoint /path/to/vjepa2_1_vitb_dist_vitG_384.pt

# 3. Run the locked H1/H2/H3 analysis
python pipeline/analyze.py

# 4. Generate all tables and figures
python pipeline/report_figures.py
```

On a single RTX 3090: rendering all 100 clips took 68s, extracting all 100 embeddings took 49s, and the full analysis took under a second — about 2 minutes of compute end to end.

## Limitations (stated plainly, not buried)

- **Descriptive, not a significance test.** The 40 scenes are a designed grid, not a random sample. No p-values or confidence intervals are computed or implied; crossing 0 or 1 is a descriptive threshold, not statistical significance.
- **One synthetic renderer, one checkpoint.** These results say nothing about whether the same signal survives on real video, a different renderer, or a different model/checkpoint.
- **Gravity necessarily changes the visible trajectory.** This experiment cannot by itself separate "the model represents gravity as an abstract variable" from "the model represents the resulting trajectory shape," since the two are causally linked in the stimuli by construction.
- **The color control is one specific appearance change** (orange→blue at fixed gravity), not an exhaustive test of all possible confounds; it also changes luminance.
- All three hypotheses are separate and descriptive; a combined claim requires all three to hold, and mixed outcomes would be reported as mixed (they were not, here — all three held).

Full precise wording of every claim, threshold, and caveat is in [`design/01_EXPERIMENT_DESIGN.md`](design/01_EXPERIMENT_DESIGN.md) and [`design/02_MATHEMATICAL_PLAN.md`](design/02_MATHEMATICAL_PLAN.md).
