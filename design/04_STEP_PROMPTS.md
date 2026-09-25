# 4. Copyable prompts for the future execution steps

**These prompts are prepared for later use. They do not authorize running anything now.**

Give the executing agent documents 01–03 and scene_plan.json. Replace bracketed placeholders with real approved values. The existing unexecuted run.py is only a draft; these documents are authoritative after user approval. If a prompt is used without the required approval or inputs, the agent should identify the missing prerequisite rather than assume it exists.

## Prompt 0 — review and finalize the design, without execution

```text
Review the attached minimal gravity experiment design, mathematical plan, workflow and scene_plan.json. Do not run the experiment, install software, download weights, render videos, or extract embeddings. Check whether the hypotheses, definitions, sample counts, paired controls, deterministic assignments, final-layer pooling and 30-train/10-test transport are mutually consistent. Identify unresolved choices or unsupported claims and propose document-only corrections. Keep exactly 40 paired scenes at g=9.8 and 4.9 plus 20 baseline color controls, total 100 videos. Require colored balls and at least five seconds per video, with the proposed 151-frame, 30-FPS rendering and fixed full-duration sampling. Do not introduce variants. Return a review and a finalized proposed protocol for my approval. Stop so I can read it and choose an execution environment.
```

## Prompt 1 — check the selected environment

```text
I have approved protocol [VERSION / DOCUMENT HASHES] for execution in [ENVIRONMENT], with artifacts stored at [LOCATION] and resource limits [LIMITS]. Read the attached approved documents. Check whether this environment can run the specified frozen V-JEPA 2.1 ViT-B/16 EMA encoder, 151-frame, 30-FPS colored-ball videos spanning five seconds, sampled at indices 0,10,…,150 for 16×384×384 model inputs, float32 inference, lossless rendering and analysis. Prepare an isolated, pinned runtime within this authorization; verify the exact official source commit and checkpoint and record SHA-256. Document capacity, versions, device, deterministic settings and commands. Do not change scientific parameters or choose a different checkpoint or precision. If blocked, identify the specific requirement and what resource is missing. Deliver the environment report; do not render the dataset or conduct inference in this step.
```

## Prompt 2 — implement and validate the approved protocol

```text
Using approved protocol [VERSION] and environment [LOCATION], implement its stimulus generation, final-layer extraction, analysis and plots. Treat the approved documents and scene_plan.json as authoritative. The older draft run.py must be reconciled, especially its scene split, matched H2 contrast, complete outputs and numerical failure handling. Add provenance-safe resume behavior. Validate formulas on small hand-computable vectors and validate scene/condition metadata without generating experimental observations. Enforce 100 clips only, no test leakage, no embedding normalization before differences, no learned scaling and no prohibited variants. Deliver source, configuration, dependency lock, exact stage commands and a check report. Stop before rendering experimental clips.
```

## Prompt 3 — generate and validate exactly 100 videos

```text
Use the approved configuration and checked implementation from [LOCATION]. Render exactly 40 baseline clips at g=9.8, their 40 matched clips at g=4.9, and 20 color-only clips for the exact approved baseline scene IDs. Use orange (235,100,60) for both gravity conditions and blue (60,140,235) for controls. Each video must contain 151 frames at 30 FPS spanning five actual simulated seconds, without loops, slowed playback or frozen-frame padding. Do not modify assignments, physics, colors, timing or rendering settings. Validate bounds, exact frame count and size, paired metadata, identical initial gravity-pair frames, color-control trajectory equality and lossless RGB encode/decode equality. Save all clips, a complete manifest with hashes, a validation report and a labeled contact sheet drawn only from these clips. Report any failure. Do not extract embeddings or add clips in this step.
```

## Prompt 4 — extract only final mean-pooled embeddings

```text
Use approved protocol [VERSION], the validated 100-video manifest [PATH], pinned model source [PATH] and verified official checkpoint [PATH]. Strictly load the EMA encoder, freeze parameters and use evaluation mode with the approved float32 deterministic configuration. Decode the 151-frame, 30-FPS video and select exactly RGB frame indices 0,10,…,150, covering all five seconds, with no crop, resize or augmentation, apply the documented normalization, and obtain only the final normalized block tokens. Assert [1,4608,768], then mean-pool the token axis to 768 dimensions. Do not use predictor features, intermediate layers, hierarchical concatenation or L2-normalize the pooled embeddings. Save exactly 100 embeddings keyed by scene and condition, with full checkpoint/video/config/code provenance, finite-value checks and execution log. Resume only matching cached outputs. Stop before experimental statistical analysis.
```

## Prompt 5 — compute only the prespecified analysis

```text
Read the approved mathematical plan and complete embedding artifacts at [LOCATION]. In float64 compute g_s=z_low−z_baseline and c_s=z_color−z_baseline. Compute all 780 unique gravity-pair cosines, the 40×20 gravity–color matrix, the 20 matched color cosines, a_s, the 20 matched contrasts and primary A,D. Estimate d_T only from the fixed 30 training gravity vectors; save it before held-out calculations. Predict each of the ten held-out low-gravity embeddings by adding d_T to its baseline embedding. Compute e0,e1,r,h, both RMSEs, primary R, fraction improved and the prescribed summaries. Follow undefined-vector rules exactly. Verify indexing, symmetry, counts, training provenance and error identities. Save arrays, CSV tables and summary JSON. Do not add statistical tests, fit test-set scales, change assignments or run variants. Report missing or unevaluable quantities explicitly.
```

## Prompt 6 — produce the report and plots, then stop

```text
Using only the verified outputs of approved protocol [VERSION] at [LOCATION], produce its five prescribed figures: gravity cosine heatmap, gravity–color heatmap, gravity-versus-matched-color histogram, matched-contrast plot, and ten-scene transport error plot. Follow fixed axes, bins and labels in the mathematical plan. Report primary A,D,R with exact sample counts and interpret H1,H2,H3 separately. Include all held-out outcomes, numerical failures, protocol deviations, model/renderer provenance and reproduction instructions. Do not imply pairwise entries are independent or claim a numerical gravity code, causal disentanglement or generalization beyond the fixed benchmark. Package design, code, locks, manifest, embeddings, tables, figures and report; give actual locations for large data. Present the result candidly even if null or mixed. Stop after the minimal experiment; propose or run no variants unless separately requested.
```

## Optional single authorization for the complete approved sequence

The prompts above can be used individually for staged supervision. If you later prefer a single continuous execution after approval, use this orchestration prompt with the same attachments:

```text
I approve protocol [VERSION / HASHES] and authorize Steps 1–6 in [ENVIRONMENT] with outputs at [LOCATION] and limits [LIMITS]. Follow the attached workflow and corresponding step prompts sequentially, treating their intermediate stop instructions as checkpoints to verify and then continue under this authorization. Do not ask again for routine already-authorized steps. Report progress and concrete blockers. Do not change scientific choices, add variants, use unauthorized paid resources or publish externally. Complete the exact 100-video experiment, deliver reproducible results and plots, then stop.
```
