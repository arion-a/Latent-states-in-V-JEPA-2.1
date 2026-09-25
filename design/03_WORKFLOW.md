# 3. Step-by-step workflow and progress

**Current boundary: documents only. No execution is authorized.**

Your next action is to read the design, request changes if needed, and choose where to conduct the experiment. There is no prescribed cloud provider or local machine. Once you explicitly approve a protocol and execution location, the workflow below can be followed within that authorization; it does not require redundant approval between routine steps.

| Step | Purpose | Current status | Required output / completion check |
|---|---|---|---|
| 0 | Review and freeze design | Ready for your review | Approved version, venue and recorded amendments |
| 1 | Verify chosen environment | Not authorized / not started under this protocol | Capacity and dependency report; no silent model/precision changes |
| 2 | Implement approved protocol | Preliminary older draft exists; reconciliation pending | Reproducible scripts, config, lockfile, software checks |
| 3 | Generate and validate stimuli | Not started | Exactly 100 lossless clips, manifest and validation report |
| 4 | Extract final embeddings | Not started | 100×768 embeddings and complete provenance |
| 5 | Compute locked analysis | Not started | A,D,R, matrices, matched table and held-out table |
| 6 | Plot and report | Not started | Prespecified figures, interpretation and reproducible bundle |

## Step 0 — review, resolve and freeze

Read documents 01 and 02 with scene_plan.json. Review the proposed checkpoint, starting-position grid, fixed velocity, five-second motion span, 151-frame videos and fixed 16-frame full-duration sampling, RGB control, rendering details, SHA-256 assignments and descriptive decision criteria. Record any approved changes before any model inference. State the execution venue, artifact storage location and any resource restrictions. Save an approved copy with file hashes; the present draft remains clearly distinguishable.

Exit condition: explicit user authorization for that design in the selected environment. Until then, do not install dependencies, download model weights, render clips, run inference, or produce experimental statistics. Existing preliminary assets do not imply approval.

## Step 1 — environment verification after approval

Check available disk, RAM, compute, network and permissible output locations. Use an isolated runtime and pin Python, PyTorch, model support, renderer, codec and plotting versions. Verify source revision and official checkpoint provenance. Record weight SHA-256 after a complete download. Validate checkpoint loading strictly and device support for the specified float32 operations. Do not infer results from a partially loaded or untrained model.

Benchmark resource use only within the authorized plan, reusing an actual planned clip after Step 3 if needed. CPU or GPU can implement the same protocol; record the device. Do not promise exact cross-device bitwise reproducibility. If the environment cannot support the approved configuration, report the concrete blocker before making scientific substitutions.

Exit condition: environment report and a viable documented execution command. No paid service or external publication is implied by this workflow.

## Step 2 — implement and check

Translate the approved formulas into separate stimulus, extraction, analysis and plotting stages. Replace the preliminary script's split logic with the approved scene assignments and include the H2 matched contrasts, both heatmaps, all required statistics and failure handling. Enforce the approved config rather than reading mutable defaults.

Use small analytical vectors to check mean pooling, vector subtraction signs, cosine, diagonal exclusion, matched indexing, training-only displacement and error ratio. Verify rejection of invalid counts and undefined cosines. These are software verification data, never experimental observations.

Produce a manifest schema with scene ID, condition, g, x₀,y₀,vₓ,vᵧ, RGB, frame times, dimensions, codec and SHA-256. Require complete matching metadata between paired conditions. Record code and configuration hashes. Extraction must refuse incompatible resumes: cached embeddings must match video, checkpoint, config and extraction-code provenance.

Exit condition: implementation matches the approved design and analysis checks pass. Existing draft code is not accepted without these checks.

## Step 3 — generate and inspect the 100 clips

Render exactly the scene-condition list, with orange baseline/low-gravity balls and blue color-control balls; every clip has 151 frames at 30 FPS spanning five seconds of physical motion. Validate trajectory bounds and every frame's dimensions. Verify paired first frames match for the gravity pairs and color pairs have identical trajectory positions. Decode each lossless clip and require equality with its rendered RGB frames. Save a metadata manifest and hashes.

Create a labeled contact sheet from existing clips for inspection: scene ID, condition and sampled time. This is a view of the dataset, not a new intervention condition. Confirm color clips retain baseline gravity and correct scene IDs. If a generation defect exists, correct the implementation consistently before inference; do not hand-select scenes based on visual appeal.

Exit condition: 40 baseline + 40 low-gravity + 20 color clips pass validation; none missing or extra. Record the dataset hash.

## Step 4 — extract final mean-pooled embeddings

Load the approved frozen EMA encoder with strict key matching. Record evaluation mode, disabled gradients, precision and deterministic settings. For each decoded 151-frame clip, select exactly indices 0,10,…,150 and apply only the approved RGB normalization, obtain final normalized tokens, assert [1,4608,768], and average the token axis.

Store each 768-vector with its scene-condition identity and provenance. Ensure no predictor, intermediate layer or hierarchical concatenation was used. Check finite values and complete coverage. Resuming failed execution may reuse only provenance-matched embeddings; no repeated run is chosen because its metrics look better.

Exit condition: exactly 100 valid embeddings with no missing pairs. Save environment versions, checkpoint hash, code hash, dataset hash and an execution log.

## Step 5 — run the prespecified analysis

Use float64 arithmetic on the saved embeddings. Compute gravity/color differences, G, C, aₛ, matched m, δ and primary A,D. Compute d_T from the 30 training IDs alone and save it. Then compute held-out predictions, per-scene errors and primary R. Report numerical failures rather than manipulating the split or metric definitions.

Perform the integrity checks in document 02. Save all arrays, CSV tables and summary JSON, including undefined-value reasons. No additional controls, alternative model runs, embeddings transformations or exploratory variants are included.

Exit condition: all planned computations are complete or explicitly marked unevaluable with a concrete reason; no invented values.

## Step 6 — report, bundle and stop

Create the five figures specified in document 02 and label sample counts and scene IDs. Report A,D,R first, followed by distributions and per-scene transport. Discuss each hypothesis separately before the combined limited claim. Include failures, all ten held-out outcomes, implementation deviations and the synthetic-distribution limitations.

Deliver design, scene plan, code, dependency lock, execution log, manifest, embeddings, matrices, tables, figures and a report. Provide actual artifact locations for large clips and checkpoint provenance; do not claim a large file is bundled if it is not. Include the exact reproduction command for the selected environment.

Exit condition: the original minimal experiment is fully reported. Stop without adding variants or beginning a follow-up study.

## Recording progress later

For each step record status (not started / running / completed / blocked), start/end time, inputs and hashes, output locations, checks, and any deviation. An environment error is a blocker, not a scientific null result. A null or mixed scientific result is a valid completed outcome, not a reason to expand the experiment.
