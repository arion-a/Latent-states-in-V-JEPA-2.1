# Minimal gravity latent experiment — review package

**Status: proposed design only. Execution is stopped.**

Read these documents in order:

1. [Experiment design](01_EXPERIMENT_DESIGN.md): question, scope, hypotheses, exact proposed stimuli and model.
2. [Mathematical analysis plan](02_MATHEMATICAL_PLAN.md): every vector operation, metric, interpretation rule, and failure rule.
3. [Step-by-step workflow](03_WORKFLOW.md): dependencies, outputs, checks, and the execution boundary.
4. [Prompts for each step](04_STEP_PROMPTS.md): reusable instructions for conducting the approved experiment later.
5. [Exact scene assignments](scene_plan.json): all 40 initial conditions, train/test IDs, and color-control IDs.

## What is fixed by your request

40 base scenes × two gravity values (9.8 and 4.9), plus 20 color-control clips drawn from the same baseline scenes: exactly 100 videos. Frozen V-JEPA, final layer, token mean pooling, gravity-vector cosine comparisons, gravity–color comparisons, and a 30-train/10-test transport test. No additional variants. Every video must show a colored ball and last at least five seconds.

## Latest revision

Your colored-ball and minimum-five-second requirements are incorporated throughout. Proposed implementation: orange balls for both gravity conditions, blue balls for the color controls, and 151 frames at 30 FPS covering five seconds of physical motion. The world and starting positions are enlarged to keep the entire trajectory visible; the encoder uses 16 evenly spaced frames covering the full duration. All these implementation details remain a design for review. No experiment has been run.

## What is proposed for your review

The precise model checkpoint, scene starting positions, velocity, exact duration above the five-second minimum, rendering details, color change, deterministic assignment algorithm, transport metric, and descriptive hypothesis criteria. These are explicit design choices, not claims that we recovered the original historical settings. The complete earlier conversation was not accessible.

You will choose the execution environment after reviewing this package. Nothing here authorizes execution now. The step prompts are for later use after that decision.

## Current state and prior preparation

No videos were generated, no embeddings were extracted, and no experimental results exist. Before your stop instruction, draft code was written, the official source was downloaded to /tmp/gravity-vjepa2, and runtime/checkpoint downloads began under /mnt/d/gravity-latent-experiment-runtime. Active preparation processes have been terminated. Partial download/cache files may remain; no deletion is needed to review this design.

The earlier ../run.py and ../README.md are unexecuted preliminary drafts, **not the approved protocol**. This review package supersedes their design choices. In particular, this package uses colored balls, five-second trajectories, enlarged world geometry and transparent SHA-256 scene ordering; the earlier code still uses the old short duration, geometry, near-white baseline and NumPy permutation. Later implementation must be reconciled to these documents after approval. Do not run the existing script as though it implements this package exactly.

No results, effects, significance, or runtime estimates are asserted in this package.
