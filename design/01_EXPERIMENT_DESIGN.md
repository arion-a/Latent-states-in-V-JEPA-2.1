# 1. Proposed experiment design

**Review draft — do not execute.**

## Research question

Does changing simulated gravity from 9.8 to 4.9 m/s² produce a consistently aligned change in the final mean-pooled V-JEPA representation across 40 matched scenes? Is that change more aligned across gravity interventions than with a fixed color intervention? Can the average gravity change from 30 scenes improve predictions of the low-gravity embeddings in 10 unseen scene pairs?

This tests a transferable representation displacement for one binary gravity intervention. It does not measure how many numerical gravity levels are encoded, estimate gravity from an embedding, or establish an abstract gravity variable independent of visible trajectory changes.

## Hypotheses and planned evidence

- **H1 — directional consistency:** gravity intervention vectors tend to align across scenes. Primary quantity: the mean cosine of the 780 distinct gravity-vector pairs. Positive values support positive average alignment; zero or negative values do not. The full distribution must accompany the mean.
- **H2 — relative appearance specificity:** a scene’s gravity vector aligns more with other scenes’ gravity vectors than with its own color vector. Primary quantity: the mean of 20 within-scene contrast scores defined in the mathematical plan. Positive values support this specific comparison; nonpositive values do not. This tests one color change, not all appearance transformations.
- **H3 — held-out transport:** adding the mean training gravity vector improves prediction of held-out low-gravity embeddings over leaving the baseline embedding unchanged. Primary quantity: normalized transport error R. R<1 supports aggregate improvement; R≥1 does not. Per-scene errors show whether improvement is widespread or concentrated.

These are separate descriptive hypotheses. All three must meet their directional criteria for the limited combined claim “average alignment, separation from this color control, and transport improvement in this fixed benchmark.” Mixed outcomes must be reported separately. Small positive differences are weak evidence; crossing zero or one is not a statistical significance claim.

The 40 scenes are a designed grid, not an independent random sample from real videos. No confirmatory population p-values or confidence intervals are proposed. No threshold may be changed after seeing embeddings. This deliberately small initial experiment characterizes effect sizes and feasibility.

## Scope and experimental unit

The experimental unit is a scene identity, not an individual video or a pair of cosine entries. Every scene contributes two gravity clips. Twenty of those same identities contribute one extra color-control clip.

| Condition | Number | Gravity | Ball RGB | Paired reference |
|---|---:|---:|---|---|
| Baseline B | 40 | 9.8 | (235,100,60) | — |
| Low gravity L | 40 | 4.9 | (235,100,60) | Same-scene B |
| Color C | 20 | 9.8 | (60,140,235) | Same-scene B |
| Total | **100** | | | |

No second low-gravity color clip is added. Colors and gravity are not fully crossed. Baseline clips are reused as references, not regenerated as extra experimental videos.

Across scene identities, only starting position changes. Within any gravity pair, only g changes. Within any color pair, only ball RGB changes. Velocity is identical everywhere: it is not a tested factor or control condition.

## Exact proposed physical and visual setup

All 100 videos use a visibly colored ball: orange RGB (235,100,60) for both gravity conditions and blue RGB (60,140,235) for the 20 color controls. One ball appears per video. The same baseline color is used across scenes to keep appearance fixed.

Each video covers **five full seconds of simulated motion**, without slowing a shorter trajectory, looping it or padding it with a frozen frame. Render 151 frames at 30 FPS, sampled at tₖ=k/30 seconds for k=0,…,150. The first-to-last sample span is exactly 5 seconds; the container duration is 151/30≈5.033 seconds including the last frame's display interval, satisfying the minimum five-second requirement.

To keep both gravity trajectories visible for the full five seconds without adding collisions or changing gravity, the proposed world is enlarged to x∈[−100,100] m and y∈[0,200] m. Ball radius r=5 m preserves the previous 9.6-pixel visual radius. For i=0,…,7 and j=0,…,4, scene s=8j+i starts at x₀=−70+10i m, y₀=140+5j m. All scenes use vₓ=15 m/s and an initial vertical velocity vᵧ₀=10 m/s. These world/position/radius/velocity changes are proposed design adaptations for review, applied identically to every condition.

**Revision (design_revision `ballistic-launch-v3`):** the original vₓ=2, vᵧ₀=0 setup (a near-vertical horizontal launch, i.e. rolling off a ledge) visually read as a weak, unresolved drop rather than a projectile — 10 m of horizontal travel against a 122.5 m vertical drop. The ball is now given a genuine ballistic launch (thrown at an angle, vᵧ₀>0), producing the standard rise-then-fall parabola. The x₀ grid spacing was tightened from 20 m to 10 m (range −70…0 instead of −70…70) to keep the larger horizontal travel (75 m over 5 s, up from 10 m) inside the world.

At any t∈[0,5] seconds:

    xₛ(t) = x₀,ₛ + 15t
    yₛ,g(t) = y₀,ₛ + 10t − (g/2)t²

Verified numerically in-bounds for all 40 scenes at both g=9.8 and g=4.9, across all 151 sampled frames: x remains within [−70,75] and y within [67.5,170.2] (world margins x∈(−95,95), y∈(5,195) given r=5). There are no collisions, drag, rotation, perspective, shadows, camera movement, textures, or additional objects — the ball reaches its ballistic apex and continues falling past it, but the clip ends at t=5s before it would reach the ground; there is still no landing/contact event in this minimal design.

Rendering is 384×384 RGB. The orthographic mapping is u=1.92(x+100), v=1.92(200−y), radius 9.6 pixels. Background RGB is (32,32,32). A fixed grid in RGB (44,44,44) occupies vertical and horizontal pixel lines 0,38,76,…,380. Draw the ball over the grid using Pillow's filled ellipse with floating-point bounding box (u−9.6,v−9.6,u+9.6,v+9.6), without added antialiasing. Pin and record the renderer version before generation.

Videos use lossless FFV1 in MKV at 30 FPS, with all 151 decoded RGB frames verified equal to the rendered frames. For the proposed 16-frame encoder input, deterministically select source indices 0,10,20,…,150. These span the full five seconds at times τₗ=l/3 seconds for l=0,…,15. The same sampling applies to every condition. This is sparse full-duration sampling, not a one-second crop; it does not let the model observe all motion between selected frames. This sampling choice is proposed for review and introduces no extra clips or analysis variants.

The color change also changes brightness. It is a broad appearance intervention, not a luminance-matched hue intervention. This is recorded as a limitation, not corrected by adding more conditions.

## Scene assignments and leakage prevention

Exact assignments and all initial conditions are in scene_plan.json. For transparent deterministic assignment, sort IDs 00–39 by ascending SHA-256 of UTF-8 text `gravity-minimal-v1|split|SS`. The first 30 train; the last 10 test. Independently sort hashes of `gravity-minimal-v1|color|SS` and choose the first 20 for color controls. Hash collisions, if any, are resolved by ascending scene ID. This ordering is fixed without reference to embeddings.

The test scenes are [4,22,8,39,33,10,0,1,37,24]. Both gravity clips of a scene remain together. All 40 scenes may enter descriptive H1/H2 summaries, but only the 30 training gravity vectors may determine the transport displacement. Color embeddings never enter transport fitting. No test-driven scaling, representation selection, or protocol revision is permitted.

## Proposed representation

Use the frozen **V-JEPA 2.1 ViT-B/16 EMA encoder**, the model named in the supplied conversation preview. This remains a model choice for approval, rather than an assumption that all V-JEPA versions are interchangeable.

- Official source: https://github.com/facebookresearch/vjepa2
- Inspected revision: `204698b45b3712590f06245fbfba32d3be539812`.
- Official checkpoint: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
- Load `ema_encoder` strictly into the official base encoder; freeze all parameters and use evaluation mode.
- Input shape [1,3,16,384,384], float32. Divide RGB by 255; normalize channels using means (0.485,0.456,0.406) and standard deviations (0.229,0.224,0.225).
- Select exactly source frames 0,10,…,150 from each 151-frame video. No further temporal sampling, crop, resize, flip, random augmentation, or model training.
- Use the final transformer block output after its model LayerNorm, with hierarchical output disabled. Expected shape [1,4608,768]: 8 temporal positions × 24×24 spatial positions.
- Mean all 4608 tokens to obtain one 768-dimensional vector. Do not L2-normalize the embedding before computing intervention differences. No predictor features are used.

The source revision's hub loader points to localhost for downloads; the future implementation must use the official public checkpoint directly, without altering the model architecture. Record checkpoint SHA-256, source commit, package versions, device and deterministic settings. Float32 is proposed; do not silently substitute reduced precision to accommodate an environment.

## Exclusions and limits

No additional gravity values, velocity controls, multi-layer probing, PCA, learned subspaces, learned probes, alternative pooling, multiple checkpoints, data augmentation, or extra variants. No claim of causal disentanglement from motion: gravity necessarily changes visible positions and trajectories here. Held-out scenes are new positions inside the same synthetic setup, not a new renderer or physical regime. Findings apply to this model, stimulus distribution and binary contrast.
