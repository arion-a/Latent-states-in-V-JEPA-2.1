# 2. Complete mathematical analysis plan

**Review draft — formulas only; no results have been computed.**

## Definitions and dimensions

Let S={0,…,39}, Q⊂S the 20 color-control scene IDs, T⊂S the 30 training IDs, and U=S\T the 10 test IDs. Fix all sets before rendering or inference.

Let Vₛᴮ,Vₛᴸ,Vₛᶜ denote baseline, low-gravity and color videos; Vₛᶜ exists only for s∈Q. Define the fixed sampling operator P(V)=(V[0],V[10],…,V[150]) for each 151-frame, 30-FPS video. These 16 frames cover times 0 to 5 seconds. Let F be the frozen encoder's final normalized token output. With N=4608 and d=768, define:

    Zₛᵃ = F(P(Vₛᵃ)) ∈ R^(N×d)
    zₛᵃ = (1/N) Σₙ₌₁ᴺ Zₛᵃ[n,:] ∈ R^d, a∈{B,L,C}
    gₛ = zₛᴸ − zₛᴮ, s∈S
    cₛ = zₛᶜ − zₛᴮ, s∈Q

The physical intervention sign is always low gravity minus baseline gravity. The appearance sign is always changed color minus baseline color. Mean pooling precedes differencing; no embedding centering or normalization is inserted. Store extracted embeddings in float32; cast to float64 for the following calculations.

For nonzero a,b∈R^d:

    <a,b> = Σⱼ₌₁ᵈ aⱼbⱼ
    ||a||₂ = sqrt(Σⱼ₌₁ᵈ aⱼ²)
    cos(a,b) = <a,b> / (||a||₂ ||b||₂)

A vector with norm ≤10^−12 has undefined cosine in this protocol. Record its ID and magnitude; do not replace its cosine with zero, add an epsilon to create a result, or silently drop it. Mark the affected hypothesis unevaluable as defined, while retaining valid raw data and other unaffected calculations. Record all gravity/color vector norms as numerical diagnostics, not additional hypotheses.

## H1: gravity–gravity alignment

Define G∈R^(40×40) by Gₛₜ=cos(gₛ,gₜ). The diagonal equals one mathematically and is excluded from summaries.

    P = {(s,t): 0≤s<t≤39}, |P|=40·39/2=780
    A = (1/780) Σ_(s,t)∈P Gₛₜ
    aₛ = (1/39) Σ_(t∈S,t≠s) Gₛₜ

A is the primary H1 quantity. Report the 780 values' mean, median, population standard deviation sqrt(mean((x−mean(x))²)), min and max; also store all 40 aₛ. A>0 means positive average alignment for this benchmark. Report effect magnitude and heterogeneity; no binary “gravity understood” label follows from its sign.

## H2: gravity–color comparison

Order Q numerically as q₀,…,q₁₉. Define C∈R^(40×20):

    Cₛₖ = cos(gₛ,c_(qₖ))
    mₖ = C_(qₖ),ₖ                         [same-scene comparison]
    δₖ = a_(qₖ) − mₖ                     [within-scene contrast]
    D = (1/20) Σₖ₌₀¹⁹ δₖ

D is the primary H2 quantity. D>0 means each controlled scene's gravity vector is, on average, better aligned with gravity changes elsewhere than with that same scene's appearance change.

Report all 20 matched cosines mₖ, all 20 contrasts δₖ, D, and the mean/median/population-SD/min/max of the matched cosines. Save the full 800-entry cross matrix and its descriptive summaries as secondary context. Do not replace the matched primary comparison with whichever cross-matrix statistic looks more favorable. The 800 cross-pairs and 780 gravity pairs reuse vectors and are not independent samples.

A color cosine near zero is orthogonality in this representation, not proof of causal independence. A negative value is an oppositely directed change, not simply “more specific.” Color and gravity vectors also share a baseline embedding; acknowledge this dependence.

## H3: 30-train/10-test transport

The only fitted object is the mean training displacement:

    d_T = (1/30) Σ_(s∈T) gₛ
    ẑₛᴸ = zₛᴮ + d_T, s∈U

No regression, learned scale, intercept, whitening, unit-vector averaging, target normalization, or test-set calibration is applied. This estimates both direction and step size from raw training displacements. “Train” here means estimating this mean, not updating V-JEPA weights.

For each held-out scene:

    e₀,ₛ = ||zₛᴮ − zₛᴸ||₂ = ||gₛ||₂       [identity/no-transport error]
    e₁,ₛ = ||ẑₛᴸ − zₛᴸ||₂ = ||d_T−gₛ||₂   [transport error]
    rₛ = e₁,ₛ/e₀,ₛ                        [individual error ratio]
    hₛ = cos(d_T,gₛ)                       [held-out direction alignment]

Aggregate quantities:

    RMSE₀ = sqrt((1/10) Σ_(s∈U) e₀,ₛ²)
    RMSE₁ = sqrt((1/10) Σ_(s∈U) e₁,ₛ²)
    R = RMSE₁ / RMSE₀                      [primary H3 quantity]
    improvement = 1−R                     [relative RMSE reduction]
    f = (1/10) Σ_(s∈U) 1[e₁,ₛ < e₀,ₛ]    [fraction improved]
    H = (1/10) Σ_(s∈U) hₛ

R<1 is aggregate improvement; R=1 is no aggregate improvement; R>1 is worse transport. R=0 would be perfect prediction. Report R, both RMSEs, f, H, ||d_T||₂, and all ten rows of e₀,e₁,r,h. Do not confuse R with the mean of individual ratios, and do not call Euclidean errors squared errors.

If e₀,ₛ≤10^−12, rₛ is undefined and must be marked accordingly. If ||d_T||₂≤10^−12, directional cosines hₛ are undefined although additive predictions and absolute errors still exist. If RMSE₀≤10^−12, R is undefined and H3 cannot be evaluated as planned. No replacement split is allowed.

## Expected outputs and plots

1. Embedding table: 100 rows × 768 values, with scene and condition identifiers and video hashes.
2. Gravity vectors: 40×768; color vectors: 20×768; gravity cosine matrix: 40×40; gravity–color cosine matrix: 40×20.
3. Matched-control table: scene ID, aₛ, matched color cosine, δ, and vector norms.
4. Transport table: the ten scene IDs, e₀,e₁,r,h, plus a separate saved training mean and predicted embeddings.
5. Summary JSON: A,D,R and all prespecified descriptive summaries; undefined quantities explicitly represented and explained.
6. Gravity heatmap: fixed color scale [−1,1], all scene IDs, diagonal shown but excluded from statistics.
7. Gravity–color heatmap: fixed scale [−1,1], rows and columns labeled by their actual scene IDs.
8. Cosine histogram: 30 equal bins with edges −1+2k/30 for k=0,…,30; separately normalized densities for 780 gravity pairs and 20 matched color pairs. Note unequal sample counts and dependence.
9. Matched contrast plot: 20 δ values against scene ID with a zero line.
10. Transport plot: paired e₀/e₁ bars for each test ID; annotate aggregate R. No error bars implying independent pair sampling.

Clamp cosines to [−1,1] only for tiny floating-point boundary overshoot after verifying the excess is <10^−10; larger excess is an implementation error. Keep original values in audit data.

## Integrity checks

Check sample counts, unique IDs, disjoint train/test sets, matching physical metadata, exact RGB round-trip, finite [4608,768] token arrays, finite 768-vectors, the expected cosine symmetry/diagonal within 10^−10, and identities e₁=||d_T−gₛ|| and e₀=||gₛ||. Validate analysis on small hand-computable vectors before access to experimental embeddings. Such software checks are not additional experimental conditions.

Compute d_T using only the recorded T IDs; save it before producing test metrics. Recompute from the saved training vectors to verify provenance. If any implementation defect is discovered after inference, document the defect, correction and affected outputs. Never silently change a scientific design choice to improve results.
