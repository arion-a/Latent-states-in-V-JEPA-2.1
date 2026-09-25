"""Generate all prescribed outputs from 02_MATHEMATICAL_PLAN.md Section 'Expected outputs
and plots': embedding table, gravity/color vectors+matrices, matched-control table,
transport table, full summary JSON, and all 5 figures (2 heatmaps, histogram, matched
contrast plot, transport plot). Fixed axes/bins/labels as specified in the design doc.
Usage: python report_figures.py [--embeddings PATH] [--plan PATH] [--manifest PATH] [--out DIR]"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return None
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default="/workspace/gravity/embeddings.json")
    ap.add_argument("--plan", type=str, default="/workspace/gravity/scene_plan.json")
    ap.add_argument("--manifest", type=str, default="/workspace/gravity/manifest.json")
    ap.add_argument("--out", type=str, default="/workspace/gravity/report")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    emb = json.loads(Path(args.embeddings).read_text())
    plan = json.loads(Path(args.plan).read_text())
    manifest = {r["name"]: r for r in json.loads(Path(args.manifest).read_text())}
    T = sorted(plan["train_scene_ids"])
    U = sorted(plan["test_scene_ids"])
    Q = sorted(plan["color_control_scene_ids"])
    S = list(range(40))

    by_scene = {}
    for name, row in emb.items():
        by_scene.setdefault(row["scene"], {})[row["condition"]] = np.array(row["embedding"], dtype=np.float64)

    # ---- 1. Embedding table ----
    with open(out / "tables" / "embedding_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "scene", "condition", "gravity", "sha256"] + [f"d{i}" for i in range(768)])
        for name, row in sorted(emb.items()):
            m = manifest.get(name, {})
            w.writerow([name, row["scene"], row["condition"], row["gravity"], m.get("sha256", "")] + row["embedding"])

    # ---- gravity/color vectors ----
    g = {s: by_scene[s]["g49"] - by_scene[s]["g98"] for s in S}
    c = {s: by_scene[s]["color"] - by_scene[s]["g98"] for s in Q}

    # ---- gravity cosine matrix G (40x40) ----
    G = np.zeros((40, 40))
    for i in S:
        for j in S:
            G[i, j] = 1.0 if i == j else cos(g[i], g[j])
    with open(out / "tables" / "gravity_cosine_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [str(s) for s in S])
        for i in S:
            w.writerow([str(i)] + [f"{G[i,j]:.6f}" for j in S])

    pairs = [(i, j) for i in S for j in S if i < j]
    cos_vals = np.array([G[i, j] for i, j in pairs])
    assert len(cos_vals) == 780
    A = float(np.mean(cos_vals))
    a_s = {s: float(np.mean([G[s, t] for t in S if t != s])) for s in S}

    # ---- gravity-color cosine matrix C (40x20) ----
    C = np.zeros((40, 20))
    for i in S:
        for k, qk in enumerate(Q):
            C[i, k] = cos(g[i], c[qk])
    with open(out / "tables" / "gravity_color_cosine_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [str(q) for q in Q])
        for i in S:
            w.writerow([str(i)] + [f"{C[i,k]:.6f}" for k in range(20)])

    m = {qk: C[qk, k] for k, qk in enumerate(Q)}
    delta = {qk: a_s[qk] - m[qk] for qk in Q}
    D = float(np.mean(list(delta.values())))
    matched_vals = np.array(list(m.values()))

    # ---- matched-control table ----
    with open(out / "tables" / "matched_control_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "a_s", "matched_color_cosine_m", "delta", "norm_g_s", "norm_c_s"])
        for qk in Q:
            w.writerow([qk, a_s[qk], m[qk], delta[qk], np.linalg.norm(g[qk]), np.linalg.norm(c[qk])])

    # ---- H3 transport ----
    d_T = np.mean([g[s] for s in T], axis=0)
    norm_dT = float(np.linalg.norm(d_T))
    rows = []
    predicted = {}
    for s in U:
        z_pred = by_scene[s]["g98"] + d_T
        predicted[s] = z_pred.tolist()
        e0 = float(np.linalg.norm(g[s]))
        e1 = float(np.linalg.norm(d_T - g[s]))
        rows.append(dict(scene=s, e0=e0, e1=e1, r=e1 / e0, h=cos(d_T, g[s])))
    e0s = np.array([r["e0"] for r in rows])
    e1s = np.array([r["e1"] for r in rows])
    RMSE0 = float(np.sqrt(np.mean(e0s ** 2)))
    RMSE1 = float(np.sqrt(np.mean(e1s ** 2)))
    R = RMSE1 / RMSE0
    f_improved = float(np.mean([1.0 if r["e1"] < r["e0"] else 0.0 for r in rows]))
    H = float(np.mean([r["h"] for r in rows]))

    with open(out / "tables" / "transport_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "e0", "e1", "r", "h"])
        for r in rows:
            w.writerow([r["scene"], r["e0"], r["e1"], r["r"], r["h"]])
    np.save(out / "tables" / "d_T.npy", d_T)
    (out / "tables" / "predicted_embeddings.json").write_text(json.dumps(predicted))

    # ---- full summary JSON ----
    summary = dict(
        H1=dict(A=A, n_pairs=780, mean=A, median=float(np.median(cos_vals)),
                 std_population=float(np.sqrt(np.mean((cos_vals - A) ** 2))),
                 min=float(np.min(cos_vals)), max=float(np.max(cos_vals)), a_s={str(k): v for k, v in a_s.items()}),
        H2=dict(D=D, n_matched=20, matched_mean=float(np.mean(matched_vals)),
                 matched_median=float(np.median(matched_vals)),
                 matched_std_population=float(np.sqrt(np.mean((matched_vals - np.mean(matched_vals)) ** 2))),
                 matched_min=float(np.min(matched_vals)), matched_max=float(np.max(matched_vals)),
                 matched_cosines={str(k): v for k, v in m.items()},
                 contrasts={str(k): v for k, v in delta.items()}),
        H3=dict(R=R, RMSE0=RMSE0, RMSE1=RMSE1, improvement=1 - R, f=f_improved, H=H, norm_dT=norm_dT,
                 n_train=len(T), n_test=len(U), rows=rows),
        undefined_values=[],
    )
    (out / "analysis_summary_full.json").write_text(json.dumps(summary, indent=2))

    # ================= FIGURES =================
    # 6. Gravity heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(G, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(40)); ax.set_yticks(range(40))
    ax.set_xticklabels(S, fontsize=5, rotation=90); ax.set_yticklabels(S, fontsize=5)
    ax.set_title(f"Gravity-gravity cosine matrix G (40x40)\nA = mean of 780 off-diagonal pairs = {A:.4f}")
    ax.set_xlabel("scene t"); ax.set_ylabel("scene s")
    fig.colorbar(im, ax=ax, label="cos(g_s, g_t)")
    fig.tight_layout(); fig.savefig(out / "figures" / "01_gravity_heatmap.png", dpi=150); plt.close(fig)

    # 7. Gravity-color heatmap
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(20)); ax.set_yticks(range(40))
    ax.set_xticklabels(Q, fontsize=6, rotation=90); ax.set_yticklabels(S, fontsize=5)
    ax.set_title(f"Gravity-color cosine matrix C (40x20)\nD = mean of 20 matched contrasts = {D:.4f}")
    ax.set_xlabel("color-control scene (column)"); ax.set_ylabel("scene s (gravity vector)")
    fig.colorbar(im, ax=ax, label="cos(g_s, c_q)")
    fig.tight_layout(); fig.savefig(out / "figures" / "02_gravity_color_heatmap.png", dpi=150); plt.close(fig)

    # 8. Cosine histogram, 30 bins, edges -1+2k/30
    edges = [-1 + 2 * k / 30 for k in range(31)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cos_vals, bins=edges, density=True, alpha=0.6, label=f"780 gravity-gravity pairs (mean={A:.3f})", color="#a5581e")
    ax.hist(matched_vals, bins=edges, density=True, alpha=0.6, label=f"20 matched gravity-color pairs (mean={np.mean(matched_vals):.3f})", color="#1e5ba5")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("density (normalized separately, unequal n)")
    ax.set_title("Gravity-gravity vs. matched gravity-color cosine distributions")
    ax.legend(); ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.tight_layout(); fig.savefig(out / "figures" / "03_cosine_histogram.png", dpi=150); plt.close(fig)

    # 9. Matched contrast plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = list(range(len(Q)))
    vals = [delta[q] for q in Q]
    colors = ["#2e8b57" if v > 0 else "#c0392b" for v in vals]
    ax.bar(xs, vals, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(xs); ax.set_xticklabels(Q, fontsize=8)
    ax.set_xlabel("color-control scene ID"); ax.set_ylabel("delta_k = a_s - m_k")
    ax.set_title(f"H2 within-scene contrast per scene (D = mean = {D:.4f})")
    fig.tight_layout(); fig.savefig(out / "figures" / "04_matched_contrast.png", dpi=150); plt.close(fig)

    # 10. Transport plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(U))
    w_ = 0.35
    ax.bar(xs - w_ / 2, [r["e0"] for r in rows], width=w_, label="e0 (no-transport baseline)", color="#c0392b")
    ax.bar(xs + w_ / 2, [r["e1"] for r in rows], width=w_, label="e1 (transported prediction)", color="#2e8b57")
    ax.set_xticks(xs); ax.set_xticklabels(U, fontsize=9)
    ax.set_xlabel("held-out test scene ID"); ax.set_ylabel("Euclidean error")
    ax.set_title(f"H3 held-out transport error, per scene (aggregate R = RMSE1/RMSE0 = {R:.4f})")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "figures" / "05_transport.png", dpi=150); plt.close(fig)

    print("A =", A)
    print("D =", D)
    print("R =", R, " f =", f_improved, " H =", H)
    print("Wrote tables to", out / "tables")
    print("Wrote figures to", out / "figures")


if __name__ == "__main__":
    main()
