"""H1/H2/H3 analysis exactly per 02_MATHEMATICAL_PLAN.md. Works on whatever subset of
scenes has embeddings available (so it doubles as the dummy-run smoke test and the
real 100-clip analysis) -- reports which parts of T/U/Q are actually present rather
than assuming the full design is loaded.
Usage: python analyze.py [--embeddings PATH] [--plan PATH] [--out PATH]"""
import argparse
import json
from pathlib import Path

import numpy as np

EPS = 1e-12


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return None  # undefined per protocol
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", type=str, default="/workspace/gravity/embeddings.json")
    ap.add_argument("--plan", type=str, default="/workspace/gravity/scene_plan.json")
    ap.add_argument("--out", type=str, default="/workspace/gravity/analysis_summary.json")
    args = ap.parse_args()

    emb = json.loads(Path(args.embeddings).read_text())
    plan = json.loads(Path(args.plan).read_text())
    T_full = set(plan["train_scene_ids"])
    U_full = set(plan["test_scene_ids"])
    Q_full = set(plan["color_control_scene_ids"])

    by_scene = {}
    for name, row in emb.items():
        by_scene.setdefault(row["scene"], {})[row["condition"]] = np.array(row["embedding"], dtype=np.float64)

    g = {}   # scene -> gravity vector
    c = {}   # scene -> color vector
    for s, conds in by_scene.items():
        if "g98" in conds and "g49" in conds:
            g[s] = conds["g49"] - conds["g98"]
        if "g98" in conds and "color" in conds:
            c[s] = conds["color"] - conds["g98"]

    S = sorted(g.keys())
    Q = sorted(set(c.keys()) & set(S))
    T = sorted(T_full & set(S))
    U = sorted(U_full & set(S))

    print(f"Scenes with a gravity pair available: {len(S)} -> {S}")
    print(f"Of those, in official color set Q and have a color clip: {len(Q)} -> {Q}")
    print(f"Of those, in official train set T: {len(T)} -> {T}  "
          f"(full design has {len(T_full)})")
    print(f"Of those, in official test set U: {len(U)} -> {U}  "
          f"(full design has {len(U_full)})")

    summary = {"n_scenes_with_gravity_pair": len(S), "scenes": S,
               "n_color_available": len(Q), "n_train_available": len(T), "n_test_available": len(U)}

    # ---- H1 ----
    pairs = [(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]
    cos_vals = []
    undefined_norm_scenes = [s for s in S if np.linalg.norm(g[s]) <= EPS]
    for s, t in pairs:
        v = cos(g[s], g[t])
        if v is not None:
            cos_vals.append(v)
    a_s = {}
    for s in S:
        others = [t for t in S if t != s]
        vals = [cos(g[s], g[t]) for t in others]
        vals = [v for v in vals if v is not None]
        a_s[s] = float(np.mean(vals)) if vals else None

    if cos_vals:
        A = float(np.mean(cos_vals))
        summary["H1"] = dict(
            A=A, n_pairs=len(cos_vals), expected_full_n_pairs=780,
            mean=A, median=float(np.median(cos_vals)),
            std_population=float(np.sqrt(np.mean((np.array(cos_vals) - A) ** 2))),
            min=float(np.min(cos_vals)), max=float(np.max(cos_vals)),
            undefined_norm_scenes=undefined_norm_scenes,
        )
        print(f"\n=== H1 === A={A:.6f} over {len(cos_vals)} pair(s) "
              f"(full design: 780). {'>>> supports H1 (A>0)' if A > 0 else '>>> does NOT support H1 (A<=0)'}")
    else:
        summary["H1"] = dict(error="fewer than 2 scenes with gravity pairs available")
        print("\n=== H1 === unevaluable: fewer than 2 scenes with gravity pairs available")

    # ---- H2 ----
    if Q:
        m = {}
        delta = {}
        cross = {}
        for k, qk in enumerate(Q):
            row = {}
            for s in S:
                v = cos(g[s], c[qk])
                row[s] = v
            cross[qk] = row
            m[qk] = row.get(qk)
            if m[qk] is not None and a_s.get(qk) is not None:
                delta[qk] = a_s[qk] - m[qk]
        d_vals = [v for v in delta.values() if v is not None]
        if d_vals:
            D = float(np.mean(d_vals))
            summary["H2"] = dict(
                D=D, n_matched=len(d_vals), expected_full_n_matched=20,
                matched_cosines={str(k): v for k, v in m.items()},
                contrasts={str(k): v for k, v in delta.items()},
            )
            print(f"=== H2 === D={D:.6f} over {len(d_vals)} matched scene(s) "
                  f"(full design: 20). {'>>> supports H2 (D>0)' if D > 0 else '>>> does NOT support H2 (D<=0)'}")
        else:
            summary["H2"] = dict(error="no scene has both a_s and matched color cosine available")
            print("=== H2 === unevaluable: need at least one scene with both a_s (needs >=2 gravity scenes) and its own color vector")
    else:
        summary["H2"] = dict(error="no color-control scenes available")
        print("=== H2 === unevaluable: no color-control scenes available")

    # ---- H3 ----
    if T and U:
        d_T = np.mean([g[s] for s in T], axis=0)
        norm_dT = np.linalg.norm(d_T)
        rows = []
        for s in U:
            e0 = float(np.linalg.norm(g[s]))
            e1 = float(np.linalg.norm(d_T - g[s]))
            r = e1 / e0 if e0 > EPS else None
            h = cos(d_T, g[s]) if norm_dT > EPS else None
            rows.append(dict(scene=s, e0=e0, e1=e1, r=r, h=h))
        e0s = np.array([row["e0"] for row in rows])
        e1s = np.array([row["e1"] for row in rows])
        RMSE0 = float(np.sqrt(np.mean(e0s ** 2)))
        RMSE1 = float(np.sqrt(np.mean(e1s ** 2)))
        R = RMSE1 / RMSE0 if RMSE0 > EPS else None
        f = float(np.mean([1.0 if row["e1"] < row["e0"] else 0.0 for row in rows]))
        h_vals = [row["h"] for row in rows if row["h"] is not None]
        H = float(np.mean(h_vals)) if h_vals else None
        summary["H3"] = dict(
            R=R, RMSE0=RMSE0, RMSE1=RMSE1, f=f, H=H, norm_dT=float(norm_dT),
            n_train_used=len(T), n_test_used=len(U),
            expected_full_n_train=30, expected_full_n_test=10,
            rows=rows,
        )
        if R is not None:
            verdict = ">>> supports H3 (R<1)" if R < 1 else ">>> does NOT support H3 (R>=1)"
        else:
            verdict = ">>> R undefined (RMSE0 ~ 0)"
        print(f"=== H3 === R={R} over {len(U)} test scene(s), {len(T)} train scene(s) "
              f"(full design: 30 train / 10 test). {verdict}")
        if len(T) < 30 or len(U) < 10:
            print("    NOTE: this is a partial/dummy run -- R here is not the real experimental result, "
                  "only a check that the transport code executes correctly end to end.")
    else:
        summary["H3"] = dict(error=f"need >=1 train and >=1 test scene available; have T={len(T)}, U={len(U)}")
        print(f"=== H3 === unevaluable: have {len(T)} train scene(s) and {len(U)} test scene(s) available, need >=1 of each")

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary written to {args.out}")


if __name__ == "__main__":
    main()
