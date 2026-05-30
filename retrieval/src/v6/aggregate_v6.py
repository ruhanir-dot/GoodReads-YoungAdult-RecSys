"""Aggregate v6 sweep -> comparison tables for REPORT_v6: semantic-ID source comparison
(M0/M1/M2/M3), graded-negative ablation (M0 x N0/N1/N2), per-tier, per-language, cold-item.

  python aggregate_v6.py
"""
from __future__ import annotations
import json
import numpy as np
import config as C

SEEDS = [42, 43, 44]
UG = ["all", "u_cold", "u_warm", "u_hot"]
IG = ["i_head", "i_mid", "i_tail"]


def load(tag, d=None):
    p = (d or C.SWEEP_DIR) / f"eval_{tag}.json"
    return json.load(open(p)) if p.exists() else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))


def main():
    summary = {}
    print("=" * 100)
    print("v6 SEMANTIC-ID SOURCE COMPARISON — set Recall@100 / NDCG@10 (mean+/-std seeds), neg=N2. base=M0")
    print("=" * 100)
    desc = {"M0": "atomic (no semantic)", "M1": "hybrid + CONTENT codes", "M2": "hybrid + COLLAB codes",
            "M3": "hybrid + CONTENT-big(K1024/L4)"}
    m0r = ms([load(f"M0_N2_s{s}")["dnn"]["all"]["Recall@100"] for s in SEEDS if load(f"M0_N2_s{s}")])[0]
    print(f"{'model':5s} {'source':30s} {'R@100':>16s} {'NDCG@10':>16s} {'i_tail':>8s} {'i_mid':>8s}  vsM0")
    summary["semantic"] = {}
    for m in ["M0", "M1", "M2", "M3"]:
        runs = [load(f"{m}_N2_s{s}") for s in SEEDS]
        if not any(runs): continue
        r = ms([x["dnn"]["all"]["Recall@100"] for x in runs if x]); nd = ms([x["dnn"]["all"]["NDCG@10"] for x in runs if x])
        it = ms([x["dnn"]["i_tail"]["Recall@100"] for x in runs if x])[0]; im = ms([x["dnn"]["i_mid"]["Recall@100"] for x in runs if x])[0]
        summary["semantic"][m] = {"R@100": r, "NDCG@10": nd, "i_tail": it, "i_mid": im}
        print(f"{m:5s} {desc[m]:30s} {r[0]:.4f}+/-{r[1]:.4f} {nd[0]:8.4f}+/-{nd[1]:.4f} {it:8.4f} {im:8.4f}  {r[0]-m0r:+.4f}")

    print("\n" + "=" * 100)
    print("GRADED-NEGATIVE ABLATION (model M0) — set Recall@100 / NDCG@10")
    print("=" * 100)
    nd_desc = {"N0": "in-batch only", "N1": "+hard(rating<=2)", "N2": "+hard+soft(rating==3)"}
    print(f"{'neg':4s} {'scheme':26s} {'R@100':>16s} {'NDCG@10':>16s} {'i_tail':>8s}")
    summary["negatives"] = {}
    for nn in ["N0", "N1", "N2"]:
        runs = [load(f"M0_{nn}_s{s}") for s in SEEDS]
        if not any(runs): continue
        r = ms([x["dnn"]["all"]["Recall@100"] for x in runs if x]); nd = ms([x["dnn"]["all"]["NDCG@10"] for x in runs if x])
        it = ms([x["dnn"]["i_tail"]["Recall@100"] for x in runs if x])[0]
        summary["negatives"][nn] = {"R@100": r, "NDCG@10": nd, "i_tail": it}
        print(f"{nn:4s} {nd_desc[nn]:26s} {r[0]:.4f}+/-{r[1]:.4f} {nd[0]:8.4f}+/-{nd[1]:.4f} {it:8.4f}")

    print("\n" + "=" * 100)
    print("PER-TIER set Recall@100 (mean over seeds, neg=N2)")
    print("=" * 100)
    print(f"{'model':5s} " + " ".join(f"{g:>9s}" for g in UG + IG))
    for m in ["M0", "M1", "M2", "M3"]:
        runs = [load(f"{m}_N2_s{s}") for s in SEEDS]
        if not any(runs): continue
        row = {g: ms([x["dnn"][g]["Recall@100"] for x in runs if x])[0] for g in UG + IG}
        print(f"{m:5s} " + " ".join(f"{row[g]:9.4f}" for g in UG + IG))

    print("\n" + "=" * 100)
    print("PER-LANGUAGE set Recall@100 (mean over seeds, neg=N2)")
    print("=" * 100)
    langs = list(C.PER_LANG) + ["other"]
    a0 = [load(f"M0_N2_s{s}") for s in SEEDS]
    present = [l for l in langs if any(x and x["dnn"][f"L_{l}"]["n"] > 0 for x in a0)]
    print(f"{'model':5s} " + " ".join(f"{l:>7s}" for l in present))
    for m in ["M0", "M1", "M2", "M3"]:
        runs = [load(f"{m}_N2_s{s}") for s in SEEDS]
        if not any(runs): continue
        print(f"{m:5s} " + " ".join(f"{ms([x['dnn']['L_'+l]['Recall@100'] for x in runs if x])[0]:7.3f}" for l in present))
    print("n=   " + " ".join(f"{a0[0]['dnn']['L_'+l]['n']:7d}" for l in present))

    print("\n" + "=" * 100)
    print("COLD-ITEM HOLDOUT — content vs collab semantic codes on items with NO training signal")
    print("=" * 100)
    print(f"{'model':5s} {'source':14s} {'item_warm R@100':>16s} {'item_cold R@100':>16s} {'ColdCov@200':>12s}")
    cd = {"M0": "atomic", "M1": "content", "M2": "collab"}
    summary["cold"] = {}
    for m in ["M0", "M1", "M2"]:
        x = load(f"cold_{m}_s42", C.COLD_DIR)
        if not x: continue
        d = x["dnn"]; summary["cold"][m] = {"warm": d["item_warm"]["Recall@100"], "cold": d["item_cold"]["Recall@100"], "cov": d["cold_item_coverage@200"]}
        print(f"{m:5s} {cd[m]:14s} {d['item_warm']['Recall@100']:16.4f} {d['item_cold']['Recall@100']:16.4f} {d['cold_item_coverage@200']:12.4f}")

    json.dump(summary, open(C.V6_DIR / "v6_summary.json", "w"), indent=2)
    print(f"\nwrote {C.V6_DIR/'v6_summary.json'}")


if __name__ == "__main__":
    main()
