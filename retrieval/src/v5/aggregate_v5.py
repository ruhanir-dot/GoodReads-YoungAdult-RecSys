"""Aggregate the v5 sweep eval_*.json into comparison tables (mean +/- std over seeds) for
REPORT_v5.md: warm ablation (all + user/item tiers + per-language) and the lever-A cold-item
holdout. Prints markdown-ish tables and dumps a v5_summary.json.

  python aggregate_v5.py
"""
from __future__ import annotations
import json
import numpy as np
import config as C

WARM = ["A0", "A1", "A2", "B1", "C1", "C2", "D"]
SEEDS = [42, 43, 44]
KEYG = ["all", "u_cold", "u_warm", "u_hot", "i_head", "i_mid", "i_tail"]


def load(tag, d=None):
    p = (d or C.SWEEP_DIR) / f"eval_{tag}.json"
    return json.load(open(p)) if p.exists() else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))


def main():
    summary = {"warm": {}, "tiers": {}, "per_lang": {}, "cold": {}}
    print("=" * 96)
    print("v5 WARM ABLATION — Recall@100 / NDCG@10 (mean+/-std over seeds 42/43/44), base = A0")
    print("=" * 96)
    print(f"{'variant':6s} {'lever':28s} {'R@100':>16s} {'NDCG@10':>16s} {'Cov@200':>9s}  vs A0")
    desc = {"A0": "atomic base (=v4)", "A1": "semantic-only", "A2": "hybrid id",
            "B1": "+dislike hard-neg", "C1": "+user-lang feat", "C2": "+lang match (retr)",
            "D": "A2+B1+C2 (delivered)"}
    a0r = ms([load(f"A0_s{s}")["dnn"]["all"]["Recall@100"] for s in SEEDS if load(f"A0_s{s}")])[0]
    for v in WARM:
        runs = [load(f"{v}_s{s}") for s in SEEDS]
        if not any(runs): continue
        r = ms([x["dnn"]["all"]["Recall@100"] for x in runs if x])
        n = ms([x["dnn"]["all"]["NDCG@10"] for x in runs if x])
        cov = ms([x["dnn"]["coverage"]["Coverage@200"] for x in runs if x])
        summary["warm"][v] = {"R@100": r, "NDCG@10": n, "Cov@200": cov, "n_seeds": sum(x is not None for x in runs)}
        print(f"{v:6s} {desc[v]:28s} {r[0]:.4f}+/-{r[1]:.4f} {n[0]:8.4f}+/-{n[1]:.4f} {cov[0]:8.3f}  {r[0]-a0r:+.4f}")

    print("\n" + "=" * 96)
    print("TAGS-AS-CONTENT ABLATION — was the v3 tag signal ever tested as a content feature? (base A0; D too)")
    print("=" * 96)
    print(f"{'run':14s} {'content':8s} {'R@100':>16s} {'i_tail R@100':>13s} {'i_mid R@100':>12s}")
    content_runs = [("A0", "desc", "A0"), ("A0", "tags", "A0_tags"), ("A0", "both", "A0_both"),
                    ("D", "desc", "D"), ("D", "both", "D_both")]
    summary["content"] = {}
    for v, c, tagbase in content_runs:
        runs = [load(f"{tagbase}_s{s}") for s in SEEDS]
        if not any(runs): continue
        r = ms([x["dnn"]["all"]["Recall@100"] for x in runs if x])
        it = ms([x["dnn"]["i_tail"]["Recall@100"] for x in runs if x])
        im = ms([x["dnn"]["i_mid"]["Recall@100"] for x in runs if x])
        summary["content"][tagbase] = {"content": c, "R@100": r, "i_tail": it, "i_mid": im}
        print(f"{tagbase:14s} {c:8s} {r[0]:.4f}+/-{r[1]:.4f} {it[0]:11.4f}   {im[0]:10.4f}")

    print("\n" + "=" * 96)
    print("PER-TIER R@100 (mean over seeds) — did any lever move the v4 structural shortfalls?")
    print("=" * 96)
    print(f"{'variant':6s} " + " ".join(f"{g:>9s}" for g in KEYG))
    for v in WARM:
        runs = [load(f"{v}_s{s}") for s in SEEDS]
        if not any(runs): continue
        row = {g: ms([x["dnn"][g]["Recall@100"] for x in runs if x])[0] for g in KEYG}
        summary["tiers"][v] = row
        print(f"{v:6s} " + " ".join(f"{row[g]:9.4f}" for g in KEYG))

    print("\n" + "=" * 96)
    print("PER-LANGUAGE R@100 (mean over seeds) — lever C target (non-en tail). n = test targets in that lang")
    print("=" * 96)
    langs = [f"L_{l}" for l in C.PER_LANG] + ["L_other"]
    a0 = [load(f"A0_s{s}") for s in SEEDS]
    present = [l for l in langs if any(x and x["dnn"][l]["n"] > 0 for x in a0)]
    print(f"{'variant':6s} " + " ".join(f"{l[2:]:>8s}" for l in present))
    for v in ["A0", "C1", "C2", "D"]:
        runs = [load(f"{v}_s{s}") for s in SEEDS]
        if not any(runs): continue
        row = {l: ms([x["dnn"][l]["Recall@100"] for x in runs if x])[0] for l in present}
        summary["per_lang"][v] = row
        print(f"{v:6s} " + " ".join(f"{row[l]:8.3f}" for l in present))
    n_by_lang = {l[2:]: (a0[0]["dnn"][l]["n"] if a0[0] else 0) for l in present}
    print("n=    " + " ".join(f"{n_by_lang[l[2:]]:8d}" for l in present))

    print("\n" + "=" * 96)
    print("LEVER-A COLD-ITEM HOLDOUT — semantic vs atomic on items with NO training signal")
    print("=" * 96)
    print(f"{'variant':6s} {'item_warm R@100':>16s} {'item_cold R@100':>16s} {'ColdCov@200':>12s}")
    for v in ["A0", "A1", "A2"]:
        x = load(f"cold_{v}_s42", C.COLD_DIR)
        if not x: continue
        d = x["dnn"]
        summary["cold"][v] = {"item_warm_R100": d["item_warm"]["Recall@100"],
                              "item_cold_R100": d["item_cold"]["Recall@100"],
                              "cold_cov200": d["cold_item_coverage@200"]}
        print(f"{v:6s} {d['item_warm']['Recall@100']:16.4f} {d['item_cold']['Recall@100']:16.4f} {d['cold_item_coverage@200']:12.4f}")

    json.dump(summary, open(C.V5_DIR / "v5_summary.json", "w"), indent=2)
    print(f"\nwrote {C.V5_DIR/'v5_summary.json'}")


if __name__ == "__main__":
    main()
