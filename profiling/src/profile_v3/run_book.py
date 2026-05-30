"""Book content profiling with Qwen3.5-4B (transformers, batched, resumable).

Per book: title + description + language_code -> LLM -> {language, tags}. The LLM
normalizes the given language_code, or INFERS the language from title/description text
when the code is missing (never from reviews). No reviews are used.

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=0 python run_book.py --shard 0 --nshards 2
  python run_book.py --merge
"""
from __future__ import annotations
import argparse, json, time
import pandas as pd

import config as C
import llm
from prompts import BOOK_SYSTEM

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def shard_path(k, n): return C.OUT / f"book_profiles.shard{k}of{n}.jsonl"


def render(tok, row):
    lc = (row["language_code"] or "").strip() or "missing"
    desc = (row["description"] or "").strip() or "(no description)"
    user = f"Title: {row['title']}\nLanguage code: {lc}\nDescription: {desc}"
    return llm.chat(tok, BOOK_SYSTEM, user)


def run(shard, nshards, batch, limit):
    df = pd.read_parquet(C.BOOK_INPUTS).iloc[shard::nshards].reset_index(drop=True)
    if limit: df = df.iloc[:limit]
    out_path = shard_path(shard, nshards)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try: done.add(json.loads(line)["iid"])
            except Exception: pass
    df = df[~df["iid"].isin(done)].reset_index(drop=True)
    log(f"book shard {shard}/{nshards}: {len(df):,} to do ({len(done):,} done)")
    if len(df) == 0: return
    tok, model = llm.load(C.LLM_MODEL, C.LLM_DTYPE); log("model loaded")
    n = e = 0
    with open(out_path, "a", buffering=1) as f:
        for s in range(0, len(df), batch):
            bdf = df.iloc[s:s + batch]
            prompts = [render(tok, row) for _, row in bdf.iterrows()]
            dec = llm.generate(tok, model, prompts, C.MAX_NEW_TOKENS, C.MAX_INPUT_TOKENS)
            for (_, row), txt in zip(bdf.iterrows(), dec):
                rec = {"iid": int(row["iid"])}
                try:
                    o = llm.parse_json(txt)
                    lg = str(o.get("language", "")).strip().lower()[:5] or "unknown"
                    rec["language"] = "zh" if lg.startswith("zh") else lg
                    rec["tags"] = llm.clean_tags(o.get("tags"), 10)
                except Exception as ex:
                    e += 1; rec["_error"] = str(ex); rec["_raw"] = (txt or "")[:200]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            el = time.time() - t0; rate = n / el if el else 0
            log(f"  {n}/{len(df)} ({rate:.1f}/s, ETA {(len(df)-n)/rate/60:.1f}m, {e} err)")
    log(f"book shard {shard} done: {n} ({e} err)")


def merge():
    seen = {}
    for p in sorted(C.OUT.glob("book_profiles.shard*of*.jsonl")):
        for line in open(p):
            try: r = json.loads(line); seen[r["iid"]] = r
            except Exception: pass
    with open(C.BOOK_PROFILES, "w") as f:
        for iid in sorted(seen): f.write(json.dumps(seen[iid], ensure_ascii=False) + "\n")
    ok = sum(1 for r in seen.values() if "tags" in r)
    log(f"merged {len(seen):,} books ({ok:,} ok) -> {C.BOOK_PROFILES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=C.NSHARDS)
    ap.add_argument("--batch", type=int, default=C.BATCH_BOOK); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run(a.shard, a.nshards, a.batch, a.limit)
