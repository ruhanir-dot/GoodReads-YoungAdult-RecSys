"""User taste profiling with Qwen3.5-4B (transformers, batched, resumable).

Per user: tokenizer-truncate the concatenated recent train reviews to REVIEW_TOKEN_BUDGET,
LLM -> {like, dislike}. The `language` dict is the CODE-computed langdetect frequency
(from build_inputs), NOT produced by the LLM.

  CUDA_MPS_PIPE_DIRECTORY="" CUDA_VISIBLE_DEVICES=0 python run_user.py --shard 0 --nshards 2
  python run_user.py --merge
"""
from __future__ import annotations
import argparse, json, time
import pandas as pd

import config as C
import llm
from prompts import USER_SYSTEM

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


def shard_path(k, n): return C.OUT / f"user_profiles.shard{k}of{n}.jsonl"


def render(tok, reviews_json):
    revs = json.loads(reviews_json)
    lines = []
    for r in revs:
        tag = "[-]" if r["neg"] else "[+]"
        lines.append(f'- {tag} "{r["title"]}" ({r["rating"]}/5): {r["text"]}')
    text = "\n".join(lines)
    ids = tok(text, add_special_tokens=False)["input_ids"]      # tokenizer-truncate concat
    if len(ids) > C.REVIEW_TOKEN_BUDGET:
        text = tok.decode(ids[:C.REVIEW_TOKEN_BUDGET])
    return llm.chat(tok, USER_SYSTEM, f"A reader's recent reviews ([+]=liked, [-]=disliked):\n\n{text}")


def run(shard, nshards, batch, limit):
    df = pd.read_parquet(C.USER_INPUTS).iloc[shard::nshards].reset_index(drop=True)
    if limit: df = df.iloc[:limit]
    out_path = shard_path(shard, nshards)
    done = set()
    if out_path.exists():
        for line in open(out_path):
            try: done.add(json.loads(line)["uid"])
            except Exception: pass
    df = df[~df["uid"].isin(done)].reset_index(drop=True)
    log(f"user shard {shard}/{nshards}: {len(df):,} to do ({len(done):,} done)")
    if len(df) == 0: return
    tok, model = llm.load(C.LLM_MODEL, C.LLM_DTYPE); log("model loaded")
    n = e = 0
    with open(out_path, "a", buffering=1) as f:
        for s in range(0, len(df), batch):
            bdf = df.iloc[s:s + batch]
            prompts = [render(tok, r) for r in bdf["reviews_json"]]
            dec = llm.generate(tok, model, prompts, C.MAX_NEW_TOKENS, C.MAX_INPUT_TOKENS)
            for (_, row), txt in zip(bdf.iterrows(), dec):
                rec = {"uid": int(row["uid"]), "language": json.loads(row["language_json"])}
                try:
                    o = llm.parse_json(txt)
                    rec["like"] = llm.clean_tags(o.get("like"), 10)
                    rec["dislike"] = llm.clean_tags(o.get("dislike"), 6)
                except Exception as ex:
                    e += 1; rec["_error"] = str(ex); rec["_raw"] = (txt or "")[:200]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            el = time.time() - t0; rate = n / el if el else 0
            log(f"  {n}/{len(df)} ({rate:.1f}/s, ETA {(len(df)-n)/rate/60:.1f}m, {e} err)")
    log(f"user shard {shard} done: {n} ({e} err)")


def merge():
    seen = {}
    for p in sorted(C.OUT.glob("user_profiles.shard*of*.jsonl")):
        for line in open(p):
            try: r = json.loads(line); seen[r["uid"]] = r
            except Exception: pass
    with open(C.USER_PROFILES, "w") as f:
        for uid in sorted(seen): f.write(json.dumps(seen[uid], ensure_ascii=False) + "\n")
    ok = sum(1 for r in seen.values() if "like" in r)
    log(f"merged {len(seen):,} users ({ok:,} ok) -> {C.USER_PROFILES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=C.NSHARDS)
    ap.add_argument("--batch", type=int, default=C.BATCH_USER); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run(a.shard, a.nshards, a.batch, a.limit)
