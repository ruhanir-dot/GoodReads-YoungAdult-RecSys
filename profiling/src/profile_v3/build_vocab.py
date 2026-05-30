"""Global vocab post-process (run AFTER full profiling).

Counts phrase frequency across book.tags + user.like + user.dislike (case-insensitive),
keeps the Top VOCAB_TOP phrases as the standard discrete vocab, and drops the long tail.
Writes vocab.json {phrase2id, id2phrase, freq, scope, coverage}.
"""
from __future__ import annotations
import json
from collections import Counter

import config as C


def _iter_phrases():
    if C.BOOK_PROFILES.exists():
        for line in open(C.BOOK_PROFILES):
            try: r = json.loads(line)
            except Exception: continue
            for t in r.get("tags", []) or []:
                yield t
    if C.USER_PROFILES.exists():
        for line in open(C.USER_PROFILES):
            try: r = json.loads(line)
            except Exception: continue
            for f in ("like", "dislike"):
                for t in r.get(f, []) or []:
                    yield t


def main():
    cnt = Counter()       # lowercase-key -> count
    canon = {}            # lowercase-key -> most-common original casing
    case_cnt = Counter()
    total = 0
    for t in _iter_phrases():
        t = t.strip()
        if not t:
            continue
        k = t.lower(); cnt[k] += 1; total += 1
        case_cnt[(k, t)] += 1
    # pick canonical casing = most frequent surface form
    for (k, surf), c in case_cnt.items():
        if k not in canon or c > case_cnt[(k, canon[k])]:
            canon[k] = surf
    top = cnt.most_common(C.VOCAB_TOP)
    phrase2id = {canon[k]: i for i, (k, _) in enumerate(top)}
    id2phrase = [canon[k] for k, _ in top]
    freq = {canon[k]: c for k, c in top}
    covered = sum(c for _, c in top)
    vocab = {"n_tags": len(top), "scope": "book.tags + user.like + user.dislike",
             "total_occurrences": total, "distinct_phrases": len(cnt),
             "coverage": round(covered / max(total, 1), 4),
             "phrase2id": phrase2id, "id2phrase": id2phrase, "freq": freq}
    json.dump(vocab, open(C.VOCAB, "w"), ensure_ascii=False, indent=1)
    print(f"vocab: {len(top):,} phrases (top-{C.VOCAB_TOP}) out of {len(cnt):,} distinct; "
          f"covers {vocab['coverage']:.1%} of {total:,} occurrences -> {C.VOCAB}")
    print("  sample top-20:", [p for p in id2phrase[:20]])


if __name__ == "__main__":
    main()
