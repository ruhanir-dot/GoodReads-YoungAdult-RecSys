"""Thin Qwen3.5-4B inference helper (transformers, batched, greedy)."""
from __future__ import annotations
import json, re


def load(model_name, dtype):
    import torch
    from transformers import AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                       # required for batched .generate()
    common = dict(dtype=getattr(torch, dtype), device_map={"": 0})
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_name, attn_implementation="sdpa", **common)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **common)
    model.eval()
    return tok, model


def chat(tok, system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(tok, model, prompts, max_new, max_in):
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_in).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             repetition_penalty=1.2,          # kills greedy phrase-repeat loops; nudges list to close
                             pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


_JSON = re.compile(r"\{[\s\S]*\}")
_ARR = lambda key: re.compile(rf'"{key}"\s*:\s*\[(.*?)(?:\]|$)', re.S)   # tolerant of an unclosed list
_STR = lambda key: re.compile(rf'"{key}"\s*:\s*"([^"]*)"')
_ITEMS = re.compile(r'"([^"]*)"')

def parse_json(s):
    """Strict JSON first; on failure, salvage the (possibly truncated/looped) fields by
    pulling complete quoted items out of each array -> recovers most cut-off outputs."""
    s = s or ""
    m = _JSON.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    out = {}
    for key in ("like", "dislike", "tags"):           # arrays of complete quoted strings
        am = _ARR(key).search(s)
        if am:
            out[key] = _ITEMS.findall(am.group(1))
    lm = _STR("language").search(s)                    # book language string
    if lm:
        out["language"] = lm.group(1)
    if out:
        return out
    raise ValueError("no JSON object")


def clean_tags(x, n=12):
    """list[str] -> deduped, stripped, title-cased-ish short phrases (cap n)."""
    if not isinstance(x, list):
        return []
    out, seen = [], set()
    for t in x:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if 1 < len(t) <= 60 and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
        if len(out) >= n:
            break
    return out
