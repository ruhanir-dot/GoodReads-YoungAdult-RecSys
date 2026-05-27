"""Prediction dump/load with a fixed schema.

Every model writes `predictions/{name}_topk.parquet` with the columns
(user_id:int32, item_id:int32, rank:int32, score:float32). Notebook 08
reads this and nothing else — so if a notebook saves under a different
schema, the comparison breaks loudly here instead of silently in 08.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PRED_COLUMNS = ("user_id", "item_id", "rank", "score")


def _default_predictions_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "predictions"


def _validate(pred_df: pd.DataFrame, allow_extra: bool = True) -> pd.DataFrame:
    missing = [c for c in PRED_COLUMNS if c not in pred_df.columns]
    if missing:
        raise ValueError(
            f"Predictions DataFrame is missing required columns {missing}. "
            f"Required schema: {PRED_COLUMNS}. Got: {list(pred_df.columns)}"
        )
    out = pred_df.copy()
    out["user_id"] = out["user_id"].astype(np.int32)
    out["item_id"] = out["item_id"].astype(np.int32)
    out["rank"] = out["rank"].astype(np.int32)
    out["score"] = out["score"].astype(np.float32)
    if not allow_extra:
        out = out[list(PRED_COLUMNS)]
    # sanity: rank should start at 0 per user
    bad = out.groupby("user_id")["rank"].min()
    if (bad > 0).any():
        n_bad = int((bad > 0).sum())
        raise ValueError(
            f"{n_bad} users have rank not starting at 0. "
            f"Convention: rank is 0-indexed (rank=0 is the top recommendation)."
        )
    return out


def save_predictions(
    pred_df: pd.DataFrame,
    name: str,
    predictions_dir: Optional[str | Path] = None,
    *,
    allow_extra_columns: bool = True,
) -> Path:
    """Write `predictions/{name}_topk.parquet`. Returns the output path.

    Extra columns (e.g. ALS's `rating_pred`) are preserved by default so
    rating_metrics() can find them.
    """
    out_df = _validate(pred_df, allow_extra=allow_extra_columns)
    pdir = Path(predictions_dir) if predictions_dir is not None else _default_predictions_dir()
    pdir.mkdir(parents=True, exist_ok=True)
    out_path = pdir / f"{name}_topk.parquet"
    out_df.to_parquet(out_path, index=False)
    return out_path


def load_predictions(
    name: str,
    predictions_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Read `predictions/{name}_topk.parquet`
    """
    pdir = Path(predictions_dir) if predictions_dir is not None else _default_predictions_dir()
    path = pdir / f"{name}_topk.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No predictions found at {path}")
    return pd.read_parquet(path)


def list_predictions(predictions_dir: Optional[str | Path] = None) -> list[str]:
    """
    Return the model names that currently have predictions on disk
    """
    pdir = Path(predictions_dir) if predictions_dir is not None else _default_predictions_dir()
    if not pdir.exists():
        return []
    return sorted(p.stem.removesuffix("_topk") for p in pdir.glob("*_topk.parquet"))
