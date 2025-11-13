from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

POSSIBLE_CATEGORICAL_COLUMNS = [
    "protocol",
    "proto",
    "service",
    "flag",
    "state",
    "attack_cat",
]


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def load_csv_dataset(
    csv_path: Path, category_levels: Dict[str, List[str]] | None = None
) -> TensorDataset:
    """Load a CSV, one-hot encode categorical columns, and return tensors."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = _drop_unnamed(df)
    if "label" not in df.columns:
        raise ValueError(f"`label` column missing in {csv_path}")

    categorical_columns = [col for col in POSSIBLE_CATEGORICAL_COLUMNS if col in df.columns]
    numeric_cols = [
        column for column in df.columns if column not in set(categorical_columns + ["label"])
    ]

    effective_levels: Dict[str, List[str]] = {}
    for column in categorical_columns:
        if category_levels and column in category_levels:
            effective_levels[column] = category_levels[column]
        else:
            effective_levels[column] = sorted(df[column].astype(str).dropna().unique().tolist())
        df[column] = pd.Categorical(df[column], categories=effective_levels[column])

    if categorical_columns:
        cat_df = pd.get_dummies(
            df[categorical_columns], columns=categorical_columns, dtype="float32"
        )
        feature_df = pd.concat(
            [df[numeric_cols].astype("float32"), cat_df.reset_index(drop=True)], axis=1
        )
    else:
        feature_df = df[numeric_cols].astype("float32")

    for column in feature_df.columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce").fillna(0)
    feature_df.replace([np.inf, -np.inf], 0, inplace=True)

    features = torch.tensor(feature_df.to_numpy(dtype="float32"), dtype=torch.float32)
    labels = torch.tensor(df["label"].to_numpy(dtype="int64"), dtype=torch.long)
    return TensorDataset(features, labels)


def infer_category_levels(csv_paths: Iterable[Path]) -> Dict[str, List[str]]:
    """Scan CSV files to gather all categorical levels per column."""
    levels = {column: set() for column in POSSIBLE_CATEGORICAL_COLUMNS}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = _drop_unnamed(df)
        for column in POSSIBLE_CATEGORICAL_COLUMNS:
            if column in df.columns:
                levels[column].update(df[column].astype(str).dropna().unique().tolist())
    return {column: sorted(level_set) for column, level_set in levels.items() if level_set}
