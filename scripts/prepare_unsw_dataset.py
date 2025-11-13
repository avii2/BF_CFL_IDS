from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed columns that can appear when CSVs contain BOMs."""
    unnamed = [col for col in df.columns if col.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def split_train_normals(
    df: pd.DataFrame, clients: int, output_dir: Path, seed: int
) -> List[Path]:
    normals = df[df["label"] == 0].reset_index(drop=True)
    if normals.empty:
        raise ValueError("No normal samples (label==0) found in training CSV.")
    normals = normals.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    shards = np.array_split(normals, clients)
    client_paths: List[Path] = []
    train_dir = output_dir / "train_normal"
    train_dir.mkdir(parents=True, exist_ok=True)

    for idx, shard in enumerate(shards):
        if shard.empty:
            continue
        client_path = train_dir / f"client_{idx:02d}.csv"
        shard.to_csv(client_path, index=False)
        client_paths.append(client_path)
    return client_paths


def split_mixed_sets(
    df: pd.DataFrame,
    output_dir: Path,
    server_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[Path, Path, Path]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = len(shuffled)
    server_count = max(1, int(total * server_ratio))
    remaining = shuffled.iloc[server_count:].reset_index(drop=True)
    val_count = max(1, int(len(remaining) * val_ratio))

    server_df = shuffled.iloc[:server_count].reset_index(drop=True)
    val_df = remaining.iloc[:val_count].reset_index(drop=True)
    test_df = remaining.iloc[val_count:].reset_index(drop=True)

    server_dir = output_dir / "server_mixed"
    val_dir = output_dir / "val_mixed"
    test_dir = output_dir / "test_mixed"
    server_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    server_path = server_dir / "train.csv"
    val_path = val_dir / "val.csv"
    test_path = test_dir / "test.csv"

    server_df.to_csv(server_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    return server_path, val_path, test_path


def write_cluster_map(client_paths: List[Path], clusters: int, output_dir: Path) -> None:
    mapping = {
        client_path.stem: idx % clusters for idx, client_path in enumerate(client_paths)
    }
    (output_dir / "cluster_map.json").write_text(json.dumps(mapping, indent=2))


def write_dataset_readme(output_dir: Path, source_train: Path, source_test: Path) -> None:
    contents = f"""# UNSW-NB15 derived dataset

* Training source: `{source_train.name}`
* Testing source: `{source_test.name}`
* Clients contain only label==0 samples (normal traffic).
* Validation/Test splits keep the original label distribution. Validation/Test/Server sets come from a random split of `{source_test.name}`.
"""
    (output_dir / "README_dataset.md").write_text(contents)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare UNSW-NB15 dataset for clustered FL demo."
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("network_anomaly_dataset"))
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--server-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = sanitize(pd.read_csv(args.train_csv, encoding="utf-8-sig"))
    test_df = sanitize(pd.read_csv(args.test_csv, encoding="utf-8-sig"))

    client_paths = split_train_normals(train_df, args.clients, output_dir, args.seed)
    if not client_paths:
        raise RuntimeError("No training clients were generated.")

    split_mixed_sets(test_df, output_dir, args.server_ratio, args.val_ratio, args.seed)
    write_cluster_map(client_paths, args.clusters, output_dir)
    write_dataset_readme(output_dir, args.train_csv, args.test_csv)
    print(
        f"Wrote {len(client_paths)} clients plus mixed validation/test splits under {output_dir}"
    )


if __name__ == "__main__":
    main()
