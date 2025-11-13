from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, roc_curve

from ledger.logger import ModelLedger
from data_utils import infer_category_levels, load_csv_dataset
from fl.client import Client, ClientConfig
from fl.strategy import ClusteredFLStrategy, StrategyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clustered FL + model ledger demo.")
    parser.add_argument("--rounds", type=int, default=3, help="Training rounds.")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters.")
    parser.add_argument(
        "--clients-per-round", type=int, default=None, help="Sampled clients per round."
    )
    parser.add_argument(
        "--clients-per-cluster",
        type=int,
        default=None,
        help="Sample a fixed number of clients from each cluster assignment every round.",
    )
    parser.add_argument(
        "--client-epochs",
        type=int,
        default=2,
        help="Number of local epochs each client runs per round.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="network_anomaly_dataset",
        help="Root folder of the anomaly dataset.",
    )
    parser.add_argument(
        "--cluster-map",
        type=str,
        default=None,
        help="Optional path to cluster_map.json (defaults to dataset root).",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Validation CSV (defaults to dataset_root/val_mixed/val.csv).",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Test CSV (defaults to dataset_root/test_mixed/test.csv).",
    )
    parser.add_argument(
        "--ifca-trace",
        action="store_true",
        help="Print step-by-step IFCA events (broadcast, selection, aggregation).",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default=None,
        help="Optional path to save a bar chart of ROC-AUC/Accuracy for clusters and global.",
    )
    parser.add_argument(
        "--server-mixed-file",
        type=str,
        default=None,
        help="Mixed-labeled CSV used for server-side fine-tuning "
        "(defaults to dataset_root/server_mixed/train.csv).",
    )
    parser.add_argument(
        "--server-finetune-epochs",
        type=int,
        default=1,
        help="Epochs for server-side fine-tuning on the mixed dataset.",
    )
    parser.add_argument(
        "--server-finetune-lr",
        type=float,
        default=0.01,
        help="Learning rate for server-side fine-tuning.",
    )
    return parser.parse_args()


def extract_client_id(csv_path: Path) -> int:
    suffix = csv_path.stem.split("_")[-1]
    return int(suffix)


def build_clients(
    shard_paths: Sequence[Path],
    device: torch.device,
    category_levels: Dict[str, list[str]],
    client_config: ClientConfig,
) -> list[Client]:
    clients: list[Client] = []
    for shard_path in shard_paths:
        client_id = extract_client_id(shard_path)
        clients.append(Client(client_id, shard_path, device, client_config, category_levels))
    return clients


def load_cluster_assignments(path: Path) -> Dict[int, int]:
    raw = json.loads(path.read_text())
    assignments: Dict[int, int] = {}
    if isinstance(raw, list):
        assignments = {idx: int(value) for idx, value in enumerate(raw)}
    elif isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, str) and key.startswith("client_"):
                key = key.split("_")[-1]
            assignments[int(key)] = int(value)
    else:
        raise ValueError("Unsupported cluster map format.")
    return assignments


def evaluate_clusters(
    models: Sequence[torch.nn.Module],
    dataset: TensorDataset,
    device: torch.device,
    split: str,
    return_scores: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    for model in models:
        model.eval()

    loader = DataLoader(dataset, batch_size=256)
    scores = []
    labels = []
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            logits = torch.stack([model(features) for model in models], dim=0)
            probs = torch.softmax(logits, dim=-1).mean(dim=0)
            anomaly_prob = probs[:, 1]
            scores.append(anomaly_prob.cpu())
            labels.append(target)

    y_true = torch.cat(labels).numpy()
    y_score = torch.cat(scores).numpy()
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_score >= 0.5).astype(int)
    accuracy = (y_pred == y_true).mean()

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    roc_auc = roc_auc_score(y_true, y_score)

    metrics = {
        "split": split,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
    }
    if return_scores:
        return metrics, torch.from_numpy(y_true), torch.from_numpy(y_score)
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    split: str,
    return_scores: bool = False,
) -> Dict[str, float] | Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    loader = DataLoader(dataset, batch_size=256)
    scores = []
    labels = []
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            anomaly_prob = probs[:, 1]
            scores.append(anomaly_prob.cpu())
            labels.append(target)

    y_true = torch.cat(labels).numpy()
    y_score = torch.cat(scores).numpy()
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_score >= 0.5).astype(int)
    accuracy = (y_pred == y_true).mean()

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    roc_auc = roc_auc_score(y_true, y_score)
    metrics = {
        "split": split,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
    }
    if return_scores:
        return metrics, torch.from_numpy(y_true), torch.from_numpy(y_score)
    return metrics


def save_metrics_plot(metrics_list: Sequence[Dict[str, float]], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:  # pragma: no cover - graceful fallback
        print("matplotlib is not installed; skipping metric plot.")
        return False

    labels = [entry["label"] for entry in metrics_list]
    roc_aucs = [entry["metrics"]["roc_auc"] for entry in metrics_list]
    accuracies = [entry["metrics"]["accuracy"] for entry in metrics_list]

    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, roc_aucs, marker="o", linestyle="--", color="#4C72B0")
    axes[0].set_title("ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30)
    axes[0].set_ylim(0, 1)

    axes[1].plot(x, accuracies, marker="o", linestyle="--", color="#55A868")
    axes[1].set_title("Accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30)
    axes[1].set_ylim(0, 1)

    fig.suptitle("Cluster vs Global Performance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def finetune_global_model(
    model: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    if epochs <= 0:
        return
    labels_tensor = dataset.tensors[1]
    pos_count = (labels_tensor == 1).sum().item()
    neg_count = (labels_tensor == 0).sum().item()
    weights = torch.ones(len(dataset), dtype=torch.float32)
    if pos_count > 0:
        weights[labels_tensor == 1] = neg_count / max(pos_count, 1)
    if neg_count > 0:
        weights[labels_tensor == 0] = pos_count / max(neg_count, 1)
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler)
    if pos_count > 0 and neg_count > 0:
        weight_pos = neg_count / (pos_count + neg_count)
        weight_neg = pos_count / (pos_count + neg_count)
        class_weights = torch.tensor([weight_neg, weight_pos], device=device, dtype=torch.float32)
    else:
        class_weights = None
    criterion = (
        torch.nn.CrossEntropyLoss(weight=class_weights)
        if class_weights is not None
        else torch.nn.CrossEntropyLoss()
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def sanitize_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    idx = np.argmax(youden)
    return thresholds[idx]


def apply_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (y_pred == y_true).mean()
    return {"tpr": float(tpr), "fpr": float(fpr), "accuracy": float(acc)}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.clients_per_cluster and args.clients_per_round:
        raise ValueError("Use either --clients-per-cluster or --clients-per-round, not both.")

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "train_normal"
    if not train_dir.exists():
        raise RuntimeError(f"Training directory not found: {train_dir}")

    train_paths = sorted(train_dir.glob("client_*.csv"))
    if not train_paths:
        raise RuntimeError(f"No client CSVs found in {train_dir}")

    val_path = Path(args.val_file) if args.val_file else dataset_root / "val_mixed" / "val.csv"
    test_path = Path(args.test_file) if args.test_file else dataset_root / "test_mixed" / "test.csv"
    for split_path, split_name in [(val_path, "validation"), (test_path, "test")]:
        if not split_path.exists():
            raise RuntimeError(f"{split_name.title()} file not found: {split_path}")

    server_path = (
        Path(args.server_mixed_file)
        if args.server_mixed_file
        else dataset_root / "server_mixed" / "train.csv"
    )
    csv_paths_for_levels = list(train_paths) + [val_path, test_path]
    if server_path.exists():
        csv_paths_for_levels.append(server_path)
    category_levels = infer_category_levels(csv_paths_for_levels)

    sample_dataset = load_csv_dataset(train_paths[0], category_levels)
    feature_dim = sample_dataset.tensors[0].shape[1]

    client_config = ClientConfig(epochs=args.client_epochs)
    clients = build_clients(train_paths, device, category_levels, client_config)

    cluster_map_path = (
        Path(args.cluster_map) if args.cluster_map else dataset_root / "cluster_map.json"
    )
    initial_assignments = (
        load_cluster_assignments(cluster_map_path) if cluster_map_path.exists() else None
    )

    strategy = ClusteredFLStrategy(
        clients,
        device=device,
        config=StrategyConfig(
            clusters=args.clusters,
            rounds=args.rounds,
            in_features=feature_dim,
            clients_per_round=args.clients_per_round,
            verbose=args.ifca_trace,
            clients_per_cluster=args.clients_per_cluster,
        ),
        logger=ModelLedger(),
        initial_assignments=initial_assignments,
    )

    history = strategy.run()
    print("Training history:")
    for idx, metrics in enumerate(history):
        readable = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"round {idx}: {readable}")

    val_dataset = load_csv_dataset(val_path, category_levels)
    test_dataset = load_csv_dataset(test_path, category_levels)
    if server_path.exists():
        server_dataset = load_csv_dataset(server_path, category_levels)
        print(
            f"Server fine-tuning on {len(server_dataset)} samples "
            f"for {args.server_finetune_epochs} epoch(s)"
        )
        finetune_global_model(
            strategy.global_model,
            server_dataset,
            device,
            args.server_finetune_epochs,
            args.server_finetune_lr,
        )
    sanitize_model(strategy.global_model)
    for model in strategy.cluster_models:
        sanitize_model(model)

    val_metrics = evaluate_clusters(
        strategy.cluster_models, val_dataset, device, "val (clusters)", return_scores=True
    )
    test_metrics = evaluate_clusters(
        strategy.cluster_models, test_dataset, device, "test (clusters)", return_scores=True
    )
    val_global_metrics = evaluate_model(
        strategy.global_model, val_dataset, device, "val (global)", return_scores=True
    )
    test_global_metrics = evaluate_model(
        strategy.global_model, test_dataset, device, "test (global)", return_scores=True
    )

    if isinstance(val_metrics, tuple):
        val_metrics_dict, _, _ = val_metrics
    else:
        val_metrics_dict = val_metrics
    if isinstance(test_metrics, tuple):
        test_metrics_dict, _, _ = test_metrics
    else:
        test_metrics_dict = test_metrics

    if isinstance(val_global_metrics, tuple):
        val_global_metrics_dict, val_global_true, val_global_scores = val_global_metrics
    else:
        val_global_metrics_dict = val_global_metrics
        raise RuntimeError("Global evaluation must return scores for thresholding.")
    if isinstance(test_global_metrics, tuple):
        test_global_metrics_dict, test_global_true, test_global_scores = test_global_metrics
    else:
        test_global_metrics_dict = test_global_metrics
        raise RuntimeError("Global evaluation must return scores for thresholding.")

    threshold = best_threshold(
        val_global_true.numpy(), val_global_scores.numpy()
    )
    val_adj = apply_threshold_metrics(
        val_global_true.numpy(), val_global_scores.numpy(), threshold
    )
    test_adj = apply_threshold_metrics(
        test_global_true.numpy(), test_global_scores.numpy(), threshold
    )
    val_global_metrics_dict.update(val_adj)
    test_global_metrics_dict.update(test_adj)

    summary_entries = [
        {"label": val_metrics_dict["split"], "metrics": val_metrics_dict},
        {"label": test_metrics_dict["split"], "metrics": test_metrics_dict},
        {"label": val_global_metrics_dict["split"], "metrics": val_global_metrics_dict},
        {"label": test_global_metrics_dict["split"], "metrics": test_global_metrics_dict},
    ]

    print("Evaluation metrics:")
    for metrics in [entry["metrics"] for entry in summary_entries]:
        print(
            f"{metrics['split']}: TPR={metrics['tpr']:.4f}, "
            f"FPR={metrics['fpr']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
            f"ACC={metrics['accuracy']:.4f}"
        )

    if args.plot_file:
        if save_metrics_plot(summary_entries, Path(args.plot_file)):
            print(f"Saved metric plot to {args.plot_file}")


if __name__ == "__main__":
    main()
