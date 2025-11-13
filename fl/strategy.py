from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from fl.client import Client
from model import build_model, vector_to_weights, weights_to_vector, SimpleNet


@dataclass
class StrategyConfig:
    clusters: int = 2
    rounds: int = 3
    in_features: int = 2
    clients_per_round: int | None = None
    verbose: bool = False
    clients_per_cluster: int | None = None


def coordinate_median(updates: Sequence[torch.Tensor]) -> torch.Tensor:
    """Coordinate-wise median to fuse client updates robustly."""
    stacked = torch.stack(updates)
    return torch.median(stacked, dim=0).values


class ClusteredFLStrategy:
    """Minimal IFCA-style coordinator with robust aggregation."""

    def __init__(
        self,
        clients: Iterable[Client],
        device: torch.device,
        config: StrategyConfig,
        logger: Any | None = None,
        initial_assignments: Dict[int, int] | None = None,
    ) -> None:
        self.clients = list(clients)
        self.device = device
        self.config = config
        self.logger = logger
        self.verbose = config.verbose
        self.cluster_models = build_model(
            device=self.device, clusters=config.clusters, in_features=config.in_features
        )
        self.global_model = SimpleNet(in_features=config.in_features).to(self.device)
        self.assignments: Dict[int, int] = {}
        for client in self.clients:
            if initial_assignments and client.client_id in initial_assignments:
                cluster_id = int(initial_assignments[client.client_id])
            else:
                cluster_id = client.client_id % config.clusters
            self.assignments[client.client_id] = cluster_id % self.config.clusters

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _select_clients(self) -> List[Client]:
        if self.config.clients_per_cluster:
            selected: List[Client] = []
            for cluster_idx in range(self.config.clusters):
                candidates = [
                    client
                    for client in self.clients
                    if self.assignments.get(client.client_id, client.client_id % self.config.clusters)
                    == cluster_idx
                ]
                if not candidates:
                    continue
                quota = min(len(candidates), self.config.clients_per_cluster)
                selected.extend(random.sample(candidates, quota))
            return selected

        if not self.config.clients_per_round or self.config.clients_per_round >= len(
            self.clients
        ):
            return self.clients
        return random.sample(self.clients, self.config.clients_per_round)

    def _ifca_client_step(
        self, client: Client
    ) -> Tuple[int, torch.Tensor, Dict[str, float]]:
        best_cluster = -1
        best_loss = float("inf")
        best_weights: torch.Tensor | None = None
        best_metrics: Dict[str, float] | None = None

        for cluster_idx, model in enumerate(self.cluster_models):
            weights_vec, metrics = client.train_for_cluster(model, cluster_idx)
            loss = metrics.get("loss", float("inf"))
            if loss < best_loss:
                best_loss = loss
                best_cluster = cluster_idx
                best_weights = weights_vec
                best_metrics = metrics

        if best_weights is None or best_metrics is None:
            raise RuntimeError("IFCA step failed to produce any cluster updates.")

        self.assignments[client.client_id] = best_cluster
        return best_cluster, best_weights, best_metrics

    def run_round(self, round_idx: int) -> Dict[str, float]:
        selected = self._select_clients()
        self._log(
            f"[Round {round_idx} | Step 1] Broadcasting {self.config.clusters} cluster heads "
            f"to {len(selected)} client(s)"
        )

        cluster_updates: Dict[int, List[torch.Tensor]] = {
            idx: [] for idx in range(self.config.clusters)
        }
        metrics: Dict[str, float] = {}
        assignment_trace: Dict[int, List[int]] = defaultdict(list)

        for client in selected:
            cluster_id, weights_vec, client_metrics = self._ifca_client_step(client)
            cluster_updates[cluster_id].append(weights_vec)
            for key, value in client_metrics.items():
                metrics.setdefault(key, 0.0)
                metrics[key] += value
            assignment_trace[cluster_id].append(client.client_id)
            loss_val = client_metrics.get("loss")
            if loss_val is not None:
                self._log(
                    f"[Round {round_idx} | Step 2] Client {client.client_id} "
                    f"joins cluster {cluster_id} (loss={loss_val:.4f})"
                )
            else:
                self._log(
                    f"[Round {round_idx} | Step 2] Client {client.client_id} joins cluster {cluster_id}"
                )

        aggregated_vectors: List[torch.Tensor] = []
        for cluster_idx, updates in cluster_updates.items():
            if not updates:
                continue
            aggregated = coordinate_median(updates)
            vector_to_weights(self.cluster_models[cluster_idx], aggregated)
            self._log(
                f"[Round {round_idx} | Step 3-4] Aggregated cluster {cluster_idx} "
                f"from {len(updates)} update(s) via coordinate median"
            )
            if self.logger:
                self.logger.record(round_idx, cluster_idx, aggregated)
            aggregated_vectors.append(aggregated)

        if aggregated_vectors:
            stacked = torch.stack(aggregated_vectors)
            global_vector = torch.mean(stacked, dim=0)
            vector_to_weights(self.global_model, global_vector)
            self._log(
                f"[Round {round_idx} | Global] Aggregated global model from "
                f"{len(aggregated_vectors)} cluster head(s)"
            )
            if self.logger:
                self.logger.record(round_idx, "global", global_vector)

        if assignment_trace:
            summary = ", ".join(
                f"cluster {cid}: {sorted(members)}"
                for cid, members in sorted(assignment_trace.items())
            )
            self._log(f"[Round {round_idx} | Step 5] Assignment summary -> {summary}")

        for key in list(metrics.keys()):
            metrics[key] = metrics[key] / max(len(selected), 1)
        metrics["clients"] = float(len(selected))
        return metrics

    def run(self) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for round_idx in range(self.config.rounds):
            history.append(self.run_round(round_idx))
        return history
