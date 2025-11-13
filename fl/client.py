import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import SimpleNet, weights_to_vector
from data_utils import load_csv_dataset


@dataclass
class ClientConfig:
    epochs: int = 2
    batch_size: int = 4
    lr: float = 0.05


class Client:
    def __init__(
        self,
        client_id: int,
        shard_path: Path,
        device: torch.device,
        config: ClientConfig | None = None,
        category_levels: Dict[str, List[str]] | None = None,
    ) -> None:
        self.client_id = client_id
        self.shard_path = shard_path
        self.device = device
        self.config = config or ClientConfig()
        self.category_levels = category_levels
        self.data = self._load_dataset()

    def _load_dataset(self) -> TensorDataset:
        return load_csv_dataset(self.shard_path, self.category_levels)

    def train_for_cluster(
        self, global_model: SimpleNet, cluster_id: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train on the client's shard using the provided cluster model."""
        local_model = copy.deepcopy(global_model).to(self.device)
        loader = DataLoader(self.data, batch_size=self.config.batch_size, shuffle=True)
        criterion: nn.Module = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.config.lr)

        local_model.train()
        total_loss = 0.0
        total_examples = 0
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = local_model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)
                total_examples += batch_x.size(0)

        metrics = {
            "loss": total_loss / max(total_examples, 1),
            "examples": float(total_examples),
        }
        return weights_to_vector(local_model), metrics
