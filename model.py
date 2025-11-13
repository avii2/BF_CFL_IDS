import torch
import torch.nn as nn
from typing import Iterable, Tuple


class SimpleNet(nn.Module):
    """Tiny MLP that works for the toy 2-D classification shards."""

    def __init__(
        self,
        in_features: int = 2,
        hidden_dims: Iterable[int] = (32, 32, 32),
        out_dim: int = 2,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = in_features
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


def build_model(
    device: torch.device, in_features: int = 2, clusters: int = 2
) -> Tuple[nn.Module, ...]:
    """Return one model instance per cluster."""
    return tuple(SimpleNet(in_features=in_features).to(device) for _ in range(clusters))


def weights_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all parameters into a single vector tensor."""
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def vector_to_weights(model: nn.Module, vector: torch.Tensor) -> None:
    """Load a flat tensor back into the model's parameters."""
    pointer = 0
    for param in model.parameters():
        numel = param.data.numel()
        param.data.copy_(vector[pointer : pointer + numel].view_as(param.data))
        pointer += numel
