import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch


class ModelLedger:
    """Lightweight ledger that records hashed model snapshots for auditing."""

    def __init__(self, log_path: str | Path = "ledger/model_ledger.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self, round_idx: int, cluster_id: int | str, weights_vec: torch.Tensor
    ) -> None:
        payload = weights_vec.detach().cpu().numpy().tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        entry = {
            "round": round_idx,
            "cluster": cluster_id,
            "weight_hash": digest,
            "timestamp": int(time.time()),
        }
        with self.log_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(entry) + "\n")
        print(f"[model-ledger] round={round_idx} cluster={cluster_id} hash={digest[:10]}...")
