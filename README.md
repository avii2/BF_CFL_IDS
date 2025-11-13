# blockchain empowered clustered federeated learning intrusion detection system 

## 1. Overview

This project demonstrates clustered federated learning (IFCA) running on the UNSW-NB15
intrusion dataset. Every client trains locally on **normal-only** traffic, cluster heads are
updated using a robust coordinate-wise median, and a global model is derived from the cluster
heads. All aggregated weights are hashed into a lightweight “model ledger” for auditability,
and an optional Solidity contract is provided if you want to anchor those hashes on-chain.

## 2. Repository Structure

```
core_fed_cluster/
├── demo.py                  # CLI entrypoint (training, evaluation, tracing, plotting)
├── run_demo.sh              # Convenience wrapper: venv + requirements + demo.py
├── model.py                 # Shared SimpleNet MLP (3 hidden layers) for every cluster head
├── data_utils.py            # CSV loader + feature encoder for UNSW-NB15 splits
├── fl/
│   ├── client.py            # ClientConfig + local SGD loop for each shard
│   └── strategy.py          # IFCA coordinator with coordinate-median + global averaging
├── ledger/
│   ├── logger.py            # File-backed model ledger (default destination for hashes)
│   ├── deploy.py            # Helper to deploy ModelLog.sol via web3/py-solc-x
│   └── ModelLog.sol         # Optional on-chain storage contract
├── scripts/
│   └── prepare_unsw_dataset.py  # Splits raw UNSW training/testing CSVs into clients + val/test/server buffers
├── network_anomaly_dataset/ # Generated shards: train_normal/, val_mixed/, test_mixed/, server_mixed/
├── data/                    # Original UNSW CSVs (training/testing) + legacy toy shards
├── requirements.txt         # Python dependencies
└── metrics.png              # Example metrics plot (created via --plot-file)
```

## 3. Dataset Preparation

1. Download the official UNSW-NB15 `*_training-set.csv` and `*_testing-set.csv` into `data/`.
2. Run the splitter (uses only normal rows for clients, mixed rows for validation/test/server):
   ```bash
   .venv/bin/python scripts/prepare_unsw_dataset.py \
     --train-csv data/UNSW_NB15_training-set.csv \
     --test-csv data/UNSW_NB15_testing-set.csv \
     --clients 20 --clusters 4 --server-ratio 0.1 --val-ratio 0.4
   ```
   - `train_normal/` → 20 evenly sized, normal-only client shards.
   - `val_mixed/val.csv` and `test_mixed/test.csv` → mixed (label 0/1) evaluation sets.
   - `server_mixed/train.csv` → balanced subset used for server-side fine-tuning.
   - `cluster_map.json` → warm-start assignments (round-robin by default).

## 4. Running the Simulation

```bash
./run_demo.sh --rounds 5 --clusters 4 \
  --clients-per-cluster 3 \
  --client-epochs 2 \
  --server-finetune-epochs 1 \
  --server-finetune-lr 0.01 \
  --plot-file metrics.png
```

What happens:
1. The script creates `.venv` (if missing), installs `requirements.txt`, and launches `demo.py`.
2. Each round:
   - **Broadcast:** all cluster heads are pushed to the sampled clients.
   - **Local training:** every selected client fine-tunes each head for `client-epochs` SGD epochs on its normal-only shard.
   - **Cluster selection:** the client keeps only the best head (lowest loss) and uploads that update.
   - **Aggregation:** the server applies a coordinate-wise median per cluster, then averages new heads into a global model.
   - **Ledger logging:** hashed weights are appended to `ledger/model_ledger.jsonl`.
3. After all rounds, the server optionally fine-tunes the global head on `server_mixed/train.csv` (class-weighted + oversampled).
4. Metrics (TPR, FPR, ROC-AUC, accuracy) are reported for both the cluster ensemble and the global head on `val_mixed` and `test_mixed`.
5. If `--plot-file` is provided and `matplotlib` is installed, a ROC-AUC/accuracy bar chart is saved.

### Useful Flags

- `--dataset-root`: point to a different dataset directory (default `network_anomaly_dataset`).
- `--cluster-map`, `--val-file`, `--test-file`, `--server-mixed-file`: override individual CSVs.
- `--clients-per-round` or `--clients-per-cluster`: control client sampling strategy.
- `--client-epochs`: local epochs per round (default 2).
- `--server-finetune-epochs` / `--server-finetune-lr`: adjust mixed-data fine-tuning depth.
- `--ifca-trace`: print step-by-step broadcast/selection/aggregation logs for transparency.

## 5. IFCA + Global Aggregation

- **Client side:** every client receives *all* heads, runs short local SGD on each, computes losses, and selects the best-fitting head.
- **Cluster head update:** the coordinator aggregates all winning updates for a cluster via coordinate-wise median, mitigating outliers.
- **Global head:** the refreshed cluster heads are averaged into a single global model, sanitized to remove NaN/Inf, and logged.
- **Server fine-tuning:** the global model performs class-weighted SGD on `server_mixed/train.csv` to regain visibility into anomalies despite clients only seeing normals.

## 6. Model Ledger & Optional On-Chain Anchor

- Default logging: `ledger/logger.py` hashes every cluster/global vector and appends JSON lines to `ledger/model_ledger.jsonl`.
- To anchor hashes on a blockchain:
  1. Start a dev chain (Ganache/Hardhat/Anvil).
  2. Export credentials:
     ```bash
     export PROVIDER_URL=http://127.0.0.1:8545
     export DEPLOYER_ADDRESS=0x...
     export DEPLOYER_KEY=0x...
     ```
  3. Deploy the contract:
     ```bash
     python ledger/deploy.py
     ```
  4. Extend `ledger/logger.py` to interact with the deployed contract instead of writing to disk.

## 7. Requirements

- Python 3.12+ (demo imports for 3.12/3.13 already handled in `.venv`)
- PyTorch 2.x, NumPy, pandas, scikit-learn, matplotlib (optional for plotting), web3, py-solc-x
- For on-chain logging: a local Ethereum-compatible node with funded account credentials

## 8. Limitations & Next Steps

- Clients currently see **only** normal traffic; even with server fine-tuning, global ROC-AUC can degrade if the server buffer is too imbalanced. Consider:
  - Injecting a small percentage of labeled anomalies into select clients.
  - Standardizing features (z-score per column) prior to training.
  - Increasing rounds / participating clients to strengthen cluster heads before global averaging.
- Intrusion mitigation logic (blocking/quarantine) is not implemented; the demo focuses on collaborative model training + logging.

## 9. Licensing & Contributions

Feel free to fork and extend the project. Suggestions for balanced dataset preparation, stronger anomaly detectors, or production-grade ledger integrations are especially welcome. Submit issues/PRs via the GitHub repository.

---

**TL;DR:** Run `./run_demo.sh --rounds 5 --clusters 4` to reproduce clustered federated learning on UNSW-NB15, inspect per-cluster and global detection metrics, and audit the hashed weights via the model ledger. Fine-tune the behavior with the CLI flags above to explore different training regimes.
