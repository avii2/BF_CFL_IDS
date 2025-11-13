# blockchain empowered clustered federeated learning intrusion detection system 

This repo contains a lightweight clustered federated learning (IFCA-inspired) simulation
paired with a lightweight model ledger. Models are trained on per-client normal-only traffic
while validation/test splits mix in anomalies so you can report detection metrics. A
Solidity contract + deploy helper are included when you want to anchor hashes on a real
(local) chain, but the default workflow simply hashes the aggregated weights to a JSON log
for easy inspection.

## Layout

```
core_fed_cluster/
├── model.py                
├── data_utils.py            # CSV loader shared by clients + evaluators
├── fl/
│   ├── client.py            # Client wrapper: load shard, per-cluster training for IFCA
│   └── strategy.py          # IFCA coordinator + coordinate-wise median aggregation
├── ledger/
│   ├── ModelLog.sol         # Solidity contract storing weight hashes
│   ├── deploy.py            # Helper to compile/deploy via web3 + py-solc-x
│   └── logger.py            # File-backed model ledger
├── network_anomaly_dataset/ # Train-normal shards + mixed val/test splits
├── demo.py                  # Python entrypoint wiring all pieces together
├── requirements.txt         # Python dependencies
└── run_demo.sh              # Convenience script to set up venv + execute demo
```

## Quickstart

```bash
./run_demo.sh --rounds 5 --clusters 4
```

The script creates `.venv/`, installs requirements, and launches the simulation using
`network_anomaly_dataset/`. Each of the 20 train clients only contains normal samples,
preserving the anomaly-centric sharding strategy. A small mixed-labeled buffer under
`server_mixed/` is reserved for server-side fine-tuning so the global head still sees
anomalies. At the end of training, the script evaluates on the provided validation/test mixes
and prints TPR, FPR, ROC-AUC, **and accuracy** for both the per-cluster ensemble and the
global head. Ledger entries land in `ledger/model_ledger.jsonl`.
entries land in `ledger/model_ledger.jsonl`.

If you relocate the dataset, point the demo at the new location via
`./run_demo.sh --dataset-root /path/to/network_anomaly_dataset`. You can also override the
cluster map or split files with `--cluster-map`, `--val-file`, and `--test-file`.
Add `--ifca-trace` to stream the Step 1–5 IFCA trace (broadcast, cluster selection, local
updates, aggregation, reassignment) directly in the terminal. Use `--client-epochs N` to
change how many SGD epochs each edge runs per round, pass `--server-finetune-epochs M` /
`--server-finetune-lr LR` to control the mixed-data fine-tuning, and `--plot-file metrics.png`
to save a quick ROC-AUC/accuracy bar chart for val/test × (cluster/global).
Need a fixed number of clients per cluster? Use `--clients-per-cluster K` (for example,
`./run_demo.sh --rounds 3 --clusters 2 --clients-per-cluster 3 --ifca-trace`) to sample
exactly `K` currently assigned clients from each cluster head every round—ideal for demos
with “3 clients → 2 heads → 1 global model.”

### IFCA-style clustering

In every training round the server broadcasts all cluster heads. Each participating client
locally fine-tunes **every** head on its shard, measures the post-update loss, and selects
the cluster that best explains its data (the IFCA assignment step). Only the winning update
is sent back and aggregated with a coordinate-wise median to keep poisoned updates in check.
The freshly updated heads are then averaged into a **global** model that also feeds metrics
and ledger logs. The optional `cluster_map.json` is used purely for warm-starting
assignments; clustering is otherwise learned online.

## Model ledger + optional on-chain anchor

1. Start a dev node (Ganache/Hardhat/Anvil) and grab an unlocked account (optional).
2. Export:
   ```bash
   export PROVIDER_URL=http://127.0.0.1:8545
   export DEPLOYER_ADDRESS=0x...
   export DEPLOYER_KEY=0x...
   ```
3. Deploy:
   ```bash
   python ledger/deploy.py
   ```

The default `ledger/logger.py` writes hashed cluster heads **and** the derived global head
to `ledger/model_ledger.jsonl`; adapt it to call the deployed contract if you want an actual
chain-backed log.
