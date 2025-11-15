# Blockchain-Empowered Clustered Federated Learning IDS

## 🚀 Project Snapshot
A lightweight demo that blends **clustered federated learning** with a **hash-anchored model ledger** to detect network intrusions.

1. Each client trains locally on normal traffic from the UNSW-NB15 dataset.  
2. Updates are merged at cluster heads via a coordinate-wise median (IFCA).  
3. Cluster heads are averaged into a global model.  
4. Every weight vector’s SHA-256 hash is appended to a file-based ledger (optionally anchored on-chain with Solidity).

---
## 🐍 Requirements

| Category | Packages / Tools | Notes |
|----------|------------------|-------|
| **Python** | 3.12 or newer | Core runtime |
| **Core ML stack** | PyTorch ≥ 2, NumPy, pandas, scikit-learn | Required for training & evaluation |
| **Optional plotting** | matplotlib | Only needed if `--plot-file` is used |
| **On-chain anchor** | web3, py-solc-x, local Ethereum node (Ganache / Hardhat / Anvil) | Needed **only** when pushing model hashes on-chain |

## 📈 IFCA in a Nutshell

1. **Broadcast** – push every cluster head to the sampled clients.  
2. **Local training** – each client fine-tunes **all** heads for a few SGD epochs on its normal-only shard.  
3. **Head selection** – the client keeps the head with the lowest loss and uploads only that update.  
4. **Robust aggregation** – the coordinator applies a coordinate-wise median to merge updates within each cluster.  
5. **Global averaging** – refreshed heads are averaged into a single global model and its SHA-256 hash is logged.  
6. **Optional fine-tune** – run a short, class-weighted SGD pass on a balanced mixed server buffer to regain anomaly visibility.


## 🧾 Model Ledger

| Mode | What Happens | How to Use |
|------|--------------|------------|
| **File-based (default)** | SHA-256 hash of every cluster/global weight vector is appended to `ledger/model_ledger.jsonl`. | No extra setup; enabled out-of-the-box. |
| **On-chain anchor (optional)** | Hashes are stored in an Ethereum smart contract (`ModelLog.sol`) for immutable auditability. | 1. Start a local node (Ganache / Hardhat / Anvil).<br>2. Export creds: <br>&nbsp;&nbsp;```bash<br>export PROVIDER_URL=http://127.0.0.1:8545<br>export DEPLOYER_ADDRESS=0x...<br>export DEPLOYER_KEY=0x...<br>```<br>3. Deploy contract: <br>&nbsp;&nbsp;```bash<br>python ledger/deploy.py<br>```<br>4. Switch `ledger/logger.py` to send transactions instead of writing to disk. |



## 🤝 Contributing

We happily welcome pull requests, feature ideas, and bug reports!

* **Fork** the repo and open a PR for enhancements or fixes.  
* **Issues** are the place to ask questions or propose larger changes.  
* Areas where help is most valuable:  
  * Smarter or more robust aggregation strategies  
  * Balanced dataset generation / preprocessing scripts  
  * Hardening the model-ledger for production (IPFS, L2, etc.)

Feel Free to reachout to me Gmail : anilkumarbarupal.01@gmail.com


Project is released under the **IIT License**—have fun hacking! 🚀



