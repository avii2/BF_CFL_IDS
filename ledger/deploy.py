"""
Utility script to compile and deploy ModelLog.sol onto a local dev chain.

Usage:
    export PROVIDER_URL=http://127.0.0.1:8545
    export DEPLOYER_KEY=0x...
    export DEPLOYER_ADDRESS=0x...
    python ledger/deploy.py

The script is intentionally simple and meant for demo purposes only.
"""

from __future__ import annotations

import os
from pathlib import Path

from solcx import compile_standard, install_solc
from web3 import Web3

CONTRACT_PATH = Path(__file__).with_name("ModelLog.sol")
SOLC_VERSION = "0.8.20"


def compile_contract() -> tuple[list[dict], str]:
    source = CONTRACT_PATH.read_text()
    install_solc(SOLC_VERSION)
    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {"ModelLog.sol": {"content": source}},
            "settings": {
                "outputSelection": {"*": {"*": ["abi", "metadata", "evm.bytecode"]}}
            },
        },
        solc_version=SOLC_VERSION,
    )
    contract_interface = compiled["contracts"]["ModelLog.sol"]["ModelLog"]
    abi = contract_interface["abi"]
    bytecode = contract_interface["evm"]["bytecode"]["object"]
    return abi, bytecode


def deploy() -> None:
    provider = os.environ.get("PROVIDER_URL", "http://127.0.0.1:8545")
    key = os.environ.get("DEPLOYER_KEY")
    address = os.environ.get("DEPLOYER_ADDRESS")

    if not key or not address:
        raise RuntimeError("DEPLOYER_KEY and DEPLOYER_ADDRESS env vars are required.")

    abi, bytecode = compile_contract()
    web3 = Web3(Web3.HTTPProvider(provider))
    if not web3.is_connected():
        raise RuntimeError(f"Cannot connect to Ethereum node at {provider}")

    contract = web3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = web3.eth.get_transaction_count(address)
    tx = contract.constructor().build_transaction(
        {"from": address, "nonce": nonce, "gas": 3_000_000, "gasPrice": web3.to_wei(2, "gwei")}
    )
    signed = web3.eth.account.sign_transaction(tx, private_key=key)
    tx_hash = web3.eth.send_raw_transaction(signed.rawTransaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Contract deployed at {receipt.contractAddress}")
    Path("deploy_receipt.json").write_text(web3.to_json(receipt))


if __name__ == "__main__":
    deploy()
