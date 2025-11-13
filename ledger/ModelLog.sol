// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ModelLog
/// @notice Minimal contract that stores hashes of aggregated cluster weights
contract ModelLog {
    struct Entry {
        uint256 roundId;
        uint256 clusterId;
        bytes32 weightHash;
        uint256 timestamp;
    }

    event ModelStored(uint256 indexed roundId, uint256 indexed clusterId, bytes32 weightHash);

    Entry[] public entries;

    function store(
        uint256 roundId,
        uint256 clusterId,
        bytes32 weightHash
    ) external {
        entries.push(
            Entry({
                roundId: roundId,
                clusterId: clusterId,
                weightHash: weightHash,
                timestamp: block.timestamp
            })
        );
        emit ModelStored(roundId, clusterId, weightHash);
    }

    function entriesCount() external view returns (uint256) {
        return entries.length;
    }
}

