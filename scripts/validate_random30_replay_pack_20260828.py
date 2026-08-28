#!/usr/bin/env python3
"""Fresh-process, no-generation validation for the random30 replay artifacts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

REQUIRED = {"prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask", "old_log_probs", "token_level_scores", "token_level_rewards", "advantages", "returns", "is_drop_mask"}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    pack = torch.load(args.pack, map_location="cpu", weights_only=False)
    snap = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    if pack.get("kind") != "agentflow_unified_authentic_pre_update_replay_pack":
        raise SystemExit("invalid replay pack kind")
    if snap.get("kind") != "agentflow_behavior_policy_snapshot":
        raise SystemExit("invalid behavior snapshot kind")
    missing = sorted(REQUIRED - set(pack.get("tensor_fields", {})))
    if missing: raise SystemExit(f"missing replay tensor fields: {missing}")
    tensors = pack["tensor_fields"]
    lengths = {key: int(value.shape[0]) for key, value in tensors.items() if hasattr(value, "shape") and value.ndim > 0}
    if len(set(lengths.values())) != 1:
        raise SystemExit(f"inconsistent first dimensions: {lengths}")
    evidence = sorted(args.evidence_dir.glob("rollout_*.json"))
    if len(evidence) != args.expected:
        raise SystemExit(f"expected {args.expected} evidence files, got {len(evidence)}")
    ids = []
    for path in evidence:
        row = json.loads(path.read_text(encoding="utf-8"))
        rid = row.get("rollout", {}).get("rollout_id")
        if not rid: raise SystemExit(f"missing rollout id in {path}")
        ids.append(rid)
    if len(set(ids)) != len(ids): raise SystemExit("duplicate rollout ids")
    descriptors = snap.get("tensor_descriptors", [])
    state = snap.get("lora_state", {})
    if len(descriptors) != len(state): raise SystemExit("snapshot descriptor/state count mismatch")
    digest = hashlib.sha256()
    for item in descriptors:
        value = state[item["name"]].contiguous()
        raw = value.view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise SystemExit(f"snapshot tensor hash mismatch: {item['name']}")
        digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if digest.hexdigest() != snap.get("lora_hash"):
        raise SystemExit("snapshot aggregate hash mismatch")
    result = {
        "status": "ok", "rollout_requests": 0, "external_calls": 0,
        "evidence_count": len(evidence), "pack_batch_size": lengths.get("prompts"),
        "pack_sha256": sha(args.pack), "snapshot_sha256": sha(args.snapshot),
        "evidence_manifest_sha256": hashlib.sha256("".join(sha(x) for x in evidence).encode()).hexdigest(),
        "tensor_fields": sorted(tensors), "snapshot_lora_hash": snap["lora_hash"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
