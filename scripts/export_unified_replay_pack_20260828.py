#!/usr/bin/env python3
"""Export a compact, immutable pre-update replay pack from saved rollouts.

This deliberately does not reconstruct missing token-level log probabilities:
the AgentFlow JSON writer in this pinned stack stores text and the complete
agent trajectory, while VERL's tensor batch is not persisted in that writer.
The pack records that limitation as an explicit recomputation contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def export_pack(raw_root: Path, output: Path, *, step: str, config_path: Path | None, route_path: Path | None) -> dict[str, Any]:
    # The current writer stores each prompt under ``idx_<id>/`` directly
    # below the step directory (there is no intermediate ``idx/`` folder).
    files = sorted(raw_root.glob(f"**/train/step_{step}/idx_*/*.json"))
    if not files:
        raise SystemExit(f"no rollout JSON found below {raw_root} for step {step}")

    config_hash = sha256_file(config_path) if config_path and config_path.exists() else None
    route_hash = sha256_file(route_path) if route_path and route_path.exists() else None
    route = load_json(route_path) if route_path and route_path.exists() else None
    records = []
    for path in files:
        item = load_json(path)
        records.append(
            {
                "source_file": str(path),
                "source_file_sha256": sha256_file(path),
                "group_id": str(item.get("idx")),
                "candidate_id": str(item.get("id")),
                "prompt": item.get("prompt"),
                "groundtruth": item.get("groundtruth"),
                "answer_extracted": item.get("answer_extracted"),
                "reward": float(item["reward"]) if item.get("reward") is not None else None,
                "tools": item.get("tools", []),
                "trajectory": item.get("total_result"),
                "timestamp": item.get("timestamp"),
                "token_ids_available": False,
                "old_log_probs_available": False,
            }
        )

    pack = {
        "schema_version": 1,
        "kind": "agentflow_unified_pre_update_replay_pack",
        "source": {
            "raw_rollout_root": str(raw_root),
            "raw_rollout_root_file_count": len(files),
            "pre_update_step": str(step),
            "config_path": str(config_path) if config_path else None,
            "config_sha256": config_hash,
            "route_path": str(route_path) if route_path else None,
            "route_sha256": route_hash,
        },
        "protocol": {
            "model_path": "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
            "temperature": 0.7,
            "top_p": 0.99,
            "rollout_n": 2,
            "lora_rank": 8,
            "lora_alpha": 16,
            "optimizer_steps": "not represented in pack; this is pre-update rollout data",
        },
        "role_routing": {
            "planner_main": "qwen-actor -> latest synchronized LoRA",
            "planner_fixed": "qwen-base -> no LoRA",
            "verifier": "qwen-base -> no LoRA",
            "executor": "qwen-base -> no LoRA",
            "route_state_snapshot": route,
        },
        "recompute_contract": {
            "response_token_ids": "not persisted by current AgentFlow rollout writer",
            "old_log_probs": "recompute from the recorded pre-update policy before any optimizer step",
            "trajectory": "complete total_result field is preserved where available",
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pack, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return pack


def dry_run(path: Path) -> dict[str, Any]:
    pack = load_json(path)
    assert pack["kind"] == "agentflow_unified_pre_update_replay_pack"
    assert pack["source"]["pre_update_step"]
    assert pack["records"]
    for record in pack["records"]:
        assert record["prompt"] is not None
        assert record["groundtruth"] is not None
        assert record["reward"] in (0.0, 1.0)
        assert record["trajectory"] is not None
    return {"status": "ok", "records": len(pack["records"]), "token_ids_available": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--step", default="1")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--route-state", type=Path)
    parser.add_argument("--dry-run", type=Path)
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(args.dry_run), sort_keys=True))
        return
    if not args.raw_root or not args.output:
        parser.error("--raw-root and --output are required for export")
    pack = export_pack(args.raw_root, args.output, step=args.step, config_path=args.config, route_path=args.route_state)
    print(json.dumps({"status": "ok", "records": len(pack["records"]), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
