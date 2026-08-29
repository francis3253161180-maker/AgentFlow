#!/usr/bin/env python3
"""Final actor-only diagnostic entry point using the vLLM OpenAI server path."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from run_game24_actor_diversity_diagnostic_20260829 import (
    GROUPS,
    SYSTEM_PROMPT,
    analyze_rows,
    load_frozen_prompts,
    model_inventory,
    prepare_chat_template,
    prepare_peft_adapter,
    sha256_bytes,
    snapshot_metadata,
    start_server,
    stop_server,
    gpu_monitor,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=Path("rollout_data/46.38.243.197/gameof24-planner-temp0-causal-sanity-20260829_20260829-135524"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--snapshot", type=Path, default=Path("/root/autodl-tmp/tmp/gameof24_planner_temp0_causal_sanity_20260829/gameof24-planner-temp0-causal-sanity-20260829_20260829_135323_behavior_snapshot.pt"))
    parser.add_argument("--work-dir", type=Path, default=Path("/root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829/server_final"))
    parser.add_argument("--output", type=Path, default=Path("log/2026-08-29_game24_actor_diversity_diagnostic_results.json"))
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--server-timeout", type=float, default=600)
    args = parser.parse_args()

    for path in (args.rollout_root, args.model, args.snapshot):
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    groups = load_frozen_prompts(args.rollout_root)
    snapshot = snapshot_metadata(args.snapshot)
    model_info = model_inventory(args.model)
    adapter_info = prepare_peft_adapter(args.model, args.snapshot, args.work_dir / "qwen-actor-lora")
    chat_template = prepare_chat_template(args.model, args.work_dir / "qwen_chat_template.jinja")
    vllm_log = args.work_dir / "vllm_actor.log"
    gpu_log = args.work_dir / "gpu.tsv"
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(gpu_log, monitor_stop), daemon=True)
    process = None
    rows: list[dict] = []
    errors: list[dict] = []
    started = time.perf_counter()
    server_info = {}
    try:
        process, server_info = start_server(args, args.work_dir / "qwen-actor-lora", Path(chat_template["path"]), vllm_log)
        monitor.start()
        from openai import OpenAI
        from transformers import AutoTokenizer
        from run_game24_actor_diversity_diagnostic_20260829 import parse_next_step
        from agentflow.models.formatters import NextStep

        schema = NextStep.model_json_schema() if hasattr(NextStep, "model_json_schema") else NextStep.schema()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="local-diagnostic-token", timeout=180)
        for group_id in GROUPS:
            frozen = groups[group_id]
            prompt_token_ids = tokenizer.encode(frozen["prompt"], add_special_tokens=False)
            for temperature in (0.0, 0.7):
                for repetition in range(1, 5):
                    request_started = time.perf_counter()
                    raw = ""
                    parsed = None
                    error = None
                    try:
                        response = client.chat.completions.create(
                            model="qwen-actor",
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": frozen["prompt"]},
                            ],
                            temperature=temperature,
                            top_p=1.0,
                            max_tokens=args.max_tokens,
                            extra_body={"guided_json": schema},
                        )
                        raw = response.choices[0].message.content or ""
                        parsed = parse_next_step(raw)
                        if parsed is None:
                            error = "next_step_parse_failed"
                    except Exception as exc:  # pragma: no cover - live server
                        error = f"{type(exc).__name__}: {exc}"
                    if error:
                        errors.append({"group_id": group_id, "temperature": temperature, "repetition": repetition, "error": error})
                    row = {
                        "group_id": group_id,
                        "temperature": temperature,
                        "repetition": repetition,
                        "prompt_sha256": frozen["prompt_sha256"],
                        "prompt_utf8_bytes": frozen["prompt_utf8_bytes"],
                        "actor_input_token_count": len(prompt_token_ids),
                        "actor_input_token_sha256": sha256_bytes(json.dumps(prompt_token_ids, separators=(",", ":")).encode()),
                        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
                        "raw_output": raw,
                        "raw_output_sha256": sha256_bytes(raw.encode("utf-8")),
                        "parsed": parsed,
                        "error": error,
                        "latency_seconds": round(time.perf_counter() - request_started, 4),
                        "request_model_id": "qwen-actor",
                    }
                    rows.append(row)
                    print(json.dumps({"group_id": group_id, "temperature": temperature, "repetition": repetition, "parsed": parsed is not None, "latency_seconds": row["latency_seconds"]}), flush=True)
    finally:
        monitor_stop.set()
        if monitor.is_alive():
            monitor.join(timeout=10)
        stop_server(process)

    aggregate = analyze_rows(rows, groups)
    result = {
        "schema_version": 1,
        "status": "complete" if len(rows) == 24 and not errors and all(row["parsed"] is not None for row in rows) else "failed",
        "mode": "actor_only_no_reward_no_downstream",
        "protocol": {
            "groups": list(GROUPS),
            "repetitions_per_group_temperature": 4,
            "temperatures": [0.0, 0.7],
            "generation_count": len(rows),
            "model": str(args.model),
            "serving_model_id": "qwen-actor",
            "base_serving_model_id": "qwen-base",
            "system_prompt": SYSTEM_PROMPT,
            "prompt_transport": "exact persisted action_predictor_1_prompt as user content in Chat Completions; system prompt unchanged",
            "top_p": 1.0,
            "top_k": {"temperature_0.0": 0, "temperature_0.7": 20},
            "seed": None,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "guided_json_schema": True,
            "next_step_parser": "agentflow.models.formatters.NextStep via json.loads + NextStep(**payload)",
            "downstream_invoked": False,
            "fixed_roles_invoked": False,
            "reward_invoked": False,
            "external_api_invoked": False,
        },
        "source": {
            "rollout_root": str(args.rollout_root),
            "rollout_root_file_count": sum(1 for _ in args.rollout_root.rglob("rollout_*.json")),
            "frozen_prompt_selection": "lexicographically first rollout filename within each requested idx group",
            "groups": groups,
        },
        "actor": {
            "model": model_info,
            "lora_snapshot": snapshot,
            "temporary_vllm_adapter": adapter_info,
            "route": "one local vLLM OpenAI-compatible server; qwen-base plus qwen-actor LoRA module",
        },
        "server": {
            "port": args.port,
            "vllm_log": str(vllm_log),
            "gpu_log": str(gpu_log),
            "command": server_info.get("command", []),
            "models_response": server_info.get("models_response"),
            "vllm_worker_multiproc_method": "spawn",
        },
        "rows": rows,
        "aggregate": aggregate,
        "errors": errors,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "generations": len(rows), "errors": len(errors), "output": str(args.output), "vllm_log": str(vllm_log)}), flush=True)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
