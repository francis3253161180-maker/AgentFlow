#!/usr/bin/env python3
"""Run exactly three local Qwen-only typed Game24 rollout smokes.

This is intentionally independent of the AgentFlow multi-role solver: only
the local planner model is called, and all state transitions/rewards are
performed by ``agentflow.models.game24_atomic``.  No network client is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentflow.models.game24_atomic import (
    AtomicState,
    extract_game24_numbers,
    parse_atomic_action,
)


QUESTIONS = (
    "Using the numbers [2,3,4,8], create an expression that equals 24.",
    "Using the numbers [1,4,6,6], create an expression that equals 24.",
    "Using the numbers [1,7,8,12], create an expression that equals 24.",
)


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _load_local_planner(model_path: str, snapshot_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    adapter_loaded = False
    snapshot_hash = None
    if snapshot_path:
        payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        lora_state = payload.get("lora_state")
        if not isinstance(lora_state, dict):
            raise RuntimeError("LoRA snapshot has no lora_state")
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # The snapshot was captured from a PEFT model whose keys already
        # contain the ``default`` adapter name.  PEFT's convenience helper in
        # this environment appends that name a second time, so use the native
        # module loader and verify every LoRA key explicitly.
        result = model.load_state_dict(lora_state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing_lora = [key for key in result.missing_keys if ".lora_" in key]
        if unexpected or missing_lora:
            raise RuntimeError(
                f"LoRA snapshot key mismatch: unexpected={unexpected[:3]} missing_lora={missing_lora[:3]}"
            )
        adapter_loaded = True
        snapshot_hash = payload.get("lora_hash")
    model.eval()
    return tokenizer, model, adapter_loaded, snapshot_hash


def _prompt(question: str, state: AtomicState, feedback: str | None = None) -> str:
    feedback_text = feedback or "none"
    return (
        "Return exactly one JSON object and nothing else.\n"
        'Schema: {"left_id":"nX","operator":"+|-|*|/","right_id":"nY"}\n'
        "Choose two distinct currently active node IDs. The divisor must be nonzero.\n"
        f"Original puzzle: {question}\n"
        f"Current active nodes: {json.dumps(state.snapshot(), sort_keys=True)}\n"
        f"Previous deterministic feedback: {feedback_text}\n"
        "Make exactly one next combine action. Do not write an expression or prose."
    )


def _generate_action(tokenizer, model, prompt: str) -> tuple[str, int, int]:
    messages = [
        {"role": "system", "content": "You are the Game24 planner. Output only the requested typed JSON action."},
        {"role": "user", "content": prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    encoded = encoded.to(device)
    with torch.inference_mode():
        output = model.generate(
            encoded,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=96,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, encoded.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, int(encoded.shape[-1]), int(generated.shape[-1])


def run_rollout(question: str, tokenizer, model) -> dict[str, object]:
    started = time.perf_counter()
    state = AtomicState(extract_game24_numbers(question))
    steps: list[dict[str, object]] = []
    failure_category = None
    for step in range(1, 4):
        feedback = None
        step_record: dict[str, object] = {"step": step, "state_before": state.snapshot(), "attempts": []}
        committed = False
        for attempt in range(1, 3):
            raw, prompt_tokens, response_tokens = _generate_action(tokenizer, model, _prompt(question, state, feedback))
            attempt_record: dict[str, object] = {
                "attempt": attempt,
                "planner_json": raw,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
            }
            try:
                action = parse_atomic_action(raw)
                node = state.apply(action)
                attempt_record.update({"status": "committed", "action": action.model_dump(), "result": {
                    "node_id": node.node_id, "value": str(node.value),
                    "expression": node.expression, "provenance": list(node.provenance),
                }})
                step_record["attempts"].append(attempt_record)
                step_record["state_after"] = state.snapshot()
                committed = True
                break
            except (ValueError, TypeError) as exc:
                message = str(exc)
                category = "INVALID ACTION" if "active" in message or "distinct" in message or "division" in message else "FORMAT/SCHEMA"
                attempt_record.update({"status": "rejected", "failure": message, "category": category})
                step_record["attempts"].append(attempt_record)
                feedback = f"Rejected action: {message}. State is unchanged; emit one valid typed action."
                failure_category = category
        steps.append(step_record)
        if not committed:
            return {
                "question": question,
                "initial_state": AtomicState(extract_game24_numbers(question)).snapshot(),
                "steps": steps,
                "final_expression": None,
                "final_value": None,
                "reward": 0,
                "failure_category": failure_category or "SEARCH/ARITHMETIC STRATEGY",
                "wall_time_seconds": round(time.perf_counter() - started, 3),
            }
    node = next(iter(state.active.values()))
    reward = state.terminal_reward()
    return {
        "question": question,
        "initial_state": AtomicState(extract_game24_numbers(question)).snapshot(),
        "steps": steps,
        "final_expression": node.expression,
        "final_value": str(node.value),
        "final_provenance": list(node.provenance),
        "reward": reward,
        "failure_category": None if reward else "SEARCH/ARITHMETIC STRATEGY",
        "wall_time_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--snapshot", default="/root/autodl-tmp/tmp/random30_fresh_rollout_replay_20260828/random30-fresh-rollout-replay-20260828_20260828_115632_behavior_snapshot.pt")
    parser.add_argument("--output", default="log/2026-08-28_game24_atomic_qwen_only_smoke_results.json")
    parser.add_argument("--trace", default="log/2026-08-28_game24_atomic_qwen_only_smoke_trace.json")
    args = parser.parse_args()

    os.environ["AGENTFLOW_DISABLE_EXTERNAL_LLM"] = "1"
    os.environ.pop("AGENTFLOW_UNIFIED_FIXED_ROLE_ENGINE", None)
    os.environ.pop("ARK_API_KEY", None)
    if not Path(args.model).is_dir():
        raise SystemExit(f"local model path missing: {args.model}")
    if args.snapshot and not Path(args.snapshot).is_file():
        raise SystemExit(f"LoRA snapshot missing: {args.snapshot}")
    t0 = time.perf_counter()
    tokenizer, model, adapter_loaded, snapshot_hash = _load_local_planner(args.model, args.snapshot)
    rollouts = [run_rollout(question, tokenizer, model) for question in QUESTIONS]
    gpu_peak = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "questions": list(QUESTIONS), "rollout_count": 3, "planner_model": args.model,
            "planner_temperature": 0.7, "typed_action_steps": 3, "max_structured_retries_per_step": 1,
            "external_llm_disabled": True, "optimizer_steps": 0, "checkpoint": False,
            "fixed_roles": "not touched in isolated atomic path",
            "lora_snapshot": args.snapshot, "lora_loaded": adapter_loaded, "lora_hash": snapshot_hash,
        },
        "rollouts": rollouts,
        "summary": {
            "reward_sum": sum(int(item["reward"]) for item in rollouts),
            "reward_mean": sum(int(item["reward"]) for item in rollouts) / 3,
            "format_schema_failures": sum(1 for item in rollouts if item.get("failure_category") == "FORMAT/SCHEMA"),
            "invalid_action_failures": sum(1 for item in rollouts if item.get("failure_category") == "INVALID ACTION"),
            "search_arithmetic_failures": sum(1 for item in rollouts if item.get("failure_category") == "SEARCH/ARITHMETIC STRATEGY"),
        },
        "runtime_seconds": round(time.perf_counter() - t0, 3),
        "gpu_peak_memory_mib": gpu_peak,
        "code_commit": os.popen("git rev-parse HEAD").read().strip(),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.trace).write_text(json.dumps({"schema_version": 1, "rollouts": rollouts}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "rollouts": 3, "reward_sum": result["summary"]["reward_sum"], "output": args.output, "trace": args.trace, "gpu_peak_memory_mib": gpu_peak}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
