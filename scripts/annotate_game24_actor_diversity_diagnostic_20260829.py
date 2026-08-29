#!/usr/bin/env python3
"""Annotate a completed actor-only result with observed vLLM parameters.

No model or network call is made.  This exists because vLLM's generation
config can supply fields not explicitly sent by the OpenAI-compatible client;
the final result must report what the server actually used.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from pathlib import Path


SAMPLING_RE = re.compile(
    r"SamplingParams\(.*?repetition_penalty=([^,]+), temperature=([^,]+), "
    r"top_p=([^,]+), top_k=([^,]+), .*?max_tokens=([^,]+),"
)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=Path("log/2026-08-29_game24_actor_diversity_diagnostic_results.json"))
    parser.add_argument("--vllm-log", type=Path, default=Path("/root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829/vllm_actor.log"))
    parser.add_argument("--gpu-log", type=Path, default=Path("/root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829/gpu.tsv"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    observed = []
    default_chat = None
    default_completion = None
    adapter_loaded = False
    lora_request_count = 0
    qwen_base_request_count = 0
    post_count = 0
    for line in args.vllm_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Loaded new LoRA adapter" in line:
            adapter_loaded = True
        if "lora_request: LoRARequest(lora_name='qwen-actor'" in line:
            lora_request_count += 1
        if "base_model_name='qwen-base'" in line and "lora_request: LoRARequest" in line:
            qwen_base_request_count += 1
        if 'POST /v1/chat/completions' in line:
            post_count += 1
        if "Using default chat sampling params from model:" in line:
            default_chat = ast.literal_eval(line.split("from model:", 1)[1].strip())
        if "Using default completion sampling params from model:" in line:
            default_completion = ast.literal_eval(line.split("from model:", 1)[1].strip())
        if "params: SamplingParams" in line:
            match = SAMPLING_RE.search(line)
            if match:
                repetition, temperature, top_p, top_k, max_tokens = match.groups()
                observed.append({
                    "repetition_penalty": float(repetition),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "max_tokens": int(max_tokens),
                })
    unique_observed = sorted({json.dumps(item, sort_keys=True) for item in observed})
    result.setdefault("protocol", {})["vllm_observed_sampling_params"] = [json.loads(item) for item in unique_observed]
    result["protocol"]["vllm_model_generation_defaults"] = {
        "chat": default_chat,
        "completion": default_completion,
    }
    result["protocol"]["vllm_sampling_observation"] = (
        "Requests explicitly set top_p=1.0, temperature, and max_tokens. "
        "vLLM 0.9.2 retained model generation-config repetition_penalty=1.05 "
        "and top_k=20 for temperature=0.7; temperature=0.0 used greedy top_k=0."
    )
    result.setdefault("server", {})["vllm_log_sha256"] = __import__("hashlib").sha256(args.vllm_log.read_bytes()).hexdigest()
    result["server"]["http_chat_completion_post_count"] = post_count
    result["server"]["vllm_adapter_loaded_log_evidence"] = adapter_loaded
    result["server"]["qwen_actor_lora_request_count"] = lora_request_count
    result["server"]["qwen_base_lora_request_count"] = qwen_base_request_count
    gpu_rows = list(csv.DictReader(args.gpu_log.open(encoding="utf-8")))
    memory = [float(row["memory_used_mib"]) for row in gpu_rows if row.get("memory_used_mib")]
    result["server"]["gpu_monitor"] = {
        "log": str(args.gpu_log),
        "samples": len(gpu_rows),
        "peak_mib": max(memory) if memory else None,
        "last_mib_before_server_stop": memory[-1] if memory else None,
    }
    try:
        current_gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception as exc:
        current_gpu = f"unavailable:{type(exc).__name__}"
    result["server"]["post_run_gpu_query"] = current_gpu
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "annotated",
        "sampling_records": len(observed),
        "http_posts": post_count,
        "adapter_loaded": adapter_loaded,
        "gpu_peak_mib": max(memory) if memory else None,
        "post_run_gpu": current_gpu,
        "result": str(args.result),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
