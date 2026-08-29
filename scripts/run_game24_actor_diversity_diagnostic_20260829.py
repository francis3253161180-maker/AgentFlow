#!/usr/bin/env python3
"""Actor-only Game24 planner diversity diagnostic.

This deliberately bypasses AgentFlow's solver and all fixed roles.  It loads
the same Qwen actor LoRA snapshot used by the planner-temperature causal
smoke, serves one local vLLM actor endpoint, and sends only frozen
``action_predictor_1_prompt`` strings to it.  No reward, tool, verifier, or
memory code is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


GROUPS = ("43", "58", "181")
SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def load_frozen_prompts(rollout_root: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(rollout_root.rglob("rollout_*.json")):
        parts = path.parts
        if "idx_" not in path.parent.name:
            continue
        group_id = path.parent.name.removeprefix("idx_")
        if group_id in GROUPS:
            grouped[group_id].append(path)
    if set(grouped) != set(GROUPS) or any(len(items) != 4 for items in grouped.values()):
        raise RuntimeError({group: len(grouped.get(group, [])) for group in GROUPS})

    selected: dict[str, dict[str, Any]] = {}
    for group_id in GROUPS:
        source = sorted(grouped[group_id], key=lambda item: item.name)[0]
        data = json.loads(source.read_text(encoding="utf-8"))
        prompt = (data.get("total_result") or {}).get("action_predictor_1_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError(f"missing action_predictor_1_prompt: {source}")
        selected[group_id] = {
            "group_id": group_id,
            "source_file": source.name,
            "source_file_sha256": sha256_file(source),
            "prompt": prompt,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        }
    return selected


def snapshot_metadata(snapshot: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(snapshot, map_location="cpu", weights_only=False)
    state = payload.get("lora_state")
    if not isinstance(state, dict) or not payload.get("lora_hash"):
        raise RuntimeError("invalid LoRA behavior snapshot")
    return {
        "path": str(snapshot),
        "file_sha256": sha256_file(snapshot),
        "lora_hash": payload["lora_hash"],
        "tensor_count": len(state),
        "total_numel": sum(int(t.numel()) for t in state.values()),
    }


def prepare_peft_adapter(model_path: Path, snapshot: Path, adapter_dir: Path) -> dict[str, Any]:
    """Materialize the captured PEFT state as a temporary vLLM LoRA adapter."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    adapter_dir.mkdir(parents=True, exist_ok=True)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not (config_path.is_file() and weights_path.is_file()):
        payload = torch.load(snapshot, map_location="cpu", weights_only=False)
        state = payload.get("lora_state")
        if not isinstance(state, dict):
            raise RuntimeError("snapshot has no lora_state")
        base = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
        )
        model = get_peft_model(
            base,
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        loaded = model.load_state_dict(state, strict=False)
        unexpected = list(loaded.unexpected_keys)
        missing_lora = [key for key in loaded.missing_keys if ".lora_" in key]
        if unexpected or missing_lora:
            raise RuntimeError(f"adapter state mismatch unexpected={unexpected[:3]} missing={missing_lora[:3]}")
        model.save_pretrained(str(adapter_dir), safe_serialization=True)
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError("PEFT adapter materialization incomplete")
    # vLLM 0.9.2 resolves a LoRA-specific tokenizer from lora_path when the
    # request is present.  A weights-only PEFT directory can otherwise be
    # interpreted as a tiny fallback tokenizer, rejecting valid Qwen token
    # IDs.  Package the unchanged base tokenizer with the temporary adapter.
    if not (adapter_dir / "tokenizer_config.json").is_file():
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(str(model_path), local_files_only=True).save_pretrained(str(adapter_dir))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "path": str(adapter_dir),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_model_sha256": sha256_file(weights_path),
        "adapter_config": {key: config[key] for key in ("peft_type", "r", "lora_alpha", "target_modules") if key in config},
    }


def prepare_chat_template(model_path: Path, output_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeError("local Qwen tokenizer has no chat template")
    output_path.write_text(template, encoding="utf-8")
    return {"path": str(output_path), "sha256": sha256_file(output_path), "source": str(model_path)}


def model_inventory(model_path: Path) -> dict[str, Any]:
    files = []
    total = 0
    for path in sorted(model_path.iterdir()):
        if path.is_file():
            size = path.stat().st_size
            total += size
            files.append({"name": path.name, "bytes": size})
    config = model_path / "config.json"
    index = model_path / "model.safetensors.index.json"
    return {
        "path": str(model_path),
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "index_sha256": sha256_file(index) if index.is_file() else None,
    }


def wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    from urllib.request import urlopen

    deadline = time.monotonic() + timeout
    last_error = ""
    url = base_url.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM actor exited with code {process.returncode}; last={last_error}")
        try:
            with urlopen(url, timeout=3) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body
        except Exception as exc:  # pragma: no cover - startup timing
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
    raise TimeoutError(f"vLLM actor readiness timeout: {last_error}")


def start_server(args: argparse.Namespace, adapter: Path, chat_template: Path, log_path: Path) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    log_handle = log_path.open("wb")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--tokenizer",
        str(args.model),
        "--chat-template",
        str(chat_template),
        "--served-model-name",
        "qwen-base",
        "--enable-lora",
        "--lora-modules",
        json.dumps({"name": "qwen-actor", "path": str(adapter), "base_model_name": "qwen-base"}, separators=(",", ":")),
        "--max-lora-rank",
        "8",
        "--max-loras",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        "1",
        "--port",
        str(args.port),
        "--disable-log-stats",
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "0", "VLLM_USE_V1": "1"})
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    process._agentflow_log_handle = log_handle  # type: ignore[attr-defined]
    models = wait_for_server(f"http://127.0.0.1:{args.port}/v1", process, args.server_timeout)
    return process, {"command": command, "models_response": models}


def stop_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    handle = getattr(process, "_agentflow_log_handle", None)
    if handle is not None:
        handle.close()


def run_direct_vllm(
    args: argparse.Namespace,
    groups: dict[str, dict[str, Any]],
    adapter_dir: Path,
    gpu_log: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run the actor through vLLM's local LLM API, without an HTTP server.

    vLLM 0.9.2's OpenAI compatibility layer in this environment attempts to
    decode token-id prompts with an unrelated tokenizer class.  The local API
    accepts the same token IDs directly and still applies the production
    LoRARequest, avoiding that HTTP preprocessing defect while retaining the
    vLLM actor engine and LoRA execution path.
    """
    # The diagnostic tokenizes prompts and materializes the adapter before
    # constructing vLLM.  vLLM 0.9.2 defaults to fork for its V1 engine;
    # force spawn so a CUDA context cannot be inherited by EngineCore.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams
    from agentflow.models.formatters import NextStep

    schema = NextStep.model_json_schema() if hasattr(NextStep, "model_json_schema") else NextStep.schema()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    # The persisted action_predictor_1_prompt is already the complete
    # production actor prompt.  Tokenize that exact string without adding
    # special tokens, then pass the resulting IDs to vLLM.  This avoids both
    # rebuilding a chat wrapper and the vLLM 0.9.2 HTTP/string length bug.
    prompt_ids = {
        group_id: tokenizer.encode(frozen["prompt"], add_special_tokens=False)
        for group_id, frozen in groups.items()
    }
    request = LoRARequest(
        lora_name="qwen-actor",
        lora_int_id=1,
        lora_path=str(adapter_dir),
        base_model_name="qwen-base",
    )
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(gpu_log, monitor_stop), daemon=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    llm = None
    started = time.perf_counter()
    try:
        monitor.start()
        llm = LLM(
            model=str(args.model),
            tokenizer=str(args.model),
            tokenizer_mode="slow",
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_lora=True,
            max_lora_rank=8,
            max_loras=1,
            seed=0,
        )
        for group_id in GROUPS:
            frozen = groups[group_id]
            for temperature in (0.0, 0.7):
                params = SamplingParams(
                    repetition_penalty=1.05,
                    temperature=temperature,
                    top_p=1.0,
                    top_k=0 if temperature == 0.0 else 20,
                    max_tokens=args.max_tokens,
                    guided_decoding=GuidedDecodingParams(json=schema),
                )
                for repetition in range(1, 5):
                    request_started = time.perf_counter()
                    raw = ""
                    parsed = None
                    error = None
                    ids = prompt_ids[group_id]
                    try:
                        outputs = llm.generate(
                            prompt_token_ids=ids,
                            sampling_params=params,
                            use_tqdm=False,
                            lora_request=request,
                        )
                        raw = outputs[0].outputs[0].text or ""
                        parsed = parse_next_step(raw)
                        if parsed is None:
                            error = "next_step_parse_failed"
                    except Exception as exc:  # pragma: no cover - live engine failure
                        error = f"{type(exc).__name__}: {exc}"
                    if error:
                        errors.append({"group_id": group_id, "temperature": temperature, "repetition": repetition, "error": error})
                    row = {
                        "group_id": group_id,
                        "temperature": temperature,
                        "repetition": repetition,
                        "prompt_sha256": frozen["prompt_sha256"],
                        "prompt_utf8_bytes": frozen["prompt_utf8_bytes"],
                        "actor_input_token_count": len(ids),
                        "actor_input_token_sha256": sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
                        "system_prompt_sha256": None,
                        "raw_output": raw,
                        "raw_output_sha256": sha256_bytes(raw.encode("utf-8")),
                        "parsed": parsed,
                        "error": error,
                        "latency_seconds": round(time.perf_counter() - request_started, 4),
                        "lora_request": {
                            "lora_name": request.lora_name,
                            "lora_int_id": request.lora_int_id,
                            "base_model_name": request.base_model_name,
                            "adapter_path": str(adapter_dir),
                        },
                    }
                    rows.append(row)
                    print(json.dumps({"group_id": group_id, "temperature": temperature, "repetition": repetition, "parsed": parsed is not None, "latency_seconds": row["latency_seconds"]}), flush=True)
    finally:
        # All requests above are synchronous. Sleep is safe after completion.
        if llm is not None:
            try:
                llm.sleep(level=2)
            except Exception as exc:  # preserve evidence but do not hide run errors
                errors.append({"cleanup": "llm.sleep(level=2)", "error": f"{type(exc).__name__}: {exc}"})
            del llm
        monitor_stop.set()
        if monitor.is_alive():
            monitor.join(timeout=10)
    return rows, {
        "backend": "vllm.LLM.generate",
        "route": "one local vLLM LLM engine; direct token-id prompts; qwen-actor LoRARequest",
        "model_id": "qwen-base",
        "request_model_id": "qwen-actor",
        "tokenizer_mode": "slow",
        "lora_request": {
            "lora_name": request.lora_name,
            "lora_int_id": request.lora_int_id,
            "base_model_name": request.base_model_name,
            "adapter_path": str(adapter_dir),
        },
        "sampling_params": {
            "repetition_penalty": 1.05,
            "top_p": 1.0,
            "top_k": {"0.0": 0, "0.7": 20},
            "max_tokens": args.max_tokens,
            "guided_json": True,
            "seed": 0,
        },
        "cleanup": "synchronous requests completed; llm.sleep(level=2) called before destruction",
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }, errors


def gpu_monitor(path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("timestamp,gpu_index,memory_used_mib,utilization_gpu\n")
        while not stop.is_set():
            stamp = time.time()
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                ).strip()
            except Exception:
                value = ""
            for line in value.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 3:
                    output.write(f"{stamp:.3f},{','.join(fields)}\n")
            output.flush()
            stop.wait(1.0)


def normalize_semantic_text(value: str) -> str:
    """Transparent lexical normalization for semantic-action grouping.

    This is not an embedding or judge.  It lowercases/NFKC-normalizes text,
    maps a few direct action synonyms, removes punctuation and boilerplate,
    and preserves numbers/tool-specific content.
    """
    text = unicodedata.normalize("NFKC", value).lower()
    synonyms = {
        "generate": "make", "create": "make", "construct": "make",
        "produce": "make", "find": "make", "calculate": "solve",
        "determine": "solve", "evaluate": "equal", "evaluates": "equal", "equals": "equal",
        "expression": "expr", "expressions": "expr",
    }
    tokens = re.findall(r"[a-z0-9]+", text)
    boilerplate = {
        "a", "an", "the", "one", "valid", "using", "use", "uses", "each", "every",
        "number", "numbers", "that", "to", "and", "with", "only", "exactly", "once",
        "step", "by", "solution", "series", "of", "from", "basic", "operations",
        "potentially", "possible", "different", "combinations", "given", "systematically",
        "verify", "explore", "make", "achieve", "target", "value", "arithmetic",
    }
    normalized = [synonyms.get(token, token) for token in tokens]
    normalized = [token for token in normalized if token not in boilerplate]
    # The remaining order is often just a paraphrase choice ("arithmetic
    # expression" vs "expression ... arithmetic").  A sorted content-token
    # signature makes that distinction explicit without embeddings or a
    # semantic judge.
    return " ".join(sorted(normalized))


def parse_next_step(raw: str) -> dict[str, str] | None:
    from agentflow.models.formatters import NextStep

    try:
        parsed = json.loads(raw)
        item = NextStep(**parsed)
        return {
            "justification": item.justification,
            "context": item.context,
            "sub_goal": item.sub_goal,
            "tool_name": item.tool_name,
        }
    except Exception:
        return None


def analyze_rows(rows: list[dict[str, Any]], groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[f"{row['group_id']}/temperature_{row['temperature']}"] .append(row)
    aggregate = {}
    for key, items in sorted(by_condition.items()):
        parsed = [item["parsed"] for item in items if item["parsed"] is not None]
        semantic = {
            f"{item['tool_name'].strip().lower()}::{normalize_semantic_text(item['sub_goal'])}"
            for item in parsed
        }
        aggregate[key] = {
            "group_id": items[0]["group_id"],
            "temperature": items[0]["temperature"],
            "repetitions": len(items),
            "prompt_sha256_values": sorted({item["prompt_sha256"] for item in items}),
            "prompt_sha256_identical": len({item["prompt_sha256"] for item in items}) == 1,
            "raw_outputs": [item["raw_output"] for item in items],
            "raw_output_sha256": [item["raw_output_sha256"] for item in items],
            "raw_output_unique_count": len({item["raw_output"] for item in items}),
            "byte_identical_repeats": len({item["raw_output"] for item in items}) == 1,
            "parsed_valid_count": len(parsed),
            "unique_tool_names": sorted({item["tool_name"] for item in parsed}),
            "unique_sub_goals": sorted({item["sub_goal"] for item in parsed}),
            "unique_contexts": sorted({item["context"] for item in parsed}),
            "semantic_action_unique_count": len(semantic),
            "semantic_action_values": sorted(semantic),
            "semantic_normalization": "NFKC/lowercase; synonym map generate/create/construct/produce/find->make, calculate/determine->solve, expression(s)->expr; punctuation removal; boilerplate removal; numbers preserved",
            "latency_seconds": [item["latency_seconds"] for item in items],
        }
    for group_id in GROUPS:
        if groups[group_id]["prompt_sha256"] != groups[group_id]["prompt_sha256"]:
            raise AssertionError(group_id)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=Path("rollout_data/46.38.243.197/gameof24-planner-temp0-causal-sanity-20260829_20260829-135524"))
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--snapshot", type=Path, default=Path("/root/autodl-tmp/tmp/gameof24_planner_temp0_causal_sanity_20260829/gameof24-planner-temp0-causal-sanity-20260829_20260829_135323_behavior_snapshot.pt"))
    parser.add_argument("--work-dir", type=Path, default=Path("/root/autodl-tmp/tmp/game24_actor_diversity_diagnostic_20260829"))
    parser.add_argument("--output", type=Path, default=Path("log/2026-08-29_game24_actor_diversity_diagnostic_results.json"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGENTFLOW_ACTOR_DIAGNOSTIC_PORT", "18080")))
    parser.add_argument("--max-model-len", type=int, default=4096)
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
    chat_template_info = prepare_chat_template(args.model, args.work_dir / "qwen_chat_template.jinja")
    vllm_log = args.work_dir / "vllm_actor.log"
    gpu_log = args.work_dir / "gpu.tsv"
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(gpu_log, monitor_stop), daemon=True)
    process = None
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    server_info: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    try:
        process, server_info = start_server(args, args.work_dir / "qwen-actor-lora", Path(chat_template_info["path"]), vllm_log)
        monitor.start()
        from openai import OpenAI
        from transformers import AutoTokenizer
        from agentflow.models.formatters import NextStep

        schema = NextStep.model_json_schema() if hasattr(NextStep, "model_json_schema") else NextStep.schema()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="local-diagnostic-token", timeout=180)
        for group_id in GROUPS:
            frozen = groups[group_id]
            for temperature in (0.0, 0.7):
                for repetition in range(1, 5):
                    request_started = time.perf_counter()
                    raw = ""
                    parsed = None
                    error = None
                    prompt_token_ids = None
                    try:
                        prompt_token_ids = tokenizer.apply_chat_template(
                            [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": frozen["prompt"]},
                            ],
                            tokenize=True,
                            add_generation_prompt=True,
                        )
                        response = client.completions.create(
                            model="qwen-actor",
                            prompt=prompt_token_ids,
                            temperature=temperature,
                            top_p=1.0,
                            max_tokens=args.max_tokens,
                            extra_body={"guided_json": schema},
                        )
                        raw = response.choices[0].message.content or ""
                        parsed = parse_next_step(raw)
                        if parsed is None:
                            error = "next_step_parse_failed"
                            errors.append({"group_id": group_id, "temperature": temperature, "repetition": repetition, "error": error})
                    except Exception as exc:  # pragma: no cover - live service failures
                        error = f"{type(exc).__name__}: {exc}"
                        errors.append({"group_id": group_id, "temperature": temperature, "repetition": repetition, "error": error})
                    row = {
                        "group_id": group_id,
                        "temperature": temperature,
                        "repetition": repetition,
                        "prompt_sha256": frozen["prompt_sha256"],
                        "prompt_utf8_bytes": frozen["prompt_utf8_bytes"],
                        "actor_input_token_count": len(prompt_token_ids) if prompt_token_ids is not None else None,
                        "actor_input_token_sha256": sha256_bytes(json.dumps(prompt_token_ids, separators=(",", ":")).encode()) if prompt_token_ids is not None else None,
                        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode()),
                        "raw_output": raw,
                        "raw_output_sha256": sha256_bytes(raw.encode("utf-8")),
                        "parsed": parsed,
                        "error": error,
                        "latency_seconds": round(time.perf_counter() - request_started, 4),
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
            "top_p": 1.0,
            "top_k": "vLLM model-generation default; see vllm_observed_sampling_params",
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
            "chat_template": chat_template_info,
            "route": "one local vLLM process; qwen-actor requests attach the only LoRA adapter",
        },
        "server": {
            "port": args.port,
            "vllm_log": str(vllm_log),
            "gpu_log": str(gpu_log),
            "command": server_info.get("command", []),
            "models_response": server_info.get("models_response"),
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
