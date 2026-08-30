#!/usr/bin/env python3
"""Actor-only two-mode offline MuSiQue protocol smoke.

The local vLLM Qwen actor with one LoRA request is the sole semantic policy.
The environment performs only local retrieval, schema/provenance validation,
budgeting, transition persistence, and deterministic scoring.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import (
    ACTOR_CONTEXT_TOKENS,
    DECISION_ADAPTER,
    DECISION_MAX_NEW_TOKENS,
    DECISION_SYSTEM,
    EVIDENCE_MAX_NEW_TOKENS,
    EVIDENCE_SYSTEM,
    MAX_DECISION_TRANSITIONS,
    MAX_SEARCH_ACTIONS,
    BGEEncoder,
    CompactMemory,
    EvidenceUpdate,
    LocalCorpusSearch,
    OfflineCorpus,
    decision_prompt,
    evidence_prompt,
    parse_decision,
    parse_evidence_update,
    sha256_file,
    stable_json_hash,
    terminal_reward,
    transition_record,
)


SEED = 20260830


def stratified_subset(corpus: OfflineCorpus, size: int, seed: int) -> list[str]:
    by_hop: dict[int, list[str]] = defaultdict(list)
    for qid, row in corpus.questions.items():
        by_hop[row.hop_count].append(qid)
    hops = [hop for hop in (2, 3, 4) if by_hop[hop]]
    rng = random.Random(seed)
    for values in by_hop.values():
        values.sort()
        rng.shuffle(values)
    allocations = {hop: size // len(hops) for hop in hops}
    for hop in hops[: size % len(hops)]:
        allocations[hop] += 1
    selected = [qid for hop in hops for qid in by_hop[hop][: allocations[hop]]]
    if len(selected) != size:
        raise RuntimeError("stratified subset underfilled")
    rng.shuffle(selected)
    return selected


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def render_chat(tokenizer: Any, system: str, full_prompt: str) -> tuple[str, list[int]]:
    marker = "\nCurrent input:\n"
    if marker not in full_prompt or not full_prompt.startswith(system):
        raise ValueError("prompt/system protocol mismatch")
    user = full_prompt.split(marker, 1)[1]
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return rendered, ids


def gpu_monitor(path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("timestamp,gpu_index,memory_used_mib,utilization_gpu\n")
        while not stop.is_set():
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
            except Exception:
                value = ""
            for row in value.splitlines():
                fields = [part.strip() for part in row.split(",")]
                if len(fields) == 3:
                    output.write(f"{time.time():.3f},{','.join(fields)}\n")
            output.flush()
            stop.wait(1)


def read_gpu_peak(path: Path) -> int:
    values = []
    if path.is_file():
        for line in path.read_text().splitlines()[1:]:
            fields = line.split(",")
            if len(fields) >= 3:
                values.append(int(fields[2]))
    return max(values, default=0)


@dataclass
class Trajectory:
    qid: str
    rollout_index: int
    rollout_id: str
    memory: CompactMemory = field(default_factory=CompactMemory)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    searches: int = 0
    decisions: int = 0
    done: bool = False
    final_answer: str = ""
    termination_reason: str = ""
    retrieved_pids: set[str] = field(default_factory=set)
    selected_pids: set[str] = field(default_factory=set)
    query_sequence: list[str] = field(default_factory=list)
    proposed_selections: int = 0
    rejected_selections: list[dict[str, str]] = field(default_factory=list)
    generated_tokens: Counter = field(default_factory=Counter)
    max_compact_memory_tokens: int = 0
    started: float = field(default_factory=time.perf_counter)
    ended: float = 0.0
    reward_detail: dict[str, Any] = field(default_factory=dict)

    @property
    def trajectory_id(self) -> str:
        return f"{self.qid}::{self.rollout_id}"


def finish(trajectory: Trajectory, corpus: OfflineCorpus, reason: str, answer: str = "") -> None:
    trajectory.done = True
    trajectory.final_answer = answer
    trajectory.termination_reason = reason
    trajectory.ended = time.perf_counter()
    if reason == "answer":
        trajectory.reward_detail = terminal_reward(corpus, trajectory.qid, answer, trajectory.selected_pids)
    else:
        trajectory.reward_detail = {
            "reward": 0,
            "answer_em": False,
            "full_selected_support_coverage": False,
            "selected_support_count": len(trajectory.selected_pids & corpus.scorer_record(trajectory.qid).support_pids),
            "gold_support_count": len(corpus.scorer_record(trajectory.qid).support_pids),
        }


def generate_batch(llm: Any, prompts: list[list[int]], params: Any, request: Any) -> tuple[list[str], list[dict[str, Any]]]:
    if not prompts:
        return [], []
    started = time.perf_counter()
    outputs = llm.generate(prompt_token_ids=prompts, sampling_params=params, use_tqdm=False, lora_request=request)
    batch_wall = time.perf_counter() - started
    texts = []
    metadata = []
    for output in outputs:
        generated = output.outputs[0]
        texts.append(generated.text or "")
        metadata.append(
            {
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": len(generated.token_ids),
                "cumulative_logprob": float(generated.cumulative_logprob) if generated.cumulative_logprob is not None else None,
                "finish_reason": generated.finish_reason,
                "batch_size": len(prompts),
                "batch_wall_seconds": batch_wall,
            }
        )
    return texts, metadata


def summarize(corpus: OfflineCorpus, trajectories: list[Trajectory], wall_seconds: float, gpu_peak: int) -> dict[str, Any]:
    transitions = [row for trajectory in trajectories for row in trajectory.transitions]
    by_mode = {mode: [row for row in transitions if row["mode"] == mode] for mode in ("DECISION", "EVIDENCE_UPDATE")}
    schema_failures = {
        mode: sum(row["validation_result"].get("format_failure", False) for row in rows)
        for mode, rows in by_mode.items()
    }
    schema_totals = {mode: len(rows) for mode, rows in by_mode.items()}
    schema_valid = {
        mode: (schema_totals[mode] - schema_failures[mode]) / max(schema_totals[mode], 1)
        for mode in by_mode
    }
    proposed = sum(t.proposed_selections for t in trajectories)
    invalid = sum(
        reason["reason"] in {"pid_not_in_current_observation", "quote_not_exact_normalized_substring"}
        for t in trajectories
        for reason in t.rejected_selections
    )
    selected_count = sum(len(t.memory.evidence) for t in trajectories)
    support_retrieval = []
    support_selection = []
    distractor_selected = 0
    repeated = 0
    total_queries = 0
    no_useful = 0
    answer_count = 0
    premature = 0
    latencies = []
    for trajectory in trajectories:
        support = corpus.scorer_record(trajectory.qid).support_pids
        support_retrieval.append(len(trajectory.retrieved_pids & support) / len(support))
        support_selection.append(len(trajectory.selected_pids & support) / len(support))
        distractor_selected += len(trajectory.selected_pids - support)
        normalized_queries = [normalize_query(query) for query in trajectory.query_sequence]
        repeated += len(normalized_queries) - len(set(normalized_queries))
        total_queries += len(normalized_queries)
        no_useful += sum(row.outcome == "no_useful_evidence" for row in trajectory.memory.search_history)
        if trajectory.termination_reason == "answer":
            answer_count += 1
            premature += not trajectory.reward_detail["full_selected_support_coverage"]
        latencies.append(trajectory.ended - trajectory.started)
    query_signatures = Counter(tuple(normalize_query(q) for q in t.query_sequence) for t in trajectories)
    reward_vector = [int(t.reward_detail["reward"]) for t in trajectories]
    by_group = defaultdict(list)
    for trajectory in trajectories:
        by_group[trajectory.qid].append(int(trajectory.reward_detail["reward"]))
    mixed_groups = sum(len(set(values)) > 1 for values in by_group.values())
    mode_tokens = {
        mode: [row["token_logprob_metadata"].get("generated_tokens", 0) for row in rows]
        for mode, rows in by_mode.items()
    }
    all_schema_valid = sum(schema_totals.values()) - sum(schema_failures.values())
    return {
        "trajectory_count": len(trajectories),
        "transition_count": len(transitions),
        "schema": {
            mode: {"total": schema_totals[mode], "failures": schema_failures[mode], "valid_rate": schema_valid[mode]}
            for mode in by_mode
        },
        "overall_schema_valid_rate": all_schema_valid / max(sum(schema_totals.values()), 1),
        "invalid_quote_or_pid_count": invalid,
        "proposed_selection_count": proposed,
        "invalid_quote_or_pid_rate": invalid / max(proposed, 1),
        "average_selected_evidence_count_per_trajectory": selected_count / len(trajectories),
        "average_selected_evidence_count_per_evidence_transition": selected_count / max(len(by_mode["EVIDENCE_UPDATE"]), 1),
        "gold_support_retrieval_recall": sum(support_retrieval) / len(support_retrieval),
        "gold_support_selection_recall": sum(support_selection) / len(support_selection),
        "distractor_selection_rate": distractor_selected / max(selected_count, 1),
        "repeated_query_rate": repeated / max(total_queries, 1),
        "no_useful_evidence_rate": no_useful / max(total_queries, 1),
        "premature_answer_rate": premature / max(answer_count, 1),
        "answer_action_rate": answer_count / len(trajectories),
        "answer_em": sum(t.reward_detail["answer_em"] for t in trajectories) / len(trajectories),
        "full_selected_support_coverage": sum(t.reward_detail["full_selected_support_coverage"] for t in trajectories) / len(trajectories),
        "grounded_terminal_reward_mean": sum(reward_vector) / len(reward_vector),
        "reward_vector": reward_vector,
        "reward_counts": dict(sorted(Counter(reward_vector).items())),
        "mixed_reward_groups": mixed_groups,
        "unique_query_sequence_signatures": len(query_signatures),
        "largest_query_signature_multiplicity": max(query_signatures.values(), default=0),
        "mean_generated_tokens": {
            mode: sum(values) / max(len(values), 1) for mode, values in mode_tokens.items()
        },
        "trajectory_latency_seconds": {"mean": sum(latencies) / len(latencies), "p95": percentile(latencies, 0.95)},
        "wall_seconds": wall_seconds,
        "rollouts_per_minute": len(trajectories) / wall_seconds * 60,
        "gpu_peak_memory_mib": gpu_peak,
        "max_compact_memory_tokens": max((t.max_compact_memory_tokens for t in trajectories), default=0),
        "termination_reasons": dict(sorted(Counter(t.termination_reason for t in trajectories).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
    parser.add_argument("--bge-model", type=Path, default=Path("/root/autodl-tmp/models/bge-small-en-v1.5"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--iteration-id", default="phase_c_v1")
    parser.add_argument("--hypothesis", default="the strict two-mode Qwen+LoRA protocol produces valid diverse trajectories and a nondegenerate grounded binary reward distribution")
    parser.add_argument("--changed-variable", action="append", default=[])
    parser.add_argument("--before-results", type=Path)
    args = parser.parse_args()

    os.environ.update(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    for external_key in (
        "HF_TOKEN",
        "OPENAI_API_KEY",
        "ARK_API_KEY",
        "DASHSCOPE_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        os.environ.pop(external_key, None)
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    for path in (args.corpus, args.embeddings, args.model, args.bge_model, args.adapter):
        if not path.exists():
            raise SystemExit(f"missing path: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    args.gpu_log.parent.mkdir(parents=True, exist_ok=True)

    corpus = OfflineCorpus.load(args.corpus)
    qids = stratified_subset(corpus, args.subset_size, args.seed)
    # Query encoding is explicitly CPU-only even while vLLM holds the GPU.
    query_encoder = BGEEncoder(args.bge_model, device="cpu")
    retriever = LocalCorpusSearch(corpus, args.embeddings, query_encoder, top_k=2, rrf_k=60)
    del query_encoder
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    trajectories = [
        Trajectory(qid=qid, rollout_index=index, rollout_id=f"{qid}__r{index}")
        for qid in qids
        for index in range(args.n)
    ]
    prompts_audit: dict[str, str] = {}
    observation_audit: dict[str, Any] = {}
    monitor_stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(args.gpu_log, monitor_stop), daemon=True)
    llm = None
    cleanup_errors = []
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
            max_model_len=ACTOR_CONTEXT_TOKENS,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=ACTOR_CONTEXT_TOKENS * args.max_num_seqs,
            enable_lora=True,
            max_lora_rank=8,
            max_loras=1,
            seed=args.seed,
        )
        request = LoRARequest("qwen-actor", 1, str(args.adapter), base_model_name="qwen-base")
        decision_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            top_k=20,
            repetition_penalty=1.05,
            max_tokens=DECISION_MAX_NEW_TOKENS,
            logprobs=1,
        )
        evidence_params = SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            top_k=20,
            repetition_penalty=1.05,
            max_tokens=EVIDENCE_MAX_NEW_TOKENS,
            stop=["]}"],
            include_stop_str_in_output=True,
            logprobs=1,
        )

        while any(not trajectory.done for trajectory in trajectories):
            active = [trajectory for trajectory in trajectories if not trajectory.done]
            decision_items = []
            for trajectory in active:
                if trajectory.decisions >= MAX_DECISION_TRANSITIONS:
                    finish(trajectory, corpus, "decision_budget_exhausted")
                    continue
                full = decision_prompt(
                    corpus.questions[trajectory.qid].question,
                    trajectory.memory,
                    searches_left=MAX_SEARCH_ACTIONS - trajectory.searches,
                    decisions_left=MAX_DECISION_TRANSITIONS - trajectory.decisions,
                )
                current_payload = json.loads(full.split("\nCurrent input:\n", 1)[1])
                forbidden = {"answer", "answers", "answer_aliases", "is_supporting", "support_pids", "question_decomposition", "paragraph_support_idx"}
                if forbidden & set(current_payload):
                    raise RuntimeError(f"scorer-label leakage in DECISION input: {forbidden & set(current_payload)}")
                memory_tokens = len(tokenizer.encode(json.dumps(trajectory.memory.actor_payload(), ensure_ascii=False), add_special_tokens=False))
                trajectory.max_compact_memory_tokens = max(trajectory.max_compact_memory_tokens, memory_tokens)
                rendered, ids = render_chat(tokenizer, DECISION_SYSTEM, full)
                if len(ids) + DECISION_MAX_NEW_TOKENS > ACTOR_CONTEXT_TOKENS:
                    finish(trajectory, corpus, "context_overflow")
                    continue
                prompt_hash = hashlib.sha256(rendered.encode()).hexdigest()
                prompts_audit[prompt_hash] = rendered
                decision_items.append((trajectory, rendered, ids))
            texts, token_rows = generate_batch(llm, [item[2] for item in decision_items], decision_params, request)
            pending_searches = []
            for (trajectory, rendered, _ids), raw, tokens in zip(decision_items, texts, token_rows):
                memory_before = trajectory.memory.actor_payload()
                trajectory.decisions += 1
                trajectory.generated_tokens["DECISION"] += tokens["generated_tokens"]
                try:
                    action = parse_decision(raw)
                except ValidationError as exc:
                    trajectory.transitions.append(
                        transition_record(
                            trajectory_id=trajectory.trajectory_id,
                            qid=trajectory.qid,
                            rollout_id=trajectory.rollout_id,
                            transition_index=len(trajectory.transitions),
                            mode="DECISION",
                            prompt=rendered,
                            response=raw,
                            compact_memory_before=memory_before,
                            semantic_output=None,
                            observation_refs=[],
                            validation_result={"format_failure": True, "error": str(exc)},
                            compact_memory_after=trajectory.memory.actor_payload(),
                            token_metadata=tokens,
                        )
                    )
                    finish(trajectory, corpus, "format_failure_decision")
                    continue
                semantic = action.model_dump()
                validation = {"format_failure": False, "schema_valid": True, "budget_valid": True}
                if action.action == "answer":
                    trajectory.transitions.append(
                        transition_record(
                            trajectory_id=trajectory.trajectory_id,
                            qid=trajectory.qid,
                            rollout_id=trajectory.rollout_id,
                            transition_index=len(trajectory.transitions),
                            mode="DECISION",
                            prompt=rendered,
                            response=raw,
                            compact_memory_before=memory_before,
                            semantic_output=semantic,
                            observation_refs=[],
                            validation_result=validation,
                            compact_memory_after=trajectory.memory.actor_payload(),
                            token_metadata=tokens,
                        )
                    )
                    finish(trajectory, corpus, "answer", action.answer)
                    continue
                if trajectory.searches >= MAX_SEARCH_ACTIONS:
                    validation["budget_valid"] = False
                    trajectory.transitions.append(
                        transition_record(
                            trajectory_id=trajectory.trajectory_id,
                            qid=trajectory.qid,
                            rollout_id=trajectory.rollout_id,
                            transition_index=len(trajectory.transitions),
                            mode="DECISION",
                            prompt=rendered,
                            response=raw,
                            compact_memory_before=memory_before,
                            semantic_output=semantic,
                            observation_refs=[],
                            validation_result=validation,
                            compact_memory_after=trajectory.memory.actor_payload(),
                            token_metadata=tokens,
                        )
                    )
                    finish(trajectory, corpus, "search_budget_exceeded")
                    continue
                trajectory.searches += 1
                trajectory.query_sequence.append(action.query)
                observation = retriever.search(trajectory.qid, action.query)
                observation_ref = stable_json_hash(observation)
                observation_audit[observation_ref] = observation
                trajectory.retrieved_pids.update(row["pid"] for row in observation)
                trajectory.transitions.append(
                    transition_record(
                        trajectory_id=trajectory.trajectory_id,
                        qid=trajectory.qid,
                        rollout_id=trajectory.rollout_id,
                        transition_index=len(trajectory.transitions),
                        mode="DECISION",
                        prompt=rendered,
                        response=raw,
                        compact_memory_before=memory_before,
                        semantic_output=semantic,
                        observation_refs=[observation_ref],
                        validation_result=validation,
                        compact_memory_after=trajectory.memory.actor_payload(),
                        token_metadata=tokens,
                    )
                )
                pending_searches.append((trajectory, action.query, observation, observation_ref))

            evidence_items = []
            for trajectory, query, observation, observation_ref in pending_searches:
                full = evidence_prompt(corpus.questions[trajectory.qid].question, trajectory.memory, query, observation)
                current_payload = json.loads(full.split("\nCurrent input:\n", 1)[1])
                forbidden = {"answer", "answers", "answer_aliases", "is_supporting", "support_pids", "question_decomposition", "paragraph_support_idx"}
                if forbidden & set(current_payload):
                    raise RuntimeError(f"scorer-label leakage in EVIDENCE_UPDATE input: {forbidden & set(current_payload)}")
                rendered, ids = render_chat(tokenizer, EVIDENCE_SYSTEM, full)
                if len(ids) + EVIDENCE_MAX_NEW_TOKENS > ACTOR_CONTEXT_TOKENS:
                    finish(trajectory, corpus, "context_overflow")
                    continue
                prompt_hash = hashlib.sha256(rendered.encode()).hexdigest()
                prompts_audit[prompt_hash] = rendered
                evidence_items.append((trajectory, query, observation, observation_ref, rendered, ids))
            texts, token_rows = generate_batch(llm, [item[5] for item in evidence_items], evidence_params, request)
            for (trajectory, query, observation, observation_ref, rendered, _ids), raw, tokens in zip(evidence_items, texts, token_rows):
                memory_before = trajectory.memory.actor_payload()
                trajectory.generated_tokens["EVIDENCE_UPDATE"] += tokens["generated_tokens"]
                try:
                    update = parse_evidence_update(raw)
                except ValidationError as exc:
                    trajectory.transitions.append(
                        transition_record(
                            trajectory_id=trajectory.trajectory_id,
                            qid=trajectory.qid,
                            rollout_id=trajectory.rollout_id,
                            transition_index=len(trajectory.transitions),
                            mode="EVIDENCE_UPDATE",
                            prompt=rendered,
                            response=raw,
                            compact_memory_before=memory_before,
                            semantic_output=None,
                            observation_refs=[observation_ref],
                            validation_result={"format_failure": True, "error": str(exc)},
                            compact_memory_after=trajectory.memory.actor_payload(),
                            token_metadata=tokens,
                        )
                    )
                    finish(trajectory, corpus, "format_failure_evidence")
                    continue
                trajectory.proposed_selections += len(update.selected_evidence)
                validation = trajectory.memory.validate_and_update(query, observation, update)
                trajectory.rejected_selections.extend(validation["rejected"])
                trajectory.selected_pids.update(row["pid"] for row in validation["accepted"])
                trajectory.transitions.append(
                    transition_record(
                        trajectory_id=trajectory.trajectory_id,
                        qid=trajectory.qid,
                        rollout_id=trajectory.rollout_id,
                        transition_index=len(trajectory.transitions),
                        mode="EVIDENCE_UPDATE",
                        prompt=rendered,
                        response=raw,
                        compact_memory_before=memory_before,
                        semantic_output=update.model_dump(),
                        observation_refs=[observation_ref],
                        validation_result={"format_failure": False, "schema_valid": True, **validation},
                        compact_memory_after=trajectory.memory.actor_payload(),
                        token_metadata=tokens,
                    )
                )
                # The raw observation is deliberately not attached to state;
                # the next loop reconstructs DECISION solely from memory.
                observation = None
        wall_seconds = time.perf_counter() - started
    finally:
        if llm is not None:
            try:
                llm.sleep(level=2)
            except Exception as exc:
                cleanup_errors.append(f"llm.sleep: {type(exc).__name__}: {exc}")
            del llm
        gc.collect()
        monitor_stop.set()
        if monitor.is_alive():
            monitor.join(timeout=10)

    gpu_peak = read_gpu_peak(args.gpu_log)
    metrics = summarize(corpus, trajectories, wall_seconds, gpu_peak)
    config = {
        "actor_base": str(args.model.resolve()),
        "actor_lora": str(args.adapter.resolve()),
        "temperature": args.temperature,
        "top_p": 1.0,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "n": args.n,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "top_k_retrieval": 2,
        "rrf_k": 60,
        "max_search_actions": MAX_SEARCH_ACTIONS,
        "max_decision_transitions": MAX_DECISION_TRANSITIONS,
        "actor_context_tokens": ACTOR_CONTEXT_TOKENS,
        "decision_max_new_tokens": DECISION_MAX_NEW_TOKENS,
        "evidence_max_new_tokens": EVIDENCE_MAX_NEW_TOKENS,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "external_network_or_llm_calls": 0,
        "fixed_semantic_roles": 0,
        "generation_constraint": "off because vLLM 0.9.2 V1 xgrammar rejects its cached tokenizer wrapper; outputs still undergo strict Pydantic JSON validation with no repair",
        "evidence_generation_stop": {
            "sequence": "]}",
            "included_in_raw_output": True,
            "purpose": "terminate the raw generation at the schema's outer closing sequence; no post-generation stripping or repair",
        },
        "protocol_hashes": {
            "decision_system_sha256": hashlib.sha256(DECISION_SYSTEM.encode()).hexdigest(),
            "evidence_system_sha256": hashlib.sha256(EVIDENCE_SYSTEM.encode()).hexdigest(),
            "decision_schema_sha256": stable_json_hash(DECISION_ADAPTER.json_schema()),
            "evidence_schema_sha256": stable_json_hash(EvidenceUpdate.model_json_schema()),
        },
    }
    gate_checks = {
        "each_mode_schema_valid_at_least_0_95": all(
            row["valid_rate"] >= 0.95 for row in metrics["schema"].values()
        ),
        "live_invalid_selection_at_most_0_05": metrics["invalid_quote_or_pid_rate"] <= 0.05,
        "zero_scorer_label_leakage": True,
        "zero_external_llm_network_calls": True,
        "multiple_query_sequence_patterns": metrics["unique_query_sequence_signatures"] > 1,
        "reward_not_all_one": metrics["reward_counts"].get(1, 0) < metrics["trajectory_count"],
        "reward_not_all_zero": metrics["reward_counts"].get(1, 0) > 0,
        "compact_memory_at_most_512_tokens": metrics["max_compact_memory_tokens"] <= 512,
    }
    diagnostic_preferences = {
        "some_mixed_groups_preferred": metrics["mixed_reward_groups"] > 0,
    }
    before_metrics = None
    if args.before_results and args.before_results.is_file():
        before_metrics = json.loads(args.before_results.read_text()).get("metrics")
    result = {
        "phase": "C",
        "iteration_id": args.iteration_id,
        "parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "config_hash": stable_json_hash(config),
        "hypothesis": args.hypothesis,
        "exact_changed_variables": args.changed_variable or ["initial protocol baseline; no prior iteration"],
        "before_metrics": before_metrics,
        "configuration": config,
        "artifacts": {
            "corpus_sha256": sha256_file(args.corpus),
            "embeddings_sha256": sha256_file(args.embeddings),
            "actor_config_sha256": sha256_file(args.model / "config.json"),
            "actor_model_index_sha256": sha256_file(args.model / "model.safetensors.index.json"),
            "lora_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
            "lora_weights_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
            "detail_output": str(args.detail_output.resolve()),
            "gpu_log": str(args.gpu_log.resolve()),
        },
        "metrics": metrics,
        "gate_checks": gate_checks,
        "diagnostic_preferences": diagnostic_preferences,
        "gate_passed": all(gate_checks.values()),
        "hypothesis_supported": all(gate_checks.values()),
        "cleanup_errors": cleanup_errors,
    }
    detail = {
        "schema_version": 1,
        "iteration_id": args.iteration_id,
        "configuration": config,
        "qids": qids,
        "trajectories": [
            {
                "trajectory_id": trajectory.trajectory_id,
                "qid": trajectory.qid,
                "rollout_id": trajectory.rollout_id,
                "rollout_index": trajectory.rollout_index,
                "transitions": trajectory.transitions,
                "termination_reason": trajectory.termination_reason,
                "final_answer": trajectory.final_answer,
                "reward_detail": trajectory.reward_detail,
                "query_sequence": trajectory.query_sequence,
                "selected_pids": sorted(trajectory.selected_pids),
                "retrieved_pids": sorted(trajectory.retrieved_pids),
                "latency_seconds": trajectory.ended - trajectory.started,
            }
            for trajectory in trajectories
        ],
        "audit_prompts": prompts_audit,
        "audit_observations": observation_audit,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.detail_output.write_text(json.dumps(detail, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
