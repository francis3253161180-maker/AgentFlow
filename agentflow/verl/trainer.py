import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Tuple

import ray
import torch
from omegaconf import OmegaConf
from pprint import pprint
from tqdm import tqdm

from codetiming import Timer
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    AdvantageEstimator,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from .daemon import AgentModeDaemon
from .advantage import compute_rollout_group_advantage
from .unified_smoke_capture import (
    _write_json_atomic,
    validate_replay_pack_for_update,
    write_replay_pack_from_dataproto,
)

import os
import json
import uuid
from collections import defaultdict
from pathlib import Path

import time

import numpy as np

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


def materialize_offline_replay(pack: dict, *, multi_turn: bool) -> DataProto:
    """Build the update DataProto without mutating the authenticated pack."""
    replay = DataProto.from_dict(
        tensors=pack["tensor_fields"],
        non_tensors=pack.get("non_tensor_batch", {}),
        meta_info=deepcopy(pack.get("meta_info", {})),
    )
    if "temperature" not in replay.meta_info:
        replay.meta_info["temperature"] = float(
            pack.get("metadata", {}).get("temperature", 0.7) or 0.7
        )
    # The ordinary online path installs this immediately after constructing
    # response_mask.  Offline replay already carries the authenticated masks,
    # but update_actor still requires this per-sample denominator for its
    # official global-loss aggregation.
    if "global_token_num" not in replay.meta_info:
        attention_mask = replay.batch.get("attention_mask")
        if attention_mask is None:
            raise ValueError("offline replay requires attention_mask for global_token_num")
        replay.meta_info["global_token_num"] = torch.sum(attention_mask, dim=-1).tolist()
    replay.meta_info["multi_turn"] = multi_turn
    return replay


def _single_worker_result(value, *, operation: str) -> dict:
    """Normalize one-worker Ray replies without accepting ambiguous identity."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{operation} expected exactly one worker result, got {len(value)}")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{operation} returned {type(value)!r}, not a mapping")
    return value


def offline_kl_audit(
    replay: DataProto,
    *,
    current_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_loss_type: str,
    kl_loss_coef: float,
) -> dict:
    """Summarize the official masked KL objective without an optimizer step."""
    response_mask = replay.batch["response_mask"].bool()
    advantages = replay.batch["advantages"].float()
    if not bool(torch.isfinite(current_log_probs).all().item()):
        raise ValueError("non-finite current actor log probabilities in KL audit")
    if not bool(torch.isfinite(ref_log_probs).all().item()):
        raise ValueError("non-finite reference log probabilities in KL audit")
    if not bool(torch.isfinite(advantages).all().item()):
        raise ValueError("non-finite task advantages in KL audit")
    if int(response_mask.sum().item()) < 1:
        raise ValueError("empty response mask in KL audit")

    from verl.trainer.ppo.core_algos import kl_penalty

    current = current_log_probs.float()[response_mask]
    reference = ref_log_probs.float()[response_mask]
    task_advantage = advantages[response_mask]
    kld = kl_penalty(current, reference, kl_loss_type)
    if not bool(torch.isfinite(kld).all().item()):
        raise ValueError("non-finite KL values in KL audit")
    # d low_var_kl/d logp = 1 - exp(ref_logp - logp), except at the
    # explicit clamp boundaries.  This is the complete initial objective
    # derivative with respect to logp when task advantage is identically zero.
    if kl_loss_type not in {"low_var_kl", "k3"}:
        raise ValueError(f"offline KL audit only supports official low_var_kl, got {kl_loss_type!r}")
    raw = torch.clamp(reference - current, min=-20, max=20)
    unclamped_kld = torch.exp(raw) - raw - 1
    derivative = torch.where(
        (unclamped_kld >= -10) & (unclamped_kld <= 10),
        1 - torch.exp(raw),
        torch.zeros_like(raw),
    )
    return {
        "schema_version": 1,
        "reference_policy": "same actor with LoRA adapter disabled (VERL FSDP LoRA ref_in_actor)",
        "kl_loss_type": kl_loss_type,
        "kl_loss_coef_beta": float(kl_loss_coef),
        "entropy_coefficient": 0.0,
        "response_masked_token_count": int(response_mask.sum().item()),
        "task_advantage": {
            "abs_max": float(task_advantage.abs().max().item()),
            "nonzero_token_count": int((task_advantage != 0).sum().item()),
            "all_zero": bool(torch.all(task_advantage == 0).item()),
        },
        "actor_minus_ref_logprob": {
            "mean": float((current - reference).mean().item()),
            "abs_mean": float((current - reference).abs().mean().item()),
            "abs_max": float((current - reference).abs().max().item()),
        },
        "masked_kl": {
            "mean": float(kld.mean().item()),
            "abs_mean": float(kld.abs().mean().item()),
            "abs_max": float(kld.abs().max().item()),
            "nonzero_token_count": int((kld != 0).sum().item()),
        },
        "full_objective_logprob_gradient": {
            "task_component_abs_max": 0.0,
            "kl_component_abs_mean": float((float(kl_loss_coef) * derivative).abs().mean().item()),
            "kl_component_abs_max": float((float(kl_loss_coef) * derivative).abs().max().item()),
            "nonzero_token_count": int((derivative != 0).sum().item()),
        },
    }


class AgentFlowTrainer(RayPPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentFlowServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    RayPPOTrainer and focusing on the agent mode workflow.

    Key differences from RayPPOTrainer:
    1. Uses AgentModeDaemon for server communication
    2. Simplified data flow without pop/union operations
    3. Direct batch processing through agent daemon
    4. Streamlined validation using agent_mode validation
    """

    def _cleanup_rollout_engine(self, reason: str):
        """Drain custom async vLLM servers before prefix-cache reset/sleep."""
        servers = list(getattr(self.async_rollout_manager, "async_llm_servers", []) or [])
        cleanup_refs = []
        for server in servers:
            cleanup = getattr(server, "cleanup", None)
            if cleanup is not None:
                cleanup_refs.append(cleanup.remote(reason=reason))

        if cleanup_refs and len(cleanup_refs) == len(servers):
            results = ray.get(cleanup_refs)
            print(f"VLLM_CLEANUP_DRIVER reason={reason} results={results}")
            return results

        # Compatibility fallback for non-AgentFlow server classes. The current
        # branch uses PatchedvLLMServer, so this path is not used in production.
        print(
            "VLLM_CLEANUP_DRIVER fallback=manager_sleep "
            f"reason={reason} custom_servers={len(cleanup_refs)}/{len(servers)}"
        )
        self.async_rollout_manager.sleep()
        return []

    def _maybe_cleanup_health_check(self):
        """Test-only wake/health check after the forced-timeout cleanup."""
        if os.environ.get("AGENTFLOW_CLEANUP_HEALTH_CHECK", "0") != "1":
            return
        self.async_rollout_manager.wake_up()
        servers = list(getattr(self.async_rollout_manager, "async_llm_servers", []) or [])
        health_refs = [server.health_check.remote() for server in servers if hasattr(server, "health_check")]
        if len(health_refs) != len(servers):
            raise RuntimeError("cleanup health check unavailable on one or more rollout servers")
        results = ray.get(health_refs)
        print(f"VLLM_CLEANUP_HEALTH_CHECK status=ok results={results}")

    def _capture_behavior_policy_snapshot(self) -> None:
        """Capture actor LoRA/RNG state before any rollout request is issued."""
        if os.environ.get("AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH", "").strip():
            result = self.actor_rollout_wg.restore_agentflow_behavior_snapshot()
            print(f"AGENTFLOW_BEHAVIOR_SNAPSHOT_RESTORE_DRIVER result={result}", flush=True)
        if not os.environ.get("AGENTFLOW_BEHAVIOR_SNAPSHOT_PATH", "").strip():
            return
        result = self.actor_rollout_wg.capture_agentflow_behavior_snapshot()
        print(f"AGENTFLOW_BEHAVIOR_SNAPSHOT_DRIVER result={result}", flush=True)

    def _capture_pre_update_replay(self) -> None:
        """Build replay tensors from completed triplets without updating weights."""
        output = os.environ.get("AGENTFLOW_REPLAY_PACK_PATH", "").strip()
        if not output:
            return
        replay, batch_metrics = self.agent_mode_daemon.get_train_data_batch(
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            device="cpu",
        )
        if replay is None:
            raise RuntimeError("cannot build replay pack: no valid rollout transitions")
        replay.non_tensor_batch["uid"] = np.asarray(replay.non_tensor_batch["prompt_id_list"], dtype=object)
        replay.batch["response_mask"] = compute_response_mask(replay)
        replay.batch["token_level_rewards"] = replay.batch["token_level_scores"]
        replay.meta_info["temperature"] = float(self.config.actor_rollout_ref.rollout.temperature)
        old_log_prob = self.actor_rollout_wg.compute_log_prob(replay)
        replay = replay.union(old_log_prob)
        replay = compute_rollout_group_advantage(
            replay,
            rollout_n=int(self.config.actor_rollout_ref.rollout.n),
            normalize_by_std=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
        )
        snapshot_meta = {}
        snapshot_metadata_path = os.environ.get("AGENTFLOW_BEHAVIOR_SNAPSHOT_METADATA_PATH", "").strip()
        if snapshot_metadata_path and Path(snapshot_metadata_path).exists():
            try:
                snapshot_meta = json.loads(Path(snapshot_metadata_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                snapshot_meta = {"read_error": "snapshot_metadata_unreadable"}
        route = None
        route_path = os.environ.get("AGENTFLOW_ROLE_ROUTING_STATE", "").strip()
        if route_path and Path(route_path).exists():
            try:
                route = json.loads(Path(route_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                route = {"read_error": "role_route_unreadable"}
        metadata = {
            "source_run_id": os.environ.get("AGENTFLOW_UNIFIED_SMOKE_RUN_ID", ""),
            "model_path": self.config.actor_rollout_ref.model.path,
            "temperature": float(self.config.actor_rollout_ref.rollout.temperature),
            "rollout_n": int(self.config.actor_rollout_ref.rollout.n),
            "seed": os.environ.get("AGENTFLOW_UNIFIED_SEED", ""),
            "scorer": os.environ.get("AGENTFLOW_UNIFIED_SCORER", "local deterministic"),
            "optimizer_steps": 0,
            "replay_only": True,
            "batch_metrics": batch_metrics,
            "behavior_snapshot": snapshot_meta,
            "lora_pre_hash": snapshot_meta.get("lora_hash") if isinstance(snapshot_meta, dict) else None,
            "role_route_state": route,
        }
        result = write_replay_pack_from_dataproto(replay, output, metadata)
        print(f"AGENTFLOW_PRE_UPDATE_REPLAY_DRIVER result={result}", flush=True)

    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."
        # Explicitly opt-in instrumentation for rollout-only group-diversity
        # experiments.  This reuses the training rollout path and its n value,
        # but fit() returns immediately through trainer.val_only before any
        # training batch, advantage computation, backward pass, or optimizer
        # update.  Normal validation remains unchanged (one rollout/sample).
        rollout_only_group_mode = os.environ.get(
            "AGENTFLOW_ROLLOUT_ONLY_GROUP_MODE", "0"
        ) == "1"

        # no empty check dataloader
        try:
            test_data = next(iter(self.val_dataloader))
        except StopIteration:
            raise ValueError("Validation dataloader is empty. Check your validation dataset.")

        # no empty check key
        print(f"Validation data keys: {test_data.keys()}")
        for key, value in test_data.items():
            if isinstance(value, list):
                print(f"Validation data {key} length: {len(value)}")
                if len(value) == 0:
                    print(f"Warning: Empty data in {key}")
            elif isinstance(value, torch.Tensor):
                print(f"Validation data {key} shape: {value.shape}")
                if value.numel() == 0:
                    print(f"Warning: Empty tensor in {key}")
            else:
                print(f"Validation data {key} type: {type(value)}")

        # no empty check
        if not test_data or all((isinstance(v, list) and len(v) == 0) or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in test_data.values()):
            raise ValueError("Validation data is empty. Check your validation dataset.")

        test_batch = DataProto.from_single_dict(test_data)
        # test_batch.non_tensor_batch["step"] = np.ones_like(test_batch.non_tensor_batch["question"]) * self.global_steps
        self.async_rollout_manager.wake_up()
        if rollout_only_group_mode:
            print(
                "Rollout-only group mode: queueing training-mode rollouts "
                "with configured rollout.n; no optimizer step will run because "
                "trainer.val_only=true."
            )
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=rollout_only_group_mode,
        )

        # whether persisting queueing 
        if self.agent_mode_daemon._total_tasks_queued == 0:
            raise ValueError("No validation tasks were queued. Check data preparation.")

        test_metrics = None
        try:
            self.agent_mode_daemon.run_until_all_finished()

            # Check if we have any completed rollouts, with more detailed error reporting
            completed_count = len(self.agent_mode_daemon._completed_rollouts)
            valid_count = len([r for r in self.agent_mode_daemon._completed_rollouts.values()
                              if r.triplets and len(r.triplets) > 0])
            original_count = self.agent_mode_daemon._total_tasks_queued

            completion_rate = completed_count / original_count if original_count > 0 else 0
            print(f"Validation summary: {completed_count}/{original_count} total rollouts ({completion_rate:.1%}), {valid_count} valid rollouts")

            # More lenient validation acceptance
            if completed_count == 0:
                raise ValueError("No validation tasks completed. Check server and agent execution.")

            # Accept partial results if we have some reasonable completion
            min_acceptable_rate = float(
                os.environ.get("AGENTFLOW_CLEANUP_SMOKE_MIN_COMPLETION_RATE", "0.1")
            )
            if completion_rate < min_acceptable_rate:
                raise ValueError(f"Insufficient validation completion: {completion_rate:.1%} < {min_acceptable_rate:.1%}. "
                               f"Only {completed_count}/{original_count} tasks completed.")

            if valid_count == 0:
                print("Warning: No valid validation rollouts (all have empty triplets), using fallback metrics")
            else:
                print(f"Validation proceeding with {valid_count} valid rollouts ({valid_count/completed_count:.1%} of completed)")

            self._capture_pre_update_replay()

            test_metrics = self.agent_mode_daemon.get_test_metrics(
                allow_train=rollout_only_group_mode
            )
        finally:
            self._cleanup_rollout_engine(self.agent_mode_daemon.get_cleanup_reason())
            self.agent_mode_daemon.clear_data_and_server()
            self._maybe_cleanup_health_check()

        return test_metrics

    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        offline_pack_path = os.environ.get("AGENTFLOW_OFFLINE_REPLAY_PACK_PATH", "").strip()
        if offline_pack_path:
            # Opt-in diagnostic path: use the exact pre-update DataProto captured
            # by the actor worker.  No AgentFlow task is queued and no rollout
            # or reward computation is performed here.
            pack = torch.load(offline_pack_path, map_location="cpu", weights_only=False)
            if pack.get("kind") != "agentflow_unified_authentic_pre_update_replay_pack":
                raise ValueError(f"unsupported offline replay pack kind: {pack.get('kind')!r}")
            metadata = pack.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("offline replay metadata must be a mapping")
            source_path = os.environ.get("AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH", "").strip()
            if not source_path:
                raise ValueError("offline replay update requires AGENTFLOW_BEHAVIOR_SNAPSHOT_SOURCE_PATH")
            verify = getattr(self.actor_rollout_wg, "restore_agentflow_behavior_snapshot", None)
            if not callable(verify):
                raise ValueError("actor worker cannot verify behavior snapshot")
            verified = _single_worker_result(verify(), operation="behavior snapshot verification")
            current_lora_hash = verified.get("lora_hash")
            validate_replay_pack_for_update(
                pack,
                expected_model_path=self.config.actor_rollout_ref.model.path,
                expected_rollout_n=int(self.config.actor_rollout_ref.rollout.n),
                expected_temperature=float(self.config.actor_rollout_ref.rollout.temperature),
                expected_seed=os.environ.get("AGENTFLOW_UNIFIED_SEED"),
                current_lora_hash=str(current_lora_hash or ""),
            )
            replay = materialize_offline_replay(
                pack,
                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
            )
            if self.config.actor_rollout_ref.actor.use_kl_loss:
                # Follow the official GRPO path: pi_ref is the same LoRA actor
                # with its adapter disabled, and both values are defined only
                # on response-mask tokens.  The immutable pack keeps rollout
                # old_log_probs for PPO.  A full fresh-current pass is needed
                # only for the explicit audit mode; ordinary update execution
                # computes the reference once and lets update_actor compute the
                # trainable current policy exactly as in official VERL.
                reference = self.actor_rollout_wg.compute_ref_log_prob(replay)
                ref_log_probs = reference.batch["ref_log_prob"]
                if os.environ.get("AGENTFLOW_OFFLINE_REPLAY_AUDIT_ONLY", "0") == "1":
                    current = self.actor_rollout_wg.compute_log_prob(replay)
                    kl_audit = offline_kl_audit(
                        replay,
                        current_log_probs=current.batch["old_log_probs"],
                        ref_log_probs=ref_log_probs,
                        kl_loss_type=str(self.config.actor_rollout_ref.actor.kl_loss_type),
                        kl_loss_coef=float(self.config.actor_rollout_ref.actor.kl_loss_coef),
                    )
                    kl_audit.update(
                        {
                            "mode": "offline_replay_pre_update",
                            "pack_path": offline_pack_path,
                            "behavior_lora_hash": str(current_lora_hash),
                        }
                    )
                    audit_path = os.environ.get("AGENTFLOW_OFFLINE_REPLAY_KL_AUDIT_PATH", "").strip()
                    if audit_path:
                        _write_json_atomic(Path(audit_path), kl_audit)
                    print(f"AGENTFLOW_OFFLINE_REPLAY_KL_AUDIT {json.dumps(kl_audit, sort_keys=True)}", flush=True)
                    return {
                        "offline_replay/audit_only": 1,
                        "offline_replay/rollout_requests": 0,
                        "offline_replay/external_calls": 0,
                        "offline_replay/kl_mean": kl_audit["masked_kl"]["mean"],
                        "offline_replay/kl_grad_abs_mean": kl_audit["full_objective_logprob_gradient"]["kl_component_abs_mean"],
                    }
                replay = replay.union(reference)
            expected_hash = metadata.get("lora_pre_hash")
            print(
                "AGENTFLOW_OFFLINE_REPLAY_UPDATE "
                f"pack={offline_pack_path} batch={len(replay)} expected_lora_pre={expected_hash} "
                "rollout_requests=0 external_calls=0",
                flush=True,
            )
            output = self.actor_rollout_wg.update_actor(replay)
            replay_metrics = reduce_metrics(output.meta_info["metrics"])
            replay_metrics.update(
                {
                    "offline_replay/batch_size": len(replay),
                    "offline_replay/rollout_requests": 0,
                    "offline_replay/external_calls": 0,
                }
            )
            print(f"AGENTFLOW_OFFLINE_REPLAY_METRICS {replay_metrics}", flush=True)
            return replay_metrics

        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        # data key check & no empty check
        print(f"Training data keys: {batch_dict.keys()}")
        for key, value in batch_dict.items():
            if isinstance(value, list):
                print(f"Training data {key} length: {len(value)}")
                if len(value) == 0:
                    print(f"Warning: Empty data in {key}")
            elif isinstance(value, torch.Tensor):
                print(f"Training data {key} shape: {value.shape}")
                if value.numel() == 0:
                    print(f"Warning: Empty tensor in {key}")
            else:
                print(f"Training data {key} type: {type(value)}")

        # ensure no empty
        if not batch_dict or all((isinstance(v, list) and len(v) == 0) or (isinstance(v, torch.Tensor) and v.numel() == 0) for v in batch_dict.values()):
            raise ValueError("Training data is empty. Check your training dataset.")

        with _timer("step", timing_raw):
            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw):
                # gen_batch.non_tensor_batch["step"] = np.ones_like(gen_batch.non_tensor_batch["question"]) * self.global_steps
                self.async_rollout_manager.wake_up()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )

                if self.agent_mode_daemon._total_tasks_queued == 0:
                    raise ValueError("No training tasks were queued. Check data preparation.")

                try:
                    self.agent_mode_daemon.run_until_all_finished()

                    if len(self.agent_mode_daemon._completed_rollouts) == 0:
                        raise ValueError("No training tasks completed. Check server and agent execution.")

                    batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                        max_prompt_length=self.config.data.max_prompt_length,
                        max_response_length=self.config.data.max_response_length,
                        device=gen_batch.batch["fake_ids"].device,
                    )
                    metrics.update(agent_metrics)
                finally:
                    self._cleanup_rollout_engine(self.agent_mode_daemon.get_cleanup_reason())
                    self.agent_mode_daemon.clear_data_and_server()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["prompt_id_list"]

            batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with _timer("adv", timing_raw):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                    batch = compute_rollout_group_advantage(
                        batch,
                        rollout_n=int(self.config.actor_rollout_ref.rollout.n),
                        normalize_by_std=norm_adv_by_std_in_grpo,
                    )
                else:
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )

            # after advantages are assinged, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["agent_mode/n_dropped_sample_because_of_length"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)

            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_mini_batch"] = n_transition - n_remained_transition

            n_transition = len(batch)
            # make sure divisible by k_partitions for seqlen_balancing
            k_partitions = self.config.trainer.n_gpus_per_node  # 一般等于 num_workers 或者 8
            n_remained_transition = n_transition // k_partitions * k_partitions
            if n_remained_transition != n_transition:
                batch = batch[list(range(n_remained_transition))]
            metrics["agent_mode/n_dropped_sample_because_of_gpu_partitions"] = n_transition - n_remained_transition

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics

    def _dump_rollout_data(self, inputs, outputs, scores, reward_extra_infos_dict, metrics, dump_path, is_train, batch, data_ids=None, ground_truths=None):
        data_type = 'train' if is_train else 'val'
        current_time = time.strftime("%Y%m%d_%H%M%S")
        step_dir = os.path.join(dump_path, data_type, f"step_{self.global_steps}_{current_time}")
        os.makedirs(step_dir, exist_ok=True)

        if data_ids is None:
            data_ids = batch.non_tensor_batch.get("data_id_list", [str(uuid.uuid4()) for _ in inputs])
        else:
            data_ids = data_ids[:len(inputs)] + [str(uuid.uuid4()) for _ in range(len(inputs) - len(data_ids))]

        question_groups = defaultdict(list)
        all_metrics = metrics.copy()

        for i, (input_text, output_text, score) in enumerate(zip(inputs, outputs, scores)):
            data_id = data_ids[i] if i < len(data_ids) else str(uuid.uuid4())

            record = {
                "query_index": i,
                "data_id": data_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input": input_text,
                "output": output_text,
                "score": score,
                "metrics": {},
                "extra_info": {}
            }

            if ground_truths and i < len(ground_truths):
                record["ground_truth"] = ground_truths[i]

            for metric_name, metric_value in all_metrics.items():
                if isinstance(metric_value, (list, tuple)) and i < len(metric_value):
                    record["metrics"][metric_name] = metric_value[i]
                else:
                    record["metrics"][f"global_{metric_name}"] = metric_value

            if reward_extra_infos_dict:
                for key, values in reward_extra_infos_dict.items():
                    if i < len(values):
                        record["extra_info"][key] = values[i]

            question_groups[data_id].append(record)

        for data_id, records in question_groups.items():
            json_path = os.path.join(step_dir, f"query_{data_id}.json")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"Successfully saved rollout data to {step_dir}")

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
        self.agent_mode_daemon = AgentModeDaemon(
            self.config.agentflow.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": self.config.actor_rollout_ref.model.path,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            enable_rollout_validation=self.config.agentflow.get("enable_rollout_validation", True),
            max_empty_retries=self.config.agentflow.get("max_empty_retries", 2),
        )
        self.agent_mode_daemon.start()
        # Capture the behavior policy after actor initialization but before the
        # validation path wakes the engine or queues its first rollout.
        self._capture_behavior_policy_snapshot()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            print(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # train step
                metrics = self._train_step(batch_dict)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("validate", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                # step metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # This exit logic is to ensure a robust CI.
                    pprint(f"Flush the logger...")
                    del logger  # Make sure the loggers are flushed and closed properly
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1
