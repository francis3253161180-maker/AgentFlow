import unittest
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from agentflow.models.structured_outputs import game24_reward_decision
from agentflow.runner import filter_trainable_triplets
from agentflow.tracer.triplet import role_identity_from_attributes
from agentflow.types import Triplet
from agentflow.types import Rollout
from agentflow.verl.advantage import compute_rollout_group_advantage
from agentflow.verl.daemon import AgentModeDaemon
from agentflow.verl.unified_smoke_capture import (
    _field_digest,
    validate_replay_pack_for_update,
)
from train.utils import compute_score


class _Batch:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch

    def __len__(self):
        return int(self.batch.batch_size[0])


class P0ScientificCorrectnessTest(unittest.TestCase):
    def test_unified_filter_keeps_only_attributed_trainable_actor(self):
        actor = Triplet(
            prompt={"token_ids": [1]}, response={"token_ids": [2]}, reward=1.0,
            metadata={"role": "planner_main", "model_name": "qwen-actor", "trainable": True},
        )
        fixed = Triplet(
            prompt={"token_ids": [1]}, response={"token_ids": [3]}, reward=1.0,
            metadata={"role": "fixed", "model_name": "qwen-base", "trainable": False},
        )
        unknown = Triplet(prompt={"token_ids": [1]}, response={"token_ids": [4]}, reward=1.0)
        kept, stats = filter_trainable_triplets([actor, fixed, unknown], unified_local=True)
        self.assertEqual(kept, [actor])
        self.assertEqual(stats["excluded_fixed"], 1)
        self.assertEqual(stats["excluded_unattributed"], 1)

    def test_role_identity_maps_model_without_dataset_specific_logic(self):
        self.assertEqual(
            role_identity_from_attributes({"gen_ai.request.model": "qwen-actor"})["role"],
            "planner_main",
        )
        base = role_identity_from_attributes({"gen_ai.request.model": "qwen-base"})
        self.assertEqual(base["role"], "fixed")
        self.assertFalse(base["trainable"])

    def test_rollout_group_advantage_ignores_transition_multiplicity(self):
        # First rollout has two actor transitions, second has one.  Both rows
        # from the first rollout must receive the same outcome advantage.
        batch = TensorDict(
            {
                "response_mask": torch.ones((3, 2), dtype=torch.float32),
            },
            batch_size=[3],
        )
        data = _Batch(
            batch,
            {
                "prompt_id_list": np.array(["p", "p", "p"], dtype=object),
                "data_id_list": np.array(["d", "d", "d"], dtype=object),
                "rollout_id_list": np.array(["r0", "r0", "r1"], dtype=object),
                "rollout_reward_list": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
        )
        compute_rollout_group_advantage(data, rollout_n=2, normalize_by_std=False)
        self.assertTrue(torch.equal(data.batch["advantages"][0], data.batch["advantages"][1]))
        self.assertAlmostEqual(float(data.batch["advantages"][0, 0]), -0.5)
        self.assertAlmostEqual(float(data.batch["advantages"][2, 0]), 0.5)

    def test_incomplete_group_fails_closed(self):
        data = _Batch(
            TensorDict({"response_mask": torch.ones((1, 1))}, batch_size=[1]),
            {
                "prompt_id_list": np.array(["p"], dtype=object),
                "data_id_list": np.array(["d"], dtype=object),
                "rollout_id_list": np.array(["r0"], dtype=object),
                "rollout_reward_list": np.array([1.0], dtype=np.float32),
            },
        )
        with self.assertRaisesRegex(ValueError, "incomplete rollout groups"):
            compute_rollout_group_advantage(data, rollout_n=2)

    def test_response_truncation_is_a_drop_not_a_rewarded_transition(self):
        self.assertTrue(AgentModeDaemon.transition_should_drop(8, 11, 16, 10))
        self.assertTrue(AgentModeDaemon.transition_should_drop(17, 4, 16, 10))
        self.assertFalse(AgentModeDaemon.transition_should_drop(8, 10, 16, 10))

    def test_batch_carries_stable_group_fields_and_marks_response_truncation(self):
        daemon = object.__new__(AgentModeDaemon)
        daemon.is_train = True
        daemon.train_rollout_n = 2
        daemon.pad_token_id = 0
        daemon.reward_fillna_value = 0.0
        daemon._total_tasks_queued = 2
        daemon._completed_rollouts = {
            "r0": Rollout(
                rollout_id="r0", final_reward=0.0,
                triplets=[Triplet(prompt={"token_ids": [1]}, response={"token_ids": [2, 3, 4]})],
            ),
            "r1": Rollout(
                rollout_id="r1", final_reward=1.0,
                triplets=[Triplet(prompt={"token_ids": [1]}, response={"token_ids": [5]})],
            ),
        }
        daemon._task_id_to_original_sample = {
            "r0": {"data_id": "d", "prompt_id": "p"},
            "r1": {"data_id": "d", "prompt_id": "p"},
        }
        daemon._task_logical_slots = {"r0": ("d", 0), "r1": ("d", 1)}
        batch, _ = daemon.get_train_data_batch(4, 2, "cpu")
        self.assertEqual(list(batch.non_tensor_batch["prompt_id_list"]), ["p", "p"])
        self.assertEqual(list(batch.non_tensor_batch["rollout_id_list"]), ["r0", "r1"])
        self.assertEqual(list(batch.non_tensor_batch["rollout_reward_list"]), [0.0, 1.0])
        self.assertEqual(batch.batch["is_drop_mask"].tolist(), [True, False])

    def test_retry_budget_is_per_logical_slot(self):
        daemon = object.__new__(AgentModeDaemon)
        daemon.enable_rollout_validation = True
        daemon.max_empty_retries = 1
        daemon._empty_rollout_counts = {}
        daemon._task_logical_slots = {"r0": ("d", 1)}
        daemon._task_id_to_original_sample = {"r0": {"data_id": "d", "prompt_id": "p"}}
        invalid = Rollout(rollout_id="r0", triplets=[])
        self.assertTrue(daemon._validate_rollout_for_retry(invalid))
        self.assertFalse(daemon._validate_rollout_for_retry(invalid))
        self.assertEqual(daemon._empty_rollout_counts, {("d", 1): 1})

    def test_game24_requires_original_multiset_and_marked_answer(self):
        question = "Use numbers [2, 5, 8, 10] to make 24."
        self.assertEqual(
            game24_reward_decision(question, "<answer>(10-2)*(8-5)</answer>")[0], True
        )
        self.assertEqual(game24_reward_decision(question, "<answer>6*4</answer>")[0], False)
        self.assertEqual(game24_reward_decision(question, "6*4")[0], False)
        self.assertFalse(compute_score(question, "24", "<answer>6*4</answer>"))

    def test_replay_validation_checks_digest_identity_hash_and_drop_mask(self):
        tensors = {
            key: torch.zeros((1, 2), dtype=torch.float32)
            for key in (
                "prompts", "responses", "input_ids", "attention_mask", "position_ids",
                "response_mask", "old_log_probs", "token_level_scores", "token_level_rewards",
                "advantages", "returns",
            )
        }
        tensors["is_drop_mask"] = torch.zeros((1,), dtype=torch.bool)
        non_tensor = {
            "prompt_id_list": np.array(["p"], dtype=object),
            "data_id_list": np.array(["d"], dtype=object),
            "rollout_id_list": np.array(["r"], dtype=object),
            "rollout_reward_list": np.array([1.0], dtype=np.float32),
        }
        meta_info = {"temperature": 0.7}
        metadata = {
            "model_path": "/model",
            "temperature": 0.7,
            "rollout_n": 1,
            "seed": "seed",
            "lora_pre_hash": "lora-hash",
        }
        pack = {
            "schema_version": 3,
            "kind": "agentflow_unified_authentic_pre_update_replay_pack",
            "metadata": metadata,
            "batch_size": 1,
            "tensor_fields": tensors,
            "non_tensor_batch": non_tensor,
            "meta_info": meta_info,
            "field_inventory": {
                "tensor_fields": sorted(tensors),
                "non_tensor_fields": sorted(non_tensor),
                "meta_info_fields": sorted(meta_info),
            },
        }
        pack["captured_field_digest"] = _field_digest(tensors, non_tensor, meta_info)
        result = validate_replay_pack_for_update(
            pack,
            expected_model_path="/model",
            expected_rollout_n=1,
            expected_temperature=0.7,
            expected_seed="seed",
            current_lora_hash="lora-hash",
        )
        self.assertEqual(result["status"], "validated")
        pack["metadata"]["temperature"] = 0.8
        with self.assertRaisesRegex(ValueError, "temperature mismatch"):
            validate_replay_pack_for_update(
                pack,
                expected_model_path="/model",
                expected_rollout_n=1,
                expected_temperature=0.7,
                expected_seed="seed",
                current_lora_hash="lora-hash",
            )


if __name__ == "__main__":
    unittest.main()
