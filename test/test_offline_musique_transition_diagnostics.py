import unittest

from agentflow.offline_musique_transition_diagnostics import (
    annotate_trajectory,
    signed_distribution,
    summarize_diagnostics,
    support_scores,
    unique_with_tolerance,
)


def evidence_transition(index, accepted):
    return {
        "mode": "EVIDENCE_UPDATE",
        "transition_index": index,
        "validation_result": {
            "format_failure": False,
            "schema_valid": True,
            "accepted": [{"pid": pid, "quote": "q"} for pid in accepted],
        },
    }


class OfflineMusiqueTransitionDiagnosticTest(unittest.TestCase):
    def test_support_scores_empty_bookkeeping_does_not_create_f_score(self):
        row = support_scores(set(), {"g1", "g2"})
        self.assertEqual(row["precision"], 1.0)
        self.assertEqual(row["recall"], 0.0)
        self.assertEqual(row["F1"], 0.0)
        self.assertEqual(row["F2"], 0.0)
        self.assertTrue(row["empty_selected_bookkeeping_precision"])

    def test_annotation_uses_deduplicated_cumulative_validated_pids(self):
        trajectory = {
            "trajectory_id": "q::q__r0",
            "qid": "q",
            "rollout_index": 0,
            "selected_pids": ["g1", "d1", "g2"],
            "transitions": [
                evidence_transition(1, ["g1"]),
                evidence_transition(3, ["g1", "d1"]),
                evidence_transition(5, ["g2"]),
            ],
        }
        row = annotate_trajectory(trajectory, {"g1", "g2"})
        updates = row["transition_scores"]
        self.assertEqual([u["evidence_update_ordinal"] for u in updates], [1, 2, 3])
        self.assertEqual(updates[0]["new_gold_support_count"], 1)
        self.assertEqual(updates[1]["new_gold_support_count"], 0)
        self.assertEqual(updates[1]["new_distractor_count"], 1)
        self.assertLess(updates[1]["delta_F1"], 0)
        self.assertLess(updates[1]["delta_F2"], 0)
        self.assertTrue(updates[2]["full_support_coverage"])
        self.assertFalse(updates[2]["exact_support_set"])

    def test_format_failure_is_not_a_valid_evidence_update(self):
        trajectory = {
            "trajectory_id": "q::q__r0",
            "qid": "q",
            "rollout_index": 0,
            "selected_pids": [],
            "transitions": [
                {
                    "mode": "EVIDENCE_UPDATE",
                    "transition_index": 1,
                    "validation_result": {"format_failure": True},
                }
            ],
        }
        self.assertEqual(annotate_trajectory(trajectory, {"g1"})["transition_scores"], [])

    def test_tolerance_and_signed_rates(self):
        self.assertEqual(unique_with_tolerance([0.0, 1e-13, 0.5, 0.5 + 1e-13]), 2)
        row = signed_distribution([-0.25, 0.0, 1e-13, 0.5])
        self.assertEqual((row["negative_count"], row["zero_count"], row["positive_count"]), (1, 2, 1))
        self.assertEqual(row["positive_rate"], 0.25)

    def test_outcome_group_persists_binary_reward_variance(self):
        trajectories = []
        for index, reward in enumerate((0, 1)):
            trajectories.append(
                {
                    "trajectory_id": f"q::q__r{index}",
                    "qid": "q",
                    "rollout_index": index,
                    "reward_detail": {"reward": reward},
                    "termination_reason": "answer",
                    "selected_pids": [],
                    "retrieved_pids": [],
                    "transitions": [],
                }
            )
        annotations = [annotate_trajectory(row, {"g1"}) for row in trajectories]
        summary = summarize_diagnostics(trajectories, annotations, {"q": {"g1"}}, ["q"], rollout_n=2)
        group = summary["outcome"]["per_qid"][0]
        self.assertEqual(group["terminal_reward_variance"], 0.25)
        self.assertEqual(group["group_type"], "mixed")


if __name__ == "__main__":
    unittest.main()
