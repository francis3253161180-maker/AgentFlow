import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from agentflow.offline_musique import (
    BGEEncoder,
    CompactMemory,
    EvidenceUpdate,
    LocalCorpusSearch,
    OfflineCorpus,
    Paragraph,
    QuestionRecord,
    ScorerRecord,
    build_corpus,
    decision_prompt,
    evidence_prompt,
    paragraph_pid,
    parse_decision,
    parse_evidence_update,
    terminal_reward,
    terminal_reward_coverage_v1,
)


class TinyEncoder:
    def encode(self, texts, batch_size=1):
        rows = []
        for text in texts:
            lower = text.lower()
            vector = np.array(
                [lower.count("amber"), lower.count("token"), lower.count("mass") + 0.1],
                dtype=np.float32,
            )
            rows.append(vector / np.linalg.norm(vector))
        return np.stack(rows)


def tiny_corpus():
    rows = [
        ("Token", "The token is amber and kept in a drawer."),
        ("Mass", "The box has a mass of two grams."),
        ("Noise", "Clouds form above the distant ridge."),
    ]
    paragraphs = {}
    pids = []
    for title, text in rows:
        pid = paragraph_pid(title, text)
        paragraphs[pid] = Paragraph(pid, title, text)
        pids.append(pid)
    question = QuestionRecord("q1", "What color is the token?", tuple(pids), 2)
    scorer = ScorerRecord(frozenset(pids[:2]), ("amber", "amber color"))
    corpus = OfflineCorpus(paragraphs, {"q1": question}, {"q1": scorer}, {"manifest_sha256": "tiny"})
    return corpus, pids


def tiny_cache(path, corpus):
    pids = sorted(corpus.paragraphs)
    texts = [f"{corpus.paragraphs[pid].title}\n{corpus.paragraphs[pid].text}" for pid in pids]
    np.savez(path, pids=np.array(pids), embeddings=TinyEncoder().encode(texts))


class OfflineMusiquePhaseATest(unittest.TestCase):
    def test_scorer_only_labels_never_appear_actor_visible(self):
        corpus, pids = tiny_corpus()
        actor = corpus.actor_question("q1")
        self.assertEqual(actor, {"qid": "q1", "question": "What color is the token?"})
        self.assertFalse({"answer", "answers", "answer_aliases", "support_pids", "is_supporting"} & set(actor))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "embeddings.npz"
            tiny_cache(cache, corpus)
            result = LocalCorpusSearch(corpus, cache, TinyEncoder()).search("q1", "amber token")
        for row in result:
            self.assertEqual(set(row), {"pid", "title", "text", "rank"})
            self.assertNotIn("is_supporting", json.dumps(row))
        self.assertEqual(corpus.scorer_record("q1").support_pids, frozenset(pids[:2]))

    def test_decision_never_receives_raw_prior_observations(self):
        corpus, pids = tiny_corpus()
        memory = CompactMemory()
        observation = [corpus.paragraphs[pids[0]].actor_payload(), corpus.paragraphs[pids[2]].actor_payload()]
        update = EvidenceUpdate.model_validate(
            {"selected_evidence": [{"pid": pids[0], "quote": "The token is amber"}]}
        )
        memory.validate_and_update("token color", observation, update)
        prompt = decision_prompt(corpus.questions["q1"].question, memory, searches_left=5, decisions_left=6)
        self.assertIn("The token is amber", prompt)
        self.assertNotIn("kept in a drawer", prompt)
        self.assertNotIn("Clouds form above", prompt)

    def test_outside_pid_cannot_persist(self):
        corpus, pids = tiny_corpus()
        memory = CompactMemory()
        observation = [corpus.paragraphs[pids[0]].actor_payload()]
        result = memory.validate_and_update(
            "token color",
            observation,
            EvidenceUpdate.model_validate(
                {"selected_evidence": [{"pid": pids[1], "quote": "The box has a mass"}]}
            ),
        )
        self.assertEqual(memory.evidence, [])
        self.assertEqual(result["rejected"][0]["reason"], "pid_not_in_current_observation")

    def test_exact_quote_provenance_passes_and_fails(self):
        corpus, pids = tiny_corpus()
        observation = [corpus.paragraphs[pids[0]].actor_payload()]
        good = CompactMemory()
        result = good.validate_and_update(
            "token color",
            observation,
            EvidenceUpdate.model_validate(
                {"selected_evidence": [{"pid": pids[0], "quote": "The  token is amber"}]}
            ),
        )
        self.assertEqual(result["outcome"], "useful")
        bad = CompactMemory()
        result = bad.validate_and_update(
            "token color",
            observation,
            EvidenceUpdate.model_validate(
                {"selected_evidence": [{"pid": pids[0], "quote": "The token is blue"}]}
            ),
        )
        self.assertEqual(result["rejected"][0]["reason"], "quote_not_exact_normalized_substring")

    def test_raw_topk_discarded_before_next_decision(self):
        corpus, pids = tiny_corpus()
        memory = CompactMemory()
        observation = [corpus.paragraphs[pids[0]].actor_payload(), corpus.paragraphs[pids[2]].actor_payload()]
        evidence_input = evidence_prompt(corpus.questions["q1"].question, memory, "token color", observation)
        self.assertIn("Clouds form above", evidence_input)
        memory.validate_and_update("token color", observation, EvidenceUpdate(selected_evidence=[]))
        observation = None
        next_input = decision_prompt(corpus.questions["q1"].question, memory, searches_left=5, decisions_left=6)
        self.assertNotIn("Clouds form above", next_input)

    def test_no_useful_evidence_only_updates_compact_history(self):
        corpus, pids = tiny_corpus()
        memory = CompactMemory()
        observation = [corpus.paragraphs[pids[2]].actor_payload()]
        memory.validate_and_update("token weight", observation, EvidenceUpdate(selected_evidence=[]))
        self.assertEqual(memory.actor_payload()["search_history"], [{"query": "token weight", "outcome": "no_useful_evidence"}])
        self.assertNotIn("Clouds", json.dumps(memory.actor_payload()))

    def test_retrievers_are_deterministic(self):
        corpus, _ = tiny_corpus()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "embeddings.npz"
            tiny_cache(cache, corpus)
            search = LocalCorpusSearch(corpus, cache, TinyEncoder())
            first = search.rank_all("q1", "amber token")
            second = search.rank_all("q1", "amber token")
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"bm25", "dense", "rrf"})

    def test_cached_embeddings_match_fresh_dense_ranking(self):
        model_path = Path("/root/autodl-tmp/models/bge-small-en-v1.5")
        if not model_path.is_dir():
            self.skipTest("local BGE model absent")
        corpus, _ = tiny_corpus()
        encoder = BGEEncoder(model_path, device="cpu", max_length=128)
        pids = sorted(corpus.paragraphs)
        texts = [f"{corpus.paragraphs[pid].title}\n{corpus.paragraphs[pid].text}" for pid in pids]
        fresh_docs = encoder.encode(texts, batch_size=3)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "embeddings.npz"
            np.savez(cache, pids=np.array(pids), embeddings=fresh_docs)
            search = LocalCorpusSearch(corpus, cache, encoder)
            cached_rank = search.rank_all("q1", "amber token")["dense"]
        query = encoder.encode(["amber token"], batch_size=1)[0]
        fresh_rank = sorted(pids, key=lambda pid: (-float(query @ fresh_docs[pids.index(pid)]), pid))
        self.assertEqual(cached_rank, fresh_rank)

    def test_correct_answer_without_full_support_is_zero(self):
        corpus, pids = tiny_corpus()
        result = terminal_reward(corpus, "q1", "amber", [pids[0]])
        self.assertTrue(result["answer_em"])
        self.assertFalse(result["full_selected_support_coverage"])
        self.assertEqual(result["reward"], 0)

    def test_reward_one_only_for_answer_and_full_selected_support(self):
        corpus, pids = tiny_corpus()
        self.assertEqual(terminal_reward(corpus, "q1", "amber", pids[:2])["reward"], 1)
        self.assertEqual(terminal_reward(corpus, "q1", "blue", pids[:2])["reward"], 0)

    def test_exact_set_reward_rejects_missing_support(self):
        corpus, pids = tiny_corpus()
        self.assertEqual(terminal_reward(corpus, "q1", "amber", [pids[0]])["reward"], 0)

    def test_exact_set_reward_rejects_all_support_plus_distractor(self):
        corpus, pids = tiny_corpus()
        result = terminal_reward(corpus, "q1", "amber", pids)
        self.assertTrue(result["full_selected_support_coverage"])
        self.assertFalse(result["exact_selected_support_set"])
        self.assertEqual(result["reward"], 0)
        self.assertEqual(terminal_reward_coverage_v1(corpus, "q1", "amber", pids)["reward"], 1)

    def test_exact_set_reward_accepts_exact_gold_and_correct_answer(self):
        corpus, pids = tiny_corpus()
        result = terminal_reward(corpus, "q1", "amber", pids[:2])
        self.assertTrue(result["exact_selected_support_set"])
        self.assertEqual(result["reward"], 1)

    def test_exact_set_reward_rejects_exact_gold_and_wrong_answer(self):
        corpus, pids = tiny_corpus()
        self.assertEqual(terminal_reward(corpus, "q1", "blue", pids[:2])["reward"], 0)

    def test_offline_runtime_constructs_no_external_llm_or_network_path(self):
        import agentflow.offline_musique as module

        source = inspect.getsource(module)
        for forbidden in ("import openai", "import requests", "import aiohttp", "urllib.request", "Web_RAG", "Wikipedia_Search"):
            self.assertNotIn(forbidden, source)
        self.assertFalse(LocalCorpusSearch.network_enabled)

    def test_synthetic_two_hop_trajectory_uses_both_modes(self):
        corpus, pids = tiny_corpus()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "embeddings.npz"
            tiny_cache(cache, corpus)
            search = LocalCorpusSearch(corpus, cache, TinyEncoder(), top_k=2)
            memory = CompactMemory()
            action1 = parse_decision('{"action":"search","query":"amber token"}')
            observation1 = search.search("q1", action1.query)
            chosen1 = next(row for row in observation1 if row["pid"] == pids[0])
            update1 = parse_evidence_update(json.dumps({"selected_evidence": [{"pid": pids[0], "quote": chosen1["text"][:18]}]}))
            memory.validate_and_update(action1.query, observation1, update1)
            action2 = parse_decision('{"action":"search","query":"box mass"}')
            observation2 = search.search("q1", action2.query)
            chosen2 = next(row for row in observation2 if row["pid"] == pids[1])
            update2 = parse_evidence_update(json.dumps({"selected_evidence": [{"pid": pids[1], "quote": chosen2["text"]}]}))
            memory.validate_and_update(action2.query, observation2, update2)
            answer = parse_decision('{"action":"answer","answer":"amber"}')
        self.assertEqual(terminal_reward(corpus, "q1", answer.answer, [row.pid for row in memory.evidence])["reward"], 1)

    def test_malformed_json_is_not_repaired(self):
        with self.assertRaises(ValidationError):
            parse_decision('```json\n{"action":"answer","answer":"amber"}\n```')
        with self.assertRaises(ValidationError):
            parse_evidence_update('{"selected_evidence":[],"answer":"amber"}')

    def test_official_source_mapping_drops_gold_actor_fields(self):
        row = {
            "id": "2hop__synthetic",
            "question": "What color is the token?",
            "paragraphs": [
                {"idx": 0, "title": "Token", "paragraph_text": "The token is amber.", "is_supporting": True},
                {"idx": 1, "title": "Box", "paragraph_text": "The box is small.", "is_supporting": False},
            ],
            "answer": "amber",
            "answer_aliases": ["amber color"],
            "answerable": True,
            "question_decomposition": [{"answer": "hidden", "paragraph_support_idx": 0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            corpus = build_corpus([source])
        actor = json.dumps(corpus.actor_question(row["id"]))
        self.assertNotIn("amber", actor)
        self.assertNotIn("support", actor)
        self.assertNotIn("decomposition", actor)


if __name__ == "__main__":
    unittest.main()
