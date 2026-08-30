#!/usr/bin/env python3
"""Run the fixed Phase-B MuSiQue retrieval/query-sensitivity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import BGEEncoder, LocalCorpusSearch, OfflineCorpus, lexical_tokens, sha256_file


SEED = 20260830
STOPWORDS = {
    "a", "about", "after", "an", "and", "are", "as", "at", "be", "before", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why", "with",
}


def compact_query(question: str) -> str:
    terms = lexical_tokens(question)
    content = [term for term in terms if term not in STOPWORDS]
    dates = re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", question)
    selected = list(dict.fromkeys([*dates, *content]))[:8]
    return " ".join(selected) or " ".join(terms[:4])


def weak_query(question: str) -> str:
    terms = lexical_tokens(question)
    return " ".join(terms[:2]) or question[:12]


def stratified_subset(corpus: OfflineCorpus, size: int, seed: int) -> list[str]:
    by_hop = defaultdict(list)
    for qid, row in corpus.questions.items():
        by_hop[row.hop_count].append(qid)
    hops = [hop for hop in (2, 3, 4) if by_hop[hop]]
    if size < len(hops):
        raise ValueError("subset too small for hop strata")
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


def digest_rankings(rows) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--embedding-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/bge-small-en-v1.5"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    corpus = OfflineCorpus.load(args.corpus)
    encoder = BGEEncoder(args.model, device="cpu")
    search = LocalCorpusSearch(corpus, args.embeddings, encoder, top_k=2, rrf_k=60)
    qids = stratified_subset(corpus, args.subset_size, args.seed)
    query_sets = {}
    for index, qid in enumerate(qids):
        question = corpus.questions[qid].question
        other_question = corpus.questions[qids[(index + 1) % len(qids)]].question
        query_sets[qid] = {
            "full": question,
            "compact": compact_query(question),
            "weak": weak_query(question),
            "unrelated": other_question,
        }

    rankings = {}
    repeat_rankings = {}
    for qid in qids:
        rankings[qid] = {name: search.rank_all(qid, query) for name, query in query_sets[qid].items()}
        repeat_rankings[qid] = {name: search.rank_all(qid, query) for name, query in query_sets[qid].items()}
    repeatable = rankings == repeat_rankings

    metrics = {}
    full_coverage = {}
    for query_class in ("full", "compact", "weak", "unrelated"):
        metrics[query_class] = {}
        full_coverage[query_class] = {}
        for retriever in ("bm25", "dense", "rrf"):
            metrics[query_class][retriever] = {}
            full_coverage[query_class][retriever] = {}
            for k in (1, 2, 5):
                recalls = []
                covers = []
                for qid in qids:
                    support = corpus.scorer_record(qid).support_pids
                    retrieved = set(rankings[qid][query_class][retriever][:k])
                    recalls.append(len(retrieved & support) / len(support))
                    covers.append(support.issubset(retrieved))
                metrics[query_class][retriever][f"recall_at_{k}"] = sum(recalls) / len(recalls)
                full_coverage[query_class][retriever][f"coverage_at_{k}"] = sum(covers) / len(covers)

    full_margin = metrics["full"]["rrf"]["recall_at_2"] - metrics["unrelated"]["rrf"]["recall_at_2"]
    compact_margin = metrics["compact"]["rrf"]["recall_at_2"] - metrics["unrelated"]["rrf"]["recall_at_2"]
    rrf_r2 = metrics["full"]["rrf"]["recall_at_2"]
    component_r2 = [metrics["full"][name]["recall_at_2"] for name in ("bm25", "dense")]
    fusion_ok = not (rrf_r2 + 0.05 < min(component_r2))
    not_saturated = not all(metrics[name]["rrf"]["recall_at_2"] >= 0.95 for name in metrics)
    gate_checks = {
        "exact_repeatability": repeatable,
        "full_margin_at_least_0_20": full_margin >= 0.20,
        "compact_margin_at_least_0_20": compact_margin >= 0.20,
        "not_saturated_all_query_classes": not_saturated,
        "rrf_not_materially_worse_than_both_components": fusion_ok,
        "reranker_off": True,
    }
    parent_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    configuration = {"top_k": 2, "rrf_k": 60, "reranker": "off", "query_encoder_device": "cpu"}
    result = {
        "phase": "B",
        "iteration_id": "phase_b_v1",
        "parent_commit": parent_commit,
        "config_hash": hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        "hypothesis": "question-derived meaningful queries retrieve materially more gold support than unrelated queries with deterministic BM25+BGE RRF",
        "exact_changed_variables": ["initial baseline; no prior iteration"],
        "before_metrics": None,
        "seed": args.seed,
        "subset_size": len(qids),
        "hop_distribution": dict(sorted(Counter(corpus.questions[qid].hop_count for qid in qids).items())),
        "configuration": configuration,
        "artifacts": {
            "corpus_path": str(args.corpus.resolve()),
            "corpus_sha256": sha256_file(args.corpus),
            "embeddings_path": str(args.embeddings.resolve()),
            "embeddings_sha256": sha256_file(args.embeddings),
            "embedding_manifest_path": str(args.embedding_manifest.resolve()),
            "embedding_manifest_sha256": sha256_file(args.embedding_manifest),
        },
        "metrics": metrics,
        "full_support_coverage_diagnostic": full_coverage,
        "query_sensitivity_margins": {"full_minus_unrelated_recall_at_2": full_margin, "compact_minus_unrelated_recall_at_2": compact_margin},
        "repeatability": {
            "exact": repeatable,
            "first_digest": digest_rankings(rankings),
            "repeat_digest": digest_rankings(repeat_rankings),
        },
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
        "hypothesis_supported": gate_checks["full_margin_at_least_0_20"] and gate_checks["compact_margin_at_least_0_20"],
    }
    details = {
        "phase": "B",
        "iteration_id": "phase_b_v1",
        "seed": args.seed,
        "qids": qids,
        "queries": query_sets,
        # Rankings are local audit evidence. Gold support IDs are deliberately
        # omitted; aggregate scoring above is the only use of scorer labels.
        "rankings": rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    args.detail_output.write_text(json.dumps(details, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
