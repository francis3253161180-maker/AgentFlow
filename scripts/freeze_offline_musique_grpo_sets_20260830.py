#!/usr/bin/env python3
"""Freeze hop-stratified MuSiQue dev-eval and train-pilot qid orders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import OfflineCorpus, sha256_file, stable_json_hash

_RUNNER_PATH = Path(__file__).with_name("run_offline_musique_protocol_smoke_20260830.py")
_RUNNER_SPEC = importlib.util.spec_from_file_location("offline_musique_frozen_runner", _RUNNER_PATH)
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)
stratified_subset = _RUNNER.stratified_subset


def describe(corpus: OfflineCorpus, qids: list[str]) -> dict:
    return {
        "qids": qids,
        "ordered_qids_sha256": stable_json_hash(qids),
        "count": len(qids),
        "hop_distribution": dict(sorted(Counter(corpus.questions[qid].hop_count for qid in qids).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-corpus", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    dev = OfflineCorpus.load(args.dev_corpus)
    train = OfflineCorpus.load(args.train_corpus)
    dev_qids = stratified_subset(dev, 64, args.seed)
    train_qids = stratified_subset(train, 128, args.seed)
    if set(dev_qids) & set(train_qids):
        raise RuntimeError("train/dev qid overlap")
    result = {
        "schema_version": 1,
        "seed": args.seed,
        "selection": "deterministic hop-stratified sampling only; no reward- or prompt-based selection",
        "dev_corpus": {"path": str(args.dev_corpus.resolve()), "sha256": sha256_file(args.dev_corpus)},
        "train_corpus": {"path": str(args.train_corpus.resolve()), "sha256": sha256_file(args.train_corpus)},
        "dev_eval": describe(dev, dev_qids),
        "train_pilot": describe(train, train_qids),
        "qid_overlap": 0,
        "rollout_n": 8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
