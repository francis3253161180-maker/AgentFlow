#!/usr/bin/env python3
"""Build the versioned local MuSiQue scorer corpus and one BGE cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.offline_musique import BGEEncoder, build_corpus, build_embedding_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/models/bge-small-en-v1.5"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "offline_musique_corpus_v1.json"
    cache_path = args.output_dir / "offline_musique_bge_v1.npz"
    embedding_manifest_path = args.output_dir / "offline_musique_bge_v1_manifest.json"
    corpus = build_corpus(args.source)
    corpus.save(corpus_path)
    print(json.dumps({"event": "corpus_saved", **corpus.manifest, "path": str(corpus_path)}, sort_keys=True), flush=True)
    encoder = BGEEncoder(args.model, device=args.device)
    embedding_manifest = build_embedding_cache(
        corpus,
        encoder,
        cache_path,
        embedding_manifest_path,
        batch_size=args.batch_size,
    )
    print(json.dumps({"event": "embedding_cache_saved", **embedding_manifest}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
