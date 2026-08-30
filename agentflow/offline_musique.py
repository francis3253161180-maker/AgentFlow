"""Deterministic offline MuSiQue environment for one trainable actor policy.

This module intentionally does not import AgentFlow's solver, role models, or
network tools.  The only semantic outputs are supplied by the caller's actor;
everything here is corpus isolation, retrieval, validation, or scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


PREPROCESSING_VERSION = "offline-musique-v1"
RRF_K = 60
TOP_K = 2
MAX_SEARCH_ACTIONS = 6
MAX_DECISION_TRANSITIONS = 7
MAX_EVIDENCE = 6
MAX_SEARCH_HISTORY = 6
MAX_QUOTE_CHARS = 300
ACTOR_CONTEXT_TOKENS = 4096
DECISION_MAX_NEW_TOKENS = 96
EVIDENCE_MAX_NEW_TOKENS = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paragraph_pid(title: str, text: str) -> str:
    return "p_" + stable_json_hash({"title": title, "text": text})[:24]


def lexical_tokens(text: str) -> list[str]:
    return [match.group(0).lower().replace("’", "'") for match in _TOKEN_RE.finditer(text)]


def normalize_quote_text(text: str) -> str:
    """Normalize whitespace and Unicode without changing words or punctuation."""
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    text = _ARTICLE_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchAction(StrictModel):
    action: Literal["search"]
    query: str = Field(min_length=1, max_length=512)


class AnswerAction(StrictModel):
    action: Literal["answer"]
    answer: str = Field(min_length=1, max_length=512)


DecisionAction = SearchAction | AnswerAction
DECISION_ADAPTER = TypeAdapter(DecisionAction)


class EvidenceSelection(StrictModel):
    pid: str = Field(pattern=r"^p_[0-9a-f]{24}$")
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)


class EvidenceUpdate(StrictModel):
    selected_evidence: list[EvidenceSelection] = Field(max_length=MAX_EVIDENCE)


def parse_decision(raw: str) -> DecisionAction:
    """Strictly parse one JSON decision; no fence stripping or repair."""
    return DECISION_ADAPTER.validate_json(raw)


def parse_evidence_update(raw: str) -> EvidenceUpdate:
    """Strictly parse one JSON evidence update; no fence stripping or repair."""
    return EvidenceUpdate.model_validate_json(raw)


@dataclass(frozen=True)
class Paragraph:
    pid: str
    title: str
    text: str

    def actor_payload(self, *, rank: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"pid": self.pid, "title": self.title, "text": self.text}
        if rank is not None:
            payload["rank"] = rank
        return payload


@dataclass(frozen=True)
class QuestionRecord:
    qid: str
    question: str
    candidate_pids: tuple[str, ...]
    hop_count: int

    def actor_payload(self) -> dict[str, Any]:
        return {"qid": self.qid, "question": self.question}


@dataclass(frozen=True)
class ScorerRecord:
    support_pids: frozenset[str]
    answers: tuple[str, ...]


@dataclass
class OfflineCorpus:
    paragraphs: dict[str, Paragraph]
    questions: dict[str, QuestionRecord]
    _scorer: dict[str, ScorerRecord] = field(repr=False)
    manifest: dict[str, Any] = field(default_factory=dict)

    def actor_question(self, qid: str) -> dict[str, Any]:
        record = self.questions[qid]
        # Candidate IDs are safe, but support/answer/decomposition data never is.
        return record.actor_payload()

    def scorer_record(self, qid: str) -> ScorerRecord:
        return self._scorer[qid]

    def save(self, path: str | Path) -> None:
        """Save a local scorer artifact.  This file must never be actor input."""
        payload = {
            "schema_version": 1,
            "preprocessing_version": PREPROCESSING_VERSION,
            "manifest": self.manifest,
            "paragraphs": {
                pid: {"title": p.title, "text": p.text} for pid, p in sorted(self.paragraphs.items())
            },
            "questions": {
                qid: {
                    "question": q.question,
                    "candidate_pids": list(q.candidate_pids),
                    "hop_count": q.hop_count,
                }
                for qid, q in sorted(self.questions.items())
            },
            "scorer_only": {
                qid: {"support_pids": sorted(s.support_pids), "answers": list(s.answers)}
                for qid, s in sorted(self._scorer.items())
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OfflineCorpus":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("preprocessing_version") != PREPROCESSING_VERSION:
            raise ValueError("offline corpus preprocessing version mismatch")
        paragraphs = {
            pid: Paragraph(pid=pid, title=row["title"], text=row["text"])
            for pid, row in payload["paragraphs"].items()
        }
        questions = {
            qid: QuestionRecord(
                qid=qid,
                question=row["question"],
                candidate_pids=tuple(row["candidate_pids"]),
                hop_count=int(row["hop_count"]),
            )
            for qid, row in payload["questions"].items()
        }
        scorer = {
            qid: ScorerRecord(frozenset(row["support_pids"]), tuple(row["answers"]))
            for qid, row in payload["scorer_only"].items()
        }
        return cls(paragraphs=paragraphs, questions=questions, _scorer=scorer, manifest=payload["manifest"])


def build_corpus(source_paths: Sequence[str | Path]) -> OfflineCorpus:
    paragraphs: dict[str, Paragraph] = {}
    questions: dict[str, QuestionRecord] = {}
    scorer: dict[str, ScorerRecord] = {}
    source_rows = []
    for source_path_value in source_paths:
        source_path = Path(source_path_value).resolve()
        row_count = 0
        with source_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("answerable") is not True:
                    continue
                qid = str(raw["id"])
                if qid in questions:
                    raise ValueError(f"duplicate qid: {qid}")
                candidate_pids: list[str] = []
                support_pids: set[str] = set()
                for paragraph in raw["paragraphs"]:
                    title = str(paragraph["title"])
                    text = str(paragraph["paragraph_text"])
                    pid = paragraph_pid(title, text)
                    existing = paragraphs.get(pid)
                    current = Paragraph(pid=pid, title=title, text=text)
                    if existing is not None and existing != current:
                        raise ValueError(f"pid collision: {pid}")
                    paragraphs[pid] = current
                    candidate_pids.append(pid)
                    if paragraph.get("is_supporting") is True:
                        support_pids.add(pid)
                if len(candidate_pids) != len(raw["paragraphs"]):
                    raise AssertionError("candidate mapping changed cardinality")
                answers = tuple(dict.fromkeys([str(raw["answer"]), *map(str, raw.get("answer_aliases", []))]))
                questions[qid] = QuestionRecord(
                    qid=qid,
                    question=str(raw["question"]),
                    candidate_pids=tuple(candidate_pids),
                    hop_count=len(support_pids),
                )
                scorer[qid] = ScorerRecord(frozenset(support_pids), answers)
                row_count += 1
        source_rows.append({"path": str(source_path), "sha256": sha256_file(source_path), "answerable_rows": row_count})
    manifest = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "sources": source_rows,
        "question_count": len(questions),
        "unique_paragraph_count": len(paragraphs),
        "pid_method": "p_ + first 24 hex chars of SHA256(canonical JSON {title,text})",
        "actor_visible_fields": ["qid", "question", "retrieved_pid", "retrieved_title", "retrieved_text"],
        "scorer_only_fields": ["support_pids", "answers"],
    }
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    return OfflineCorpus(paragraphs, questions, scorer, manifest)


class BGEEncoder:
    """Minimal local transformers BGE encoder with normalized CLS pooling."""

    pooling = "last_hidden_state[:,0] (CLS), L2-normalized"

    def __init__(self, model_path: str | Path, *, device: str = "cpu", max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_path = str(Path(model_path).resolve())
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True).to(device)
        self.model.eval()

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
                outputs.append(hidden.float().cpu().numpy())
        if not outputs:
            dimension = int(self.model.config.hidden_size)
            return np.empty((0, dimension), dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def model_inventory_hash(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).resolve()
    files = []
    for item in sorted(path.iterdir()):
        if item.is_file():
            files.append({"name": item.name, "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    return {"path": str(path), "files": files, "inventory_sha256": stable_json_hash(files)}


def build_embedding_cache(
    corpus: OfflineCorpus,
    encoder: BGEEncoder,
    cache_path: str | Path,
    manifest_path: str | Path,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    pids = sorted(corpus.paragraphs)
    texts = [f"{corpus.paragraphs[pid].title}\n{corpus.paragraphs[pid].text}" for pid in pids]
    embeddings = encoder.encode(texts, batch_size=batch_size)
    if embeddings.shape[0] != len(pids):
        raise RuntimeError("embedding row count mismatch")
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, pids=np.array(pids), embeddings=embeddings.astype(np.float32))
    manifest = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_manifest_sha256": corpus.manifest["manifest_sha256"],
        "paragraph_count": len(pids),
        "model": model_inventory_hash(encoder.model_path),
        "pooling": encoder.pooling,
        "dtype": "float32",
        "dimension": int(embeddings.shape[1]),
        "max_length": encoder.max_length,
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
    }
    manifest["manifest_sha256"] = stable_json_hash(manifest)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


class LocalCorpusSearch:
    """BM25 + cached BGE + deterministic RRF over one question's candidates."""

    network_enabled = False

    def __init__(
        self,
        corpus: OfflineCorpus,
        embedding_cache: str | Path,
        query_encoder: Any,
        *,
        top_k: int = TOP_K,
        rrf_k: int = RRF_K,
    ):
        cached = np.load(embedding_cache, allow_pickle=False)
        pids = [str(value) for value in cached["pids"].tolist()]
        vectors = cached["embeddings"].astype(np.float32, copy=False)
        if len(pids) != len(vectors) or set(pids) != set(corpus.paragraphs):
            raise ValueError("embedding cache/corpus identity mismatch")
        self.corpus = corpus
        self.query_encoder = query_encoder
        self.top_k = top_k
        self.rrf_k = rrf_k
        self._vectors = {pid: vectors[index] for index, pid in enumerate(pids)}

    @staticmethod
    def _bm25(query: str, paragraphs: Sequence[Paragraph]) -> dict[str, float]:
        docs = [lexical_tokens(f"{paragraph.title} {paragraph.text}") for paragraph in paragraphs]
        query_terms = lexical_tokens(query)
        n_docs = len(docs)
        avgdl = sum(map(len, docs)) / max(n_docs, 1)
        document_frequency = Counter(term for terms in docs for term in set(terms))
        scores: dict[str, float] = {}
        k1, b = 1.5, 0.75
        for paragraph, terms in zip(paragraphs, docs):
            frequencies = Counter(terms)
            score = 0.0
            for term in query_terms:
                df = document_frequency.get(term, 0)
                if not df:
                    continue
                inverse_document_frequency = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                frequency = frequencies[term]
                denominator = frequency + k1 * (1.0 - b + b * len(terms) / max(avgdl, 1e-9))
                score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
            scores[paragraph.pid] = score
        return scores

    @staticmethod
    def _rank(scores: dict[str, float]) -> list[str]:
        return sorted(scores, key=lambda pid: (-float(scores[pid]), pid))

    def rank_all(self, qid: str, query: str) -> dict[str, list[str]]:
        candidates = [self.corpus.paragraphs[pid] for pid in self.corpus.questions[qid].candidate_pids]
        bm25_scores = self._bm25(query, candidates)
        query_vector = self.query_encoder.encode([query], batch_size=1)[0]
        dense_scores = {p.pid: float(np.dot(query_vector, self._vectors[p.pid])) for p in candidates}
        bm25 = self._rank(bm25_scores)
        dense = self._rank(dense_scores)
        rrf_scores: dict[str, float] = {p.pid: 0.0 for p in candidates}
        for ranking in (bm25, dense):
            for rank, pid in enumerate(ranking, start=1):
                rrf_scores[pid] += 1.0 / (self.rrf_k + rank)
        return {"bm25": bm25, "dense": dense, "rrf": self._rank(rrf_scores)}

    def search(self, qid: str, query: str) -> list[dict[str, Any]]:
        ranking = self.rank_all(qid, query)["rrf"][: self.top_k]
        return [self.corpus.paragraphs[pid].actor_payload(rank=index) for index, pid in enumerate(ranking, 1)]


@dataclass(frozen=True)
class MemoryEvidence:
    pid: str
    quote: str


@dataclass(frozen=True)
class SearchHistoryEntry:
    query: str
    outcome: Literal["useful", "no_useful_evidence"]


@dataclass
class CompactMemory:
    evidence: list[MemoryEvidence] = field(default_factory=list)
    search_history: list[SearchHistoryEntry] = field(default_factory=list)

    def actor_payload(self) -> dict[str, Any]:
        return {
            "validated_evidence": [f"[{row.pid}] {row.quote}" for row in self.evidence[-MAX_EVIDENCE:]],
            "search_history": [
                {"query": row.query, "outcome": row.outcome} for row in self.search_history[-MAX_SEARCH_HISTORY:]
            ],
        }

    def validate_and_update(
        self,
        query: str,
        observation: Sequence[dict[str, Any]],
        update: EvidenceUpdate,
    ) -> dict[str, Any]:
        observed = {str(row["pid"]): normalize_quote_text(str(row["text"])) for row in observation}
        existing = {row.pid for row in self.evidence}
        accepted: list[MemoryEvidence] = []
        rejected: list[dict[str, str]] = []
        seen_in_update: set[str] = set()
        for selection in update.selected_evidence:
            normalized_quote = normalize_quote_text(selection.quote)
            reason = None
            if selection.pid not in observed:
                reason = "pid_not_in_current_observation"
            elif selection.pid in existing or selection.pid in seen_in_update:
                reason = "duplicate_evidence"
            elif normalized_quote not in observed[selection.pid]:
                reason = "quote_not_exact_normalized_substring"
            if reason is not None:
                rejected.append({"pid": selection.pid, "reason": reason})
                continue
            accepted.append(MemoryEvidence(selection.pid, normalized_quote))
            seen_in_update.add(selection.pid)
        remaining = max(0, MAX_EVIDENCE - len(self.evidence))
        if len(accepted) > remaining:
            for row in accepted[remaining:]:
                rejected.append({"pid": row.pid, "reason": "memory_evidence_cap"})
            accepted = accepted[:remaining]
        self.evidence.extend(accepted)
        outcome: Literal["useful", "no_useful_evidence"] = "useful" if accepted else "no_useful_evidence"
        self.search_history.append(SearchHistoryEntry(query=query, outcome=outcome))
        self.search_history = self.search_history[-MAX_SEARCH_HISTORY:]
        return {"accepted": [row.__dict__ for row in accepted], "rejected": rejected, "outcome": outcome}


DECISION_SYSTEM = """[MODE: DECISION]
You are the only intelligent policy in an offline question-answering environment.
Input contains only a question, compact validated memory, and remaining budgets.
Output exactly one JSON object matching one of these forms:
{"action":"search","query":"..."}
{"action":"answer","answer":"..."}
Use only the compact memory for evidence. Do not include commentary or markdown.
Structural examples (format only):
Input: Question: What material is the sample made from? Memory: none. Search budget: 2. Decision budget: 3.
Output: {"action":"search","query":"sample material composition"}
Input: Question: What color is the token? Memory: [p_demo] The token is amber. Search budget: 0. Decision budget: 1.
Output: {"action":"answer","answer":"amber"}"""


EVIDENCE_SYSTEM = """[MODE: EVIDENCE_UPDATE]
Select only useful exact evidence from the current offline search observation.
Output exactly one JSON object: {"selected_evidence":[{"pid":"...","quote":"..."}, ...]}
Select at most one entry per observed paragraph (at most two entries total). A useful fact may answer the question directly or resolve an intermediate entity, date, or relation needed by the question. Return an empty list only when neither paragraph contains such a fact.
Every pid must be in the current observation. Use the shortest sufficient exact sentence or clause for each quote, preferably at most 240 characters and never more than 300 characters. Quotes must be exact substrings apart from Unicode/whitespace normalization.
The response must begin with {"selected_evidence":[ and end with the exact outer suffix ]}. Stop immediately after that suffix; never omit its final }. Do not plan or answer. Do not include commentary, markdown, literal newline escapes, or text after the final }.
Structural examples (format only):
Input: Question: What color is the token? Query: token color. Observation: [p_demo] The token is amber.
Output: {"selected_evidence":[{"pid":"p_demo","quote":"The token is amber."}]}
Input: Question: What color is the token? Query: token weight. Observation: [p_other] The box weighs two grams.
Output: {"selected_evidence":[]}"""


def decision_prompt(question: str, memory: CompactMemory, *, searches_left: int, decisions_left: int) -> str:
    payload = {
        "question": question,
        "compact_memory": memory.actor_payload(),
        "remaining_search_budget": searches_left,
        "remaining_decision_budget": decisions_left,
    }
    return DECISION_SYSTEM + "\nCurrent input:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def evidence_prompt(
    question: str,
    memory: CompactMemory,
    query: str,
    observation: Sequence[dict[str, Any]],
) -> str:
    payload = {
        "question": question,
        "compact_memory": memory.actor_payload(),
        "immediately_preceding_query": query,
        "current_search_observation": list(observation),
    }
    return EVIDENCE_SYSTEM + "\nCurrent input:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def terminal_reward(corpus: OfflineCorpus, qid: str, answer: str, selected_pids: Iterable[str]) -> dict[str, Any]:
    scorer = corpus.scorer_record(qid)
    normalized = normalize_answer(answer)
    answer_em = any(normalized == normalize_answer(alias) for alias in scorer.answers)
    selected = set(selected_pids)
    full_support = scorer.support_pids.issubset(selected)
    reward = int(answer_em and full_support)
    return {
        "reward": reward,
        "answer_em": answer_em,
        "full_selected_support_coverage": full_support,
        "selected_support_count": len(selected & scorer.support_pids),
        "gold_support_count": len(scorer.support_pids),
    }


def transition_record(
    *,
    trajectory_id: str,
    qid: str,
    rollout_id: str,
    transition_index: int,
    mode: Literal["DECISION", "EVIDENCE_UPDATE"],
    prompt: str,
    response: str,
    compact_memory_before: dict[str, Any],
    semantic_output: dict[str, Any] | None,
    observation_refs: Sequence[str],
    validation_result: dict[str, Any],
    compact_memory_after: dict[str, Any],
    token_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "qid": qid,
        "rollout_id": rollout_id,
        "transition_index": transition_index,
        "mode": mode,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response": response,
        "token_logprob_metadata": token_metadata or {},
        "compact_memory_before": compact_memory_before,
        "semantic_output": semantic_output,
        "observation_refs": list(observation_refs),
        "validation_result": validation_result,
        "compact_memory_after": compact_memory_after,
    }
