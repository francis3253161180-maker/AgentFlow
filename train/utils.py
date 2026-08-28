import os
import re
import unicodedata
from dataclasses import dataclass
from pydantic import BaseModel

try:
    import sympy
except ImportError:  # pragma: no cover - AgentFlow includes sympy in production
    sympy = None


class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool

_default_reward_scorer = None

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _answer_candidates(value: str) -> list[str]:
    """Return the most likely final-answer text before the full response."""
    text = str(value).strip()
    tagged = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        return [tagged[-1].strip()]

    markers = list(
        re.finditer(
            r"(?:\*\*\s*)?(?:final\s+answer|answer|conclusion|result)"
            r"(?:\s*\*\*)?\s*(?::|is)\s*",
            text,
            flags=re.IGNORECASE,
        )
    )
    if markers:
        return [text[markers[-1].end() :].strip()]
    return [text]


def _compact_candidates(value: str) -> set[str]:
    """Preserve the previous compact normalization for short math answers."""
    text = str(value).lower()
    text = re.sub(r"<answer>|</answer>|\\boxed\s*", "", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    forms = {text}
    if "=" in text:
        forms.add(text.rsplit("=", 1)[-1])
    return {re.sub(r"[^a-z0-9./+\-√]", "", form) for form in forms}


def _normalized_tokens(value: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = text.replace("’", "'")
    text = re.sub(r"['’]s\b", "", text)
    return re.findall(r"[a-z0-9]+", text)


def _negated_token(tokens: list[str], start: int) -> bool:
    prefix = tokens[max(0, start - 4) : start]
    return bool(prefix and prefix[-1] in {"not", "never", "no", "without"})


def _negated_phrase(tokens: list[str], start: int, width: int) -> bool:
    if _negated_token(tokens, start):
        return True
    suffix = tokens[start + width : start + width + 3]
    return len(suffix) >= 2 and suffix[0] in {"is", "was", "are", "were"} and suffix[1] == "not"


def _phrase_match(groundtruth: str, answer: str) -> bool:
    truth_tokens = _normalized_tokens(groundtruth)
    answer_tokens = _normalized_tokens(answer)
    if not truth_tokens or not answer_tokens:
        return False

    width = len(truth_tokens)
    for start in range(len(answer_tokens) - width + 1):
        if answer_tokens[start : start + width] == truth_tokens:
            if not _negated_phrase(answer_tokens, start, width):
                return True

    # Possessives often turn a benchmark phrase such as "Chicago's Grant
    # Park" into a grammatical answer such as "Chicago ... in Grant Park".
    # Permit an ordered, bounded token span only when the ground truth itself
    # contains a possessive; this avoids general bag-of-words matching.
    if re.search(r"['’]s\b", str(groundtruth)) and width > 1:
        for start, token in enumerate(answer_tokens):
            if token != truth_tokens[0] or _negated_phrase(answer_tokens, start, 1):
                continue
            pos = start
            for wanted in truth_tokens[1:]:
                try:
                    pos = answer_tokens.index(wanted, pos + 1)
                except ValueError:
                    break
            else:
                if pos - start <= 32:
                    return True
    return False


def _yes_no(value: str) -> bool | None:
    text = re.sub(r"[*_`]+", "", str(value)).strip().lower()
    if re.match(r"^(?:the\s+)?(?:answer|conclusion|result)\s+is\s+not\s+(?:yes|no)\b", text):
        return None
    match = re.match(r"^(?:the\s+)?(?:answer|conclusion|result)\s*(?::|is)\s*(yes|no)\b", text)
    if match:
        return match.group(1) == "yes"
    match = re.match(r"^(yes|no)\b", text)
    return None if not match else match.group(1) == "yes"


def _date_values(value: str) -> list[tuple[int, int, int, int, int]]:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    found: list[tuple[int, int, int, int, int]] = []
    month_names = "|".join(_MONTHS)
    patterns = [
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
        rf"\b(\d{{1,2}})\s+({month_names})\s+(\d{{4}})\b",
    ]
    for pattern_index, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text):
            if pattern_index == 0:
                month, day, year = match.group(1), match.group(2), match.group(3)
            else:
                day, month, year = match.group(1), match.group(2), match.group(3)
            found.append((int(year), _MONTHS[month], int(day), match.start(), match.end()))
    return found


def _date_match(groundtruth: str, candidates: list[str]) -> bool | None:
    truth_dates = _date_values(groundtruth)
    if not truth_dates:
        return None
    target = truth_dates[0][:3]
    for candidate in candidates:
        for year, month, day, start, _ in _date_values(candidate):
            if (year, month, day) == target:
                prefix = candidate[max(0, start - 24) : start].lower()
                if not re.search(r"\b(?:not|never|wrong|incorrectly)\s*$", prefix):
                    return True
    return False


def _numeric_target(value: str) -> str | None:
    compact = re.sub(r"\s+", "", str(value)).strip(".,:;()[]")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", compact):
        return compact
    return None


def _numeric_match(groundtruth: str, candidates: list[str]) -> bool | None:
    target = _numeric_target(groundtruth)
    if target is None:
        return None
    for candidate in candidates:
        matches = []
        for match in re.finditer(r"(?<![\w.])[-+]?\d+(?:\.\d+)?", candidate):
            end = match.end()
            following = candidate[end : end + 2]
            if following.startswith(".") and len(following) > 1 and following[1].isdigit():
                continue
            if following.startswith(".") and len(following) == 1:
                pass
            elif following and (following[0].isalnum() or following[0] == "_"):
                continue
            matches.append(match)
        if not matches:
            continue
        for match in matches:
            if match.group(0) != target:
                continue
            prefix = candidate[max(0, match.start() - 24) : match.start()].lower()
            if not re.search(r"\b(?:not|never|wrong|incorrectly)\s*$", prefix):
                # Do not accept a number merely because it shares a year or an
                # intermediate value. A concise answer or explicit answer
                # segment is required.
                numbers = [item.group(0) for item in matches]
                explicit_value = bool(
                    re.search(
                        r"(?:there\s+(?:is|are)|answer|result|value|equals?)\s*(?:is|are|:|=)?\s*$",
                        prefix,
                    )
                )
                if (len(set(numbers)) == 1 and numbers[0] == target) or explicit_value:
                    return True
    return False


def _math_expressions(value: str) -> list[str]:
    text = str(value).strip()
    expressions = [text]
    # Responses commonly put the final claim inside TeX delimiters.  The
    # previous extractor only saw the whole response and the text after its
    # last '=', so a truthful claim such as ``Yes, it is true that
    # \(a=b\)`` was treated as an unsafe mismatch.  Extract delimited spans
    # while retaining the old compact/full-expression candidates.
    expressions.extend(
        match.strip()
        for pattern in (
            r"\\\((.*?)\\\)",
            r"\\\[(.*?)\\\]",
            r"\$\$(.*?)\$\$",
            r"\$(.*?)\$",
        )
        for match in re.findall(pattern, text, flags=re.DOTALL)
        if match.strip()
    )
    expressions.extend(
        match.strip()
        for match in re.findall(r"\\boxed\s*\{([^{}]+)\}", text, flags=re.DOTALL)
        if match.strip()
    )
    # Also capture short bare equations in a concise answer such as
    # ``Yes, a=b``.  The deterministic decision layer separately rejects
    # unmarked multi-equation candidate lists.
    expressions.extend(
        match.strip()
        for match in re.findall(
            r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*\s*=\s*"
            r"(?:[-+]?(?:\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]*)))(?![A-Za-z0-9_])",
            text,
        )
        if match.strip()
    )
    # Keep the right-hand side of each extracted equation as a candidate too;
    # a labelled equation such as ``|H_n| = 2^{n+1}`` has a non-sympifiable
    # left side but a perfectly safe scalar conclusion.
    for expression in list(expressions):
        if "=" in expression:
            expressions.append(expression.rsplit("=", 1)[-1].strip())
    expressions.extend(re.findall(r"\\(?:d?frac)\s*\{[^{}]+\}\s*\{[^{}]+\}", text))
    expressions.extend(re.findall(r"\\sqrt\s*\{[^{}]+\}|√\s*[0-9]+", text))
    expressions.extend(re.findall(r"(?<![A-Za-z])[-+]?\d+\s*/\s*\d+", text))
    # A trailing sentence terminator is prose around the formula, not part of
    # the formula.  Only trim punctuation at the edges; do not normalize the
    # interior, where it may carry mathematical meaning.
    cleaned = []
    for expr in expressions:
        expr = re.sub(r"\\[)\]]\s*$", "", expr)
        expr = expr.strip(" $.,;:")
        if expr:
            cleaned.append(expr)
    return list(dict.fromkeys(cleaned))


def _to_sympy(value: str):
    if sympy is None:
        return None
    text = str(value).strip()
    text = re.sub(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"√\s*([A-Za-z0-9]+)", r"sqrt(\1)", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("^", "**")
    text = re.sub(r"\\text\s*\{([^{}]+)\}", r"\1", text)
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "")
    text = re.sub(r"\b([A-Za-z])_(\d+)\b", r"\1\2", text)
    text = re.sub(r"\s*\]\s*$", "", text)
    text = re.sub(r"\)\s*\.\s*$", ")", text)
    # Parse a simple equality as a relation rather than asking sympify to
    # parse Python assignment syntax.  This is intentionally limited to one
    # plain '=' and leaves inequalities/compound statements on the judge
    # route.
    if text.count("=") == 1 and not re.search(r"[<>!=]=", text):
        lhs, rhs = (part.strip() for part in text.split("=", 1))
        if lhs and rhs:
            parsed_lhs = _to_sympy(lhs)
            parsed_rhs = _to_sympy(rhs)
            if parsed_lhs is not None and parsed_rhs is not None:
                return sympy.Eq(parsed_lhs, parsed_rhs)
    text = text.strip(" $\\.,;:")
    if not text or len(text) > 160 or re.search(r"[^A-Za-z0-9_+\-*/()., ]", text):
        return None
    try:
        return sympy.sympify(text, locals={"sqrt": sympy.sqrt})
    except (TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return None


def _math_match(groundtruth: str, candidates: list[str]) -> bool:
    if sympy is None:
        return False
    # For an equality target, a shared right-hand side is not evidence of an
    # equivalent equation (``a=b`` must not match ``c=b``).  The SymPy path
    # below handles the relation itself; compact matching remains useful for
    # scalar/formula targets and preserves the established fraction behavior.
    if "=" not in str(groundtruth):
        truth_compact = _compact_candidates(groundtruth)
        for candidate in candidates:
            if truth_compact & _compact_candidates(candidate):
                return True
    raw_truth_exprs = _math_expressions(groundtruth)
    relation_truth_exprs = [
        _to_sympy(expr) for expr in raw_truth_exprs if "=" in expr
    ]
    relation_truth_exprs = [
        expr for expr in relation_truth_exprs if isinstance(expr, sympy.Equality)
    ]
    # Prefer a parseable equality relation over the auxiliary RHS candidate;
    # otherwise ``a=b`` could incorrectly match ``c=b`` through the shared
    # scalar b.  If the left side is not parseable (for example |H_n|), the
    # RHS fallback remains available.
    truth_exprs = relation_truth_exprs or [_to_sympy(expr) for expr in raw_truth_exprs]
    truth_exprs = [expr for expr in truth_exprs if expr is not None]
    if not truth_exprs:
        return False
    for candidate in candidates:
        for answer_expr in _math_expressions(candidate):
            parsed_answer = _to_sympy(answer_expr)
            if parsed_answer is None:
                continue
            if relation_truth_exprs and not isinstance(parsed_answer, sympy.Equality):
                continue
            for parsed_truth in truth_exprs:
                try:
                    # ``sympify('a=b')`` is not a numeric expression.  Treat
                    # two simple equality claims as equivalent when their
                    # left-minus-right forms agree up to orientation.  This
                    # covers algebraic answer statements without accepting a
                    # merely shared token.
                    if all(
                        isinstance(item, sympy.Equality)
                        for item in (parsed_answer, parsed_truth)
                    ):
                        answer_delta = sympy.expand(parsed_answer.lhs - parsed_answer.rhs)
                        truth_delta = sympy.expand(parsed_truth.lhs - parsed_truth.rhs)
                        if bool(
                            sympy.simplify(answer_delta - truth_delta) == 0
                            or sympy.simplify(answer_delta + truth_delta) == 0
                        ):
                            return True
                    elif bool(sympy.simplify(parsed_answer - parsed_truth) == 0):
                        return True
                except (TypeError, ValueError):
                    continue
    return False


def _numeric_expression_decision(target: str, candidate: str) -> bool | None:
    """Safely compare one explicit arithmetic answer with a scalar target.

    This is intentionally limited to an explicit answer candidate.  It fixes
    the common ``GT=24, answer=(... arithmetic ...)`` case without promoting
    an arbitrary numeric token embedded in open-ended prose.
    """
    if sympy is None or not re.search(r"[+*/×÷()-]", str(candidate)):
        return None
    parsed_target = _to_sympy(target)
    if parsed_target is None or isinstance(parsed_target, sympy.Equality):
        return None
    parsed_any = False
    for expression in _math_expressions(candidate):
        if not re.search(r"[+*/×÷()-]", expression):
            continue
        # A pure arithmetic expression cannot contain prose identifiers.  A
        # TeX fraction/root is still accepted by _to_sympy below.
        if re.search(r"[A-Za-z]", expression) and not re.search(r"\\(?:frac|d?frac|sqrt)", expression):
            continue
        parsed = _to_sympy(expression)
        if parsed is None or isinstance(parsed, sympy.Equality):
            continue
        parsed_any = True
        try:
            if bool(sympy.simplify(parsed - parsed_target) == 0):
                return True
        except (TypeError, ValueError):
            continue
    return False if parsed_any else None


def _looks_mathematical(value: str) -> bool:
    text = str(value)
    return bool(
        re.search(r"\\(?:d?frac|sqrt)|√|\^|=", text)
        or re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", text)
        or (
            re.fullmatch(r"\s*[0-9\s()+\-*/×÷.]+\s*", text)
            and re.search(r"[+*/×÷()-]", text)
        )
        or re.search(r"[A-Za-z]\w*\s*[+*/^=]\s*[-+A-Za-z0-9]", text)
    )


@dataclass(frozen=True)
class DeterministicDecision:
    """A high-confidence local verdict, or ``None`` when semantic review is needed."""

    value: bool | None
    reason: str


def _candidate_is_explicit(answer: str) -> bool:
    text = str(answer)
    return bool(
        re.search(r"<answer>.*?</answer>", text, flags=re.IGNORECASE | re.DOTALL)
        or re.search(
            r"(?:\*\*\s*)?(?:final\s+answer|answer|conclusion|result)"
            r"(?:\s*\*\*)?\s*(?::|is)\s*",
            text,
            flags=re.IGNORECASE,
        )
        or re.search(r"\\boxed\s*\{", text)
    )


def _answer_has_conflict(answer: str) -> bool:
    """Detect cues that make an otherwise matching mention unsafe to trust locally."""
    text = unicodedata.normalize("NFKC", str(answer)).lower()
    return bool(
        re.search(
            r"\b(?:but|however|although|instead|rather|actually|correction|corrected|"
            r"first|initially|thought|incorrect|incorrectly|wrong|false|not)\b",
            text,
        )
    )


def _yes_no_has_conflict(answer: str) -> bool:
    text = unicodedata.normalize("NFKC", str(answer)).lower()
    return bool(
        re.search(
            r"\b(?:but|however|although|instead|rather|actually|correction|corrected|"
            r"first|initially|thought|not)\b",
            text,
        )
    )


def _phrase_is_rejected(groundtruth: str, answer: str) -> bool:
    """Return true when every target phrase occurrence is locally negated/rejected."""
    truth_tokens = _normalized_tokens(groundtruth)
    answer_tokens = _normalized_tokens(answer)
    width = len(truth_tokens)
    if not truth_tokens or not answer_tokens:
        return False

    occurrences = [
        start
        for start in range(len(answer_tokens) - width + 1)
        if answer_tokens[start : start + width] == truth_tokens
    ]
    if not occurrences:
        return False

    for start in occurrences:
        before = answer_tokens[max(0, start - 5) : start]
        after = answer_tokens[start + width : start + width + 7]
        if _negated_phrase(answer_tokens, start, width):
            continue
        if len(before) >= 2 and before[-2:] == ["not", "in"]:
            continue
        if "not" in before[-3:] or "never" in before[-3:]:
            continue
        if after[:2] == ["but", "not"] or after[:2] == ["and", "not"]:
            continue
        if after[:2] in (["is", "incorrect"], ["was", "incorrect"]):
            continue
        return False
    return True


def _numeric_occurrences(value: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(r"(?<![\w.])[-+]?\d+(?:\.\d+)?", str(value))
        if not (
            str(value)[match.end() : match.end() + 2].startswith(".")
            and len(str(value)[match.end() : match.end() + 2]) > 1
            and str(value)[match.end() + 1 : match.end() + 2].isdigit()
        )
    ]


def deterministic_decision(groundtruth: str, answer_extracted: str) -> DeterministicDecision:
    """Return only high-confidence local decisions.

    ``None`` is intentional: it is the routing signal for the semantic judge.
    The function does not attempt open-ended entity or discourse resolution.
    """
    truth = str(groundtruth).strip()
    answer = str(answer_extracted).strip()
    candidates = _answer_candidates(answer)
    explicit = _candidate_is_explicit(answer)
    candidate = candidates[-1] if candidates else answer
    candidate_tokens = _normalized_tokens(candidate)
    truth_tokens = _normalized_tokens(truth)

    if not truth or not answer:
        return DeterministicDecision(False, "empty_input")

    if truth.lower() in {"yes", "no"}:
        expected = truth.lower() == "yes"
        if re.match(
            r"^(?:the\s+)?(?:answer|conclusion|result)\s+is\s+not\s+(?:yes|no)\b",
            candidate,
            flags=re.IGNORECASE,
        ):
            return DeterministicDecision(False, "explicit_yes_no_negation")
        answer_value = _yes_no(candidate)
        yes_no_tokens = [token for token in _normalized_tokens(candidate) if token in {"yes", "no"}]
        if answer_value is not None and len(set(yes_no_tokens)) == 1 and (
            explicit or len(candidate_tokens) <= 12
        ) and not _yes_no_has_conflict(candidate):
            return DeterministicDecision(answer_value == expected, "unambiguous_yes_no")
        return DeterministicDecision(None, "ambiguous_yes_no")

    truth_dates = _date_values(truth)
    if truth_dates:
        target = truth_dates[0][:3]
        for candidate_text in candidates:
            values = _date_values(candidate_text)
            unique_values = {(year, month, day) for year, month, day, _, _ in values}
            if len(unique_values) != 1:
                if len(unique_values) > 1:
                    return DeterministicDecision(None, "conflicting_dates")
                continue
            value = next(iter(unique_values))
            if value == target:
                if _phrase_is_rejected(truth, candidate_text):
                    return DeterministicDecision(False, "negated_date")
                return DeterministicDecision(True, "complete_date_match")
            if explicit or len(_normalized_tokens(candidate_text)) <= 12:
                return DeterministicDecision(False, "complete_date_mismatch")
        return DeterministicDecision(None, "uncertain_date")

    numeric_target = _numeric_target(truth)
    if numeric_target is not None and explicit:
        numeric_expression = _numeric_expression_decision(numeric_target, candidate)
        if numeric_expression is not None and not _answer_has_conflict(candidate):
            return DeterministicDecision(
                numeric_expression,
                "proved_numeric_expression" if numeric_expression else "numeric_expression_mismatch",
            )
    if numeric_target is not None:
        for candidate_text in candidates:
            values = _numeric_occurrences(candidate_text)
            unique_values = set(values)
            if len(unique_values) != 1:
                if len(unique_values) > 1:
                    return DeterministicDecision(None, "conflicting_numbers")
                continue
            value = next(iter(unique_values))
            if value == numeric_target:
                if _answer_has_conflict(candidate_text):
                    return DeterministicDecision(None, "conflicting_number_language")
                return DeterministicDecision(True, "complete_number_match")
            if explicit or len(_normalized_tokens(candidate_text)) <= 12:
                return DeterministicDecision(False, "complete_number_mismatch")
        return DeterministicDecision(None, "uncertain_number")

    if _looks_mathematical(truth):
        math_candidate = candidate
        math_conflict = _answer_has_conflict(math_candidate)
        equation_count = len(re.findall(r"(?<![<>=])=(?!=)", math_candidate))
        short_ambiguous_list = bool(
            equation_count > 1
            and (
                not explicit
                or (
                    len(candidate_tokens) <= 64
                    and re.search(
                        r"\b(?:alternative|alternatively|possibilit\w*|candidate|either|or)\b",
                        math_candidate,
                        re.IGNORECASE,
                    )
                )
            )
        )
        if short_ambiguous_list:
            return DeterministicDecision(None, "ambiguous_math_candidates")
        if not math_conflict and _math_match(truth, candidates):
            if explicit or len(_normalized_tokens(math_candidate)) <= 24:
                return DeterministicDecision(True, "proved_math_equivalence")
        if not math_conflict and _looks_mathematical(math_candidate) and (
            explicit or len(_normalized_tokens(math_candidate)) <= 24
        ):
            return DeterministicDecision(False, "safe_math_mismatch")
        return DeterministicDecision(None, "uncertain_math")

    normalized_truth = _normalized_tokens(truth)
    normalized_candidate = _normalized_tokens(candidate)
    if normalized_truth and normalized_candidate == normalized_truth:
        return DeterministicDecision(True, "normalized_exact_match")

    if _phrase_is_rejected(truth, candidate):
        if explicit and len(candidate_tokens) <= 32:
            return DeterministicDecision(False, "explicit_phrase_rejection")
        return DeterministicDecision(None, "rejected_phrase_in_open_answer")

    if explicit and _phrase_match(truth, candidate) and not _answer_has_conflict(candidate):
        return DeterministicDecision(True, "explicit_phrase_match")

    # A phrase in an unrestricted natural-language response is deliberately not
    # promoted to a reward. It may be a citation, a rejected location, or one
    # of several candidates; the semantic judge must read the full question and
    # response to resolve that discourse.
    return DeterministicDecision(None, "open_natural_language")


def deterministic_fallback_score(groundtruth: str, answer_extracted: str) -> bool:
    """Return the local high-confidence verdict; unresolved cases are false."""
    return deterministic_decision(groundtruth, answer_extracted).value is True


def compute_score(question: str, groundtruth: str, answer_extracted: str,) -> bool:
    """Score with high-confidence local rules, then the configured DeepSeek judge."""
    global _default_reward_scorer
    if _default_reward_scorer is None:
        try:
            from train.reward_judge import HybridRewardScorer
        except ModuleNotFoundError:  # direct execution from the train directory
            from reward_judge import HybridRewardScorer

        _default_reward_scorer = HybridRewardScorer.from_environment()
    result = _default_reward_scorer.score_with_metadata(
        question, groundtruth, answer_extracted
    )
    if os.getenv("AGENTFLOW_REWARD_SCORER_LOG") == "1":
        print(
            "HYBRID_REWARD_EVENT "
            f"route={result.route} score={int(result.score)} "
            f"cache_hit={int(result.cache_hit)} reason={result.reason} "
            f"error={result.judge_error or 'none'} "
            f"latency_ms={result.latency_seconds * 1000:.3f}",
            flush=True,
        )
    return result.score


def eval(question: str, groundtruth: any, answer_extracted: any, val: bool = False) -> float:
    """
    Evaluates if the extracted answer is correct by calling an LLM judge (gpt-4o).
    It strip(), and matches the final answer.
    """
    question_str = str(question)
    groundtruth_str = str(groundtruth)
    answer_extracted_str = str(answer_extracted)

    is_correct = compute_score(question_str, groundtruth_str, answer_extracted_str)
    
    return 1.0 if is_correct else 0.0

async def main():
    # ==============================================================================
    # ==============================================================================
    print("--- Running Simple Case ---")
    simple_question = "What is the capital of France?\nA) Berlin\nB) Madrid\nC) Paris\nD) Rome"
    simple_groundtruth = "C"
    simple_model_answer = "The correct answer is C."
    score1 = eval(simple_question, simple_groundtruth, simple_model_answer)
    print(f"Question: {simple_question}")
    print(f"Model Answer: '{simple_model_answer}'")
    print(f"Ground Truth: '{simple_groundtruth}'")
    print(f"==> Score: {score1}\n") # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with LaTeX Formula ---")
    latex_question = r"""
Calculate the definite integral of $f(x) = 2x$ from $x=1$ to $x=3$.
A) 4
B) 6
C) 8
D) 10
"""
    latex_groundtruth = "C"
    latex_model_answer = r"""
To solve this, we need to compute the integral $\int_{1}^{3} 2x \,dx$.
The antiderivative of $2x$ is $x^2$. 
Using the Fundamental Theorem of Calculus, we evaluate this at the bounds:
$F(b) - F(a) = 3^2 - 1^2 = 9 - 1 = 8$.
"""
    score2 = eval(latex_question, latex_groundtruth, latex_model_answer)
    print(f"Question: {latex_question.strip()}")
    print(f"Model Answer: '{latex_model_answer.strip()}'")
    print(f"Ground Truth: '{latex_groundtruth}'")
    print(f"==> Score: {score2}\n") # 1.0

    # ==============================================================================
    # ==============================================================================
    print("--- Running Case with Multiple Intermediate Answers ---")
    multi_answer_question = """
A project has two phases. Phase 1 costs $5,000 and takes 3 months. Phase 2 costs $8,000 and takes 4 months. What is the total duration of the project?
A) $13,000
B) 4 months
C) 7 months
D) $5,000
"""
    multi_answer_groundtruth = "C"
    multi_answer_model_response = """
Let's analyze the problem.
The cost of Phase 1 is $5,000 and the duration is 3 months.
The cost of Phase 2 is $8,000 and the duration is 4 months.
The total cost would be $5,000 + $8,000 = $13,000.
The question asks for the total duration, which is 3 months + 4 months = 7 months.
Therefore, the final answer is 7 months. This matches option C.
"""
    score3 = eval(multi_answer_question, multi_answer_groundtruth, multi_answer_model_response)
    print(f"Question: {multi_answer_question.strip()}")
    print(f"Model Answer: '{multi_answer_model_response.strip()}'")
    print(f"Ground Truth: '{multi_answer_groundtruth}'")
    print(f"==> Score: {score3}\n") # 1.0


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
