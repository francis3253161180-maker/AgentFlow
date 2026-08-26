import os
import re
import unicodedata
from pydantic import BaseModel
from agentflow.engine.openai import ChatOpenAI

try:
    import sympy
except ImportError:  # pragma: no cover - AgentFlow includes sympy in production
    sympy = None


class AnswerVerification(BaseModel):
    analysis: str
    true_false: bool

llm_scorer_engine = None
if os.getenv("AGENTFLOW_USE_LLM_SCORER") == "1":
    try:
        llm_scorer_engine = ChatOpenAI(
            model_string="gpt-4o",
            is_multimodal=False,
            enable_cache=True
        )
        print(f"\nLLM Scorer engine '{llm_scorer_engine.model_string}' initialized successfully.\n")
    except Exception as e:
        print(f"Failed to initialize LLM Scorer engine: {e}")

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
    if "=" in text:
        expressions.append(text.rsplit("=", 1)[-1].strip())
    expressions.extend(re.findall(r"\\(?:d?frac)\s*\{[^{}]+\}\s*\{[^{}]+\}", text))
    expressions.extend(re.findall(r"\\sqrt\s*\{[^{}]+\}|√\s*[0-9]+", text))
    expressions.extend(re.findall(r"(?<![A-Za-z])[-+]?\d+\s*/\s*\d+", text))
    return list(dict.fromkeys(expressions))


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
    text = text.strip(" $\\")
    if not text or len(text) > 160 or re.search(r"[^A-Za-z0-9_+\-*/()., ]", text):
        return None
    try:
        return sympy.sympify(text, locals={"sqrt": sympy.sqrt})
    except (TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return None


def _math_match(groundtruth: str, candidates: list[str]) -> bool:
    truth_compact = _compact_candidates(groundtruth)
    for candidate in candidates:
        if truth_compact & _compact_candidates(candidate):
            return True
    truth_exprs = [_to_sympy(expr) for expr in _math_expressions(groundtruth)]
    truth_exprs = [expr for expr in truth_exprs if expr is not None]
    if not truth_exprs:
        return False
    for candidate in candidates:
        for answer_expr in _math_expressions(candidate):
            parsed_answer = _to_sympy(answer_expr)
            if parsed_answer is None:
                continue
            for parsed_truth in truth_exprs:
                try:
                    if bool(sympy.simplify(parsed_answer - parsed_truth) == 0):
                        return True
                except (TypeError, ValueError):
                    continue
    return False


def _looks_mathematical(value: str) -> bool:
    text = str(value)
    return bool(
        re.search(r"\\(?:d?frac|sqrt)|√|\^|=", text)
        or re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*", text)
        or re.search(r"[A-Za-z]\w*\s*[+*/^=]\s*[-+A-Za-z0-9]", text)
    )


def deterministic_fallback_score(groundtruth: str, answer_extracted: str) -> bool:
    """Compare an answer deterministically without an external judge."""
    candidates = _answer_candidates(answer_extracted)
    truth = str(groundtruth).strip()
    truth_lower = truth.lower()

    if truth_lower in {"yes", "no"}:
        expected = truth_lower == "yes"
        return any(_yes_no(candidate) is expected for candidate in candidates)

    date_result = _date_match(truth, candidates)
    if date_result is not None:
        return date_result

    numeric_result = _numeric_match(truth, candidates)
    if numeric_result is not None:
        return numeric_result

    if _looks_mathematical(truth) and _math_match(truth, candidates):
        return True

    return any(_phrase_match(truth, candidate) for candidate in candidates)


def compute_score(question: str, groundtruth: str, answer_extracted: str,) -> bool:
    """
    Uses gpt-4o to determine if the extracted answer matches the groundtruth.
    
    Args:
        question: The full question text, including options.
        answer_extracted: The answer provided by the model being evaluated.
        groundtruth: The correct answer label (e.g., "A").

    Returns:
        A boolean indicating whether the answer is correct.
    """
    if llm_scorer_engine is None:
        return deterministic_fallback_score(groundtruth, answer_extracted)

    query_prompt = f"""
You are a precise evaluator. Determine if the Model Response is equivalent to the Ground Truth.

**Instructions:**
1.  **Extract:** Isolate the final answer from the Model Response, ignoring reasoning. Look for `\boxed{{...}}` or concluding statements.
2.  **Normalize & Compare:** The extracted answer and Ground Truth must be equivalent after normalization:
    - **Math:** Mathematically identical (e.g., `\\frac{{1}}{{2}}` == `0.5`).
    - **Numbers/Text:** Ignore formatting, case, and currency/units (e.g., `1,000` == `1000`).
    - **MCQ:** Match option content (e.g., "Paris") or number (e.g., `3rd` option) to the correct letter.
3.  **Verdict:** "True" only for semantically or mathematically equivalent answers.

**Inputs:**
Question: {question}
Model Response: {answer_extracted}
Ground Truth: {groundtruth}

**Format:**
<analysis>: Brief analysis of the comparison.
<true_false>: "True" or "False".
"""

    try:
        verification_result = llm_scorer_engine(query_prompt, response_format=AnswerVerification)
        if isinstance(verification_result, AnswerVerification):
            return verification_result.true_false
        if isinstance(verification_result, dict) and "true_false" in verification_result:
            return bool(verification_result["true_false"])
    except Exception as exc:
        print(f"LLM scorer unavailable; using local smoke-test comparison: {exc}")

    return deterministic_fallback_score(groundtruth, answer_extracted)


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
