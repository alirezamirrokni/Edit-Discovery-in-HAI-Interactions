import re
from typing import Any, Dict, List, Optional

from datasets import load_dataset

MATH500_MAIN_SYSTEM_PROMPT = """You are a careful mathematical problem solver.

Hard requirements:
- Solve the problem accurately.
- End with exactly one final line in this format:
#### <answer>

The final line must contain only the marker #### followed by the final answer."""


_BOX_COMMANDS = ("\\boxed", "\\fbox")


def _strip_outer_braces(text: str) -> str:
    text = text.strip()
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        ok = True
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    ok = False
                    break
        if not ok:
            break
        text = text[1:-1].strip()
    return text


def _extract_braced_argument(text: str, command: str) -> Optional[str]:
    idx = text.find(command)
    if idx < 0:
        return None
    i = idx + len(command)
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None

    depth = 0
    start = i + 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
    return None


def extract_boxed_answer(text: str | None) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    last = None
    for command in _BOX_COMMANDS:
        start = 0
        while True:
            idx = text.find(command, start)
            if idx < 0:
                break
            candidate = _extract_braced_argument(text[idx:], command)
            if candidate is not None:
                last = candidate
            start = idx + len(command)
    return last.strip() if last is not None else None


def final_answer_from_text(text: str | None) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()

    marker_matches = re.findall(r"####\s*([^\n]+)", text)
    if marker_matches:
        return marker_matches[-1].strip().rstrip(".")

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip().rstrip(".")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1].strip().rstrip(".")
    return None


def normalize_math_answer(ans: str | None) -> Optional[str]:
    if ans is None:
        return None
    s = str(ans).strip()
    s = s.strip("$ ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\u2212", "-")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace(" ", "")
    s = s.rstrip(".")

    boxed = extract_boxed_answer(s)
    if boxed is not None:
        s = boxed

    return _strip_outer_braces(s)


def _fallback_equal(a: str | None, b: str | None, tol: float = 1e-6) -> bool:
    na = normalize_math_answer(a)
    nb = normalize_math_answer(b)
    if na is None or nb is None:
        return False
    if na == nb:
        return True
    try:
        return abs(float(na.replace(",", "")) - float(nb.replace(",", ""))) <= tol
    except Exception:
        return False


def _math_verify_parse(text: str):
    from math_verify import parse

    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []


    variants = [
        text,
        f"${text}$",
        f"\\boxed{{{text}}}",
        f"$\\boxed{{{text}}}$",
    ]
    last = []
    for variant in variants:
        try:
            parsed = parse(variant)
            last = parsed
            if parsed:
                return parsed
        except Exception:
            continue
    return last


def math_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False

    try:
        from math_verify import verify

        gold = _math_verify_parse(b)
        answer = _math_verify_parse(a)
        if gold and answer:
            return bool(verify(gold, answer))
    except ImportError as exc:
        raise ImportError(
            "math-verify is required for MATH-500 correctness checking. "
            "Install it with `pip install 'math-verify[antlr4_13_2]'`."
        ) from exc
    except Exception:
        pass


    return _fallback_equal(a, b)


class Math500Wrapper:


    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def load_records(self) -> List[Dict[str, Any]]:
        ds = load_dataset(
            self.cfg.get("hf_name", "HuggingFaceH4/MATH-500"),
            split=self.cfg.get("split", "test"),
        )
        records = []
        for idx, row in enumerate(ds):
            problem = row.get("problem") or row.get("question") or row.get("input")
            solution = row.get("solution", "")
            answer = row.get("answer") or final_answer_from_text(solution)
            gold_final = final_answer_from_text(answer) or final_answer_from_text(solution)
            records.append(
                {
                    "example_id": idx,
                    "question": problem,
                    "gold_answer": solution if solution else str(answer),
                    "gold_final": gold_final,
                }
            )
        return records

    @staticmethod
    def parse_prediction(model_answer: str) -> str | None:
        return final_answer_from_text(model_answer)

    @staticmethod
    def failure_label(pred_answer: str | None, gold_final: str | None) -> int:
        return int(not math_equal(pred_answer, gold_final))

    @staticmethod
    def build_prompt(question: str) -> str:
        return (
            "Solve the following math problem.\n"
            "End with exactly one final line in this format: #### <answer>\n\n"
            f"Problem:\n{question}"
        )

    @staticmethod
    def main_system_prompt() -> str:
        return MATH500_MAIN_SYSTEM_PROMPT
