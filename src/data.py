import ast
import json
import math
import re
import string
from collections import defaultdict

import numpy as np
from datasets import load_dataset

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_choice(text, num_options):
    s = str(text or "").strip()
    if not s:
        return None
    valid = LETTERS[:num_options]
    marker = re.findall(r"####\s*\(?\s*([A-Z])\s*\)?", s, flags=re.IGNORECASE)
    if marker:
        cand = marker[-1].upper()
        return cand if cand in valid else None
    for pattern in (
        r"final\s+answer\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?",
        r"answer\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?",
        r"option\s*[:\-]?\s*\(?\s*([A-Z])\s*\)?",
        r"^\s*\(?\s*([A-Z])\s*\)?\s*$",
    ):
        matches = re.findall(pattern, s, flags=re.IGNORECASE | re.MULTILINE)
        if matches and matches[-1].upper() in valid:
            return matches[-1].upper()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        m = re.search(r"\b([A-Z])\b", lines[-1], flags=re.IGNORECASE)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    return None


def choice_failure(pred, gold):
    if pred is None or gold is None:
        return 1
    return int(str(pred).strip().upper() != str(gold).strip().upper())


def format_choices(header, question, options):
    lines = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options))
    return f"{header}Question:\n{question}\n\nOptions:\n{lines}"


def _braced_argument(text, command):
    i = text.find(command) + len(command)
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1 : j]
    return None


def extract_boxed(text):
    if text is None:
        return None
    text = str(text)
    last = None
    for command in ("\\boxed", "\\fbox"):
        start = 0
        while (idx := text.find(command, start)) >= 0:
            candidate = _braced_argument(text[idx:], command)
            if candidate is not None:
                last = candidate
            start = idx + len(command)
    return last.strip() if last is not None else None


def final_math_answer(text):
    if text is None:
        return None
    text = str(text).strip()
    marker = re.findall(r"####\s*([^\n]+)", text)
    if marker:
        return marker[-1].strip().rstrip(".")
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed.strip().rstrip(".")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip().rstrip(".") if lines else None


def _strip_outer_braces(text):
    text = text.strip()
    while text.startswith("{") and text.endswith("}"):
        depth = 0
        for i, ch in enumerate(text):
            depth += {"{": 1, "}": -1}.get(ch, 0)
            if depth == 0 and i != len(text) - 1:
                return text
        text = text[1:-1].strip()
    return text


def normalize_math(ans):
    if ans is None:
        return None
    s = str(ans).strip().strip("$ ")
    for old, new in (("\\left", ""), ("\\right", ""), ("\\,", ""), ("\\!", ""), ("\\;", ""), ("\\:", ""), ("\u2212", "-")):
        s = s.replace(old, new)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s).replace(" ", "").rstrip(".")
    boxed = extract_boxed(s)
    return _strip_outer_braces(boxed if boxed is not None else s)


def _math_verify_parse(text):
    from math_verify import parse

    text = str(text).strip()
    last = []
    for variant in (text, f"${text}$", f"\\boxed{{{text}}}", f"$\\boxed{{{text}}}$") if text else ():
        try:
            last = parse(variant)
            if last:
                return last
        except Exception:
            continue
    return last


def math_equal(a, b):
    if a is None or b is None:
        return False
    from math_verify import verify

    try:
        gold, answer = _math_verify_parse(b), _math_verify_parse(a)
        if gold and answer:
            return bool(verify(gold, answer))
    except Exception:
        pass
    na, nb = normalize_math(a), normalize_math(b)
    if na == nb:
        return True
    try:
        return abs(float(na.replace(",", "")) - float(nb.replace(",", ""))) <= 1e-6
    except Exception:
        return False


class Math500:
    system_prompt = """You are a careful mathematical problem solver.

Hard requirements:
- Solve the problem accurately.
- End with exactly one final line in this format:
#### <answer>

The final line must contain only the marker #### followed by the final answer."""

    def __init__(self, cfg):
        self.cfg = cfg

    def load_records(self):
        records = []
        for idx, row in enumerate(load_dataset("HuggingFaceH4/MATH-500", split=self.cfg["split"])):
            solution = row["solution"]
            answer = row["answer"] or final_math_answer(solution)
            records.append(
                {
                    "example_id": idx,
                    "question": row["problem"],
                    "gold_answer": solution if solution else str(answer),
                    "gold_final": final_math_answer(answer) or final_math_answer(solution),
                }
            )
        return records

    @staticmethod
    def build_prompt(question):
        return (
            "Solve the following math problem.\n"
            "End with exactly one final line in this format: #### <answer>\n\n"
            f"Problem:\n{question}"
        )

    @staticmethod
    def parse_prediction(answer):
        return final_math_answer(answer)

    @staticmethod
    def failure_label(pred, gold):
        return int(not math_equal(pred, gold))


class MMLUPro:
    system_prompt = """You are a careful multiple-choice reasoning system.

Hard requirements:
- Solve the question and choose exactly one option.
- The final answer must be only the option letter.
- End with exactly one final line in this format:
#### <letter>

The final line must contain only the marker #### followed by a single letter such as A, B, C, D, E, F, G, H, I, or J."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _subset(self, categories):
        size, n = int(self.cfg["subset_size"]), len(categories)
        groups = defaultdict(list)
        for idx, category in enumerate(categories):
            groups[str(category)].append(idx)
        raw = {c: size * len(ids) / float(n) for c, ids in groups.items()}
        alloc = {c: int(math.floor(v)) for c, v in raw.items()}
        remainders = sorted(raw, key=lambda c: (raw[c] - alloc[c], len(groups[c]), c), reverse=True)
        for c in remainders[: size - sum(alloc.values())]:
            alloc[c] += 1
        rng = np.random.default_rng(int(self.cfg["subset_seed"]))
        selected = []
        for c in sorted(groups):
            if alloc[c] > 0:
                selected.extend(int(x) for x in rng.choice(np.asarray(groups[c], dtype=int), size=alloc[c], replace=False))
        return sorted(selected)

    def load_records(self):
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=self.cfg["split"])
        records = []
        for idx in self._subset(ds["category"]):
            row = ds[idx]
            options = [str(x) for x in row["options"]]
            letter = row["answer"]
            records.append(
                {
                    "example_id": idx,
                    "question": format_choices(f"Category: {row['category']}\n\n", row["question"], options),
                    "gold_answer": f"{letter}. {options[LETTERS.index(letter)]}",
                    "gold_final": letter,
                    "category": str(row["category"]),
                }
            )
        return records

    @staticmethod
    def build_prompt(question):
        return (
            "Solve the following MMLU-Pro multiple-choice question.\n"
            "Choose the single best option. Think carefully if needed, but do not include reasoning in the output.\n"
            "End with exactly one final line in this format: #### <letter>\n\n"
            f"{question}"
        )

    @staticmethod
    def parse_prediction(answer):
        return extract_choice(answer, 10)

    failure_label = staticmethod(choice_failure)


class GPQA:
    system_prompt = """You are a careful graduate-level science reasoning system.

Hard requirements:
- Solve the multiple-choice question and choose exactly one option.
- The final answer must be only the option letter.
- End with exactly one final line in this format:
#### <letter>

The final line must contain only the marker #### followed by a single letter: A, B, C, or D."""

    def __init__(self, cfg):
        self.cfg = cfg

    def load_records(self):
        ds = load_dataset("Idavidrein/gpqa", self.cfg["hf_config"], split=self.cfg["split"])
        seed, size = int(self.cfg["subset_seed"]), int(self.cfg["subset_size"])
        indices = range(len(ds)) if size == len(ds) else np.random.default_rng(seed).permutation(len(ds))[:size].tolist()
        records = []
        for idx in indices:
            row = ds[int(idx)]
            choices = [(str(row["Correct Answer"]).strip(), True)]
            choices += [(str(row[f"Incorrect Answer {k}"]).strip(), False) for k in (1, 2, 3)]
            choices = [choices[i] for i in np.random.default_rng(seed + 1000003 * int(idx)).permutation(4).tolist()]
            options = [text for text, _ in choices]
            gold = next(i for i, (_, correct) in enumerate(choices) if correct)
            records.append(
                {
                    "example_id": int(idx),
                    "question": format_choices(f"Subject: {str(row['Subdomain']).strip()}\n\n", str(row["Question"]).strip(), options),
                    "gold_answer": f"{LETTERS[gold]}. {options[gold]}",
                    "gold_final": LETTERS[gold],
                }
            )
        return records

    @staticmethod
    def build_prompt(question):
        return (
            "Solve the following GPQA graduate-level science multiple-choice question.\n"
            "Choose the single best option. Think carefully if needed.\n"
            "End with exactly one final line in this format: #### <letter>\n\n"
            f"{question}"
        )

    @staticmethod
    def parse_prediction(answer):
        return extract_choice(answer, 4)

    failure_label = staticmethod(choice_failure)


def normalize_popqa(text):
    if text is None:
        return ""
    s = "".join(ch for ch in str(text).lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def _parse_list(value):
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, (list, tuple)):
                return [str(x) for x in obj if x is not None]
            if isinstance(obj, str):
                return [obj]
        except Exception:
            pass
    for sep in ("|||", "||", "\t"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def _clean_short_answer(text):
    ans = re.sub(r"^[-*\s]+", "", str(text).strip()).strip()
    ans = re.sub(r"^(the\s+answer\s+is|answer\s+is|it\s+is|it's)\s+", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.strip(" \t\r\n\"'`“”‘’").rstrip(".").strip()
    return ans or None


def final_short_answer(text):
    s = str(text or "").strip()
    if not s:
        return None
    for pattern, flags in ((r"####\s*([^\n]+)", 0), (r"final\s+answer\s*[:\-]\s*([^\n]+)", re.IGNORECASE), (r"answer\s*[:\-]\s*([^\n]+)", re.IGNORECASE)):
        matches = re.findall(pattern, s, flags=flags)
        if matches:
            return _clean_short_answer(matches[-1])
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return _clean_short_answer(lines[-1]) if lines else None


def _unique(values):
    return list(dict.fromkeys(v for v in values if v))


class PopQA:
    system_prompt = """You are a careful open-domain factual question-answering system.

Hard requirements:
- Answer the question accurately.
- Provide only the final short answer, not a sentence or explanation.
- The answer should be an entity, title, place, date, number, or short phrase.
- End with exactly one final line in this format:
#### <answer>

The final line must contain only the marker #### followed by the answer."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _subset(self, popularity):
        size, n = int(self.cfg["subset_size"]), len(popularity)
        rng = np.random.default_rng(int(self.cfg["subset_seed"]))
        if self.cfg["subset_strategy"] == "random":
            return rng.permutation(n)[:size].tolist()
        scored = sorted((p, i) for i, p in enumerate(popularity) if math.isfinite(p))
        thirds = np.array_split(np.asarray([i for _, i in scored], dtype=int), 3)
        alloc = [size // 3 + (1 if k < size % 3 else 0) for k in range(3)]
        selected = []
        for ids, k in zip(thirds, alloc):
            selected.extend(int(x) for x in rng.choice(ids, size=k, replace=False))
        return sorted(selected)

    def load_records(self):
        ds = load_dataset("akariasai/PopQA", split=self.cfg["split"])
        popularity = [float(p) if p is not None else float("nan") for p in ds["s_pop"]]
        records = []
        for idx in self._subset(popularity):
            row = ds[int(idx)]
            raw = _unique(a.strip() for a in _parse_list(row["possible_answers"]) + _parse_list(row["obj"]))
            records.append(
                {
                    "example_id": int(idx),
                    "question": str(row["question"]),
                    "gold_answer": raw[0] if raw else "",
                    "gold_final": json.dumps(_unique(normalize_popqa(a) for a in raw), ensure_ascii=False),
                    "popularity": popularity[idx],
                }
            )
        return records

    @staticmethod
    def build_prompt(question):
        return (
            "Answer the following open-domain factual question from PopQA.\n"
            "Give only the final short answer; do not include explanation, reasoning, or extra text.\n"
            "The answer should be an entity, title, place, date, number, or short phrase.\n"
            "End with exactly one final line in this format: #### <answer>\n\n"
            f"Question:\n{question}"
        )

    @staticmethod
    def parse_prediction(answer):
        return final_short_answer(answer)

    @staticmethod
    def failure_label(pred, gold):
        pred = normalize_popqa(pred)
        aliases = {normalize_popqa(a) for a in json.loads(gold)}
        return int(not pred or pred not in aliases)


DATASETS = {"math500": Math500, "mmlupro": MMLUPro, "popqa": PopQA, "gpqa": GPQA}


def build_dataset(cfg):
    return DATASETS[cfg["name"]](cfg)
