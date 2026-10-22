from src.methods.edit import Edit, LLMScorer, RandomScorer, RepScorer
from src.methods.llm_select import LLMSelect
from src.methods.random import RandomSelection


def build_method(cfg, state, encoder=None):
    method = cfg["method"]
    if method == "edit_rep":
        return Edit(cfg, RepScorer(cfg, state, encoder), state)
    if method == "edit_llm":
        return Edit(cfg, LLMScorer(cfg, state), state)
    if method == "edit_rand":
        return Edit(cfg, RandomScorer(cfg), state)
    if method == "llm_select":
        return LLMSelect(cfg)
    if method == "random":
        return RandomSelection(cfg)
    raise ValueError(f"Unknown method: {method}")
