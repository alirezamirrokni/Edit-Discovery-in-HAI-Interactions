from __future__ import annotations

import gc
from typing import Any, Dict, List

import numpy as np


class Qwen3ScoreModel:


    MODEL_ALIASES = {
        "0.6": "Qwen/Qwen3-0.6B",
        "0.6b": "Qwen/Qwen3-0.6B",
        "qwen3-0.6": "Qwen/Qwen3-0.6B",
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "qwen/qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "1.7": "Qwen/Qwen3-1.7B",
        "1.7b": "Qwen/Qwen3-1.7B",
        "qwen3-1.7": "Qwen/Qwen3-1.7B",
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen/qwen3-1.7b": "Qwen/Qwen3-1.7B",
        "4": "Qwen/Qwen3-4B",
        "4b": "Qwen/Qwen3-4B",
        "qwen3-4": "Qwen/Qwen3-4B",
        "qwen3-4b": "Qwen/Qwen3-4B",
        "qwen/qwen3-4b": "Qwen/Qwen3-4B",
    }

    def __init__(self, cfg: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "Qwen3ScoreModel requires torch and transformers>=4.51.0. "
                "Install the project requirements with `pip install -r requirements.txt`."
            ) from exc

        self.torch = torch
        self.cfg = dict(cfg)

        requested_name = str(self.cfg.get("model_name", "Qwen/Qwen3-0.6B"))
        self.model_name = self._resolve_model_name(requested_name)
        self.max_length = int(self.cfg.get("max_length", 512))
        self.encode_batch_size = int(
            self.cfg.get("encode_batch_size", self._default_batch_size(self.model_name))
        )
        self.pooling = str(self.cfg.get("pooling", "last")).lower()
        self.normalize_features = bool(self.cfg.get("normalize_features", True))
        self.append_eos = bool(self.cfg.get("append_eos", True))
        self.cache_embeddings = bool(self.cfg.get("cache_embeddings", True))
        self.cache_dir = self.cfg.get("cache_dir", None)
        self.device_cfg = self.cfg.get("device", "auto")
        self.torch_dtype_cfg = self.cfg.get("torch_dtype", "auto")
        self.trust_remote_code = bool(self.cfg.get("trust_remote_code", False))
        self.attn_implementation = self.cfg.get("attn_implementation", None)

        dataset_name = str(self.cfg.get("dataset_name", "math500")).lower().replace("-", "_")
        if dataset_name in {"popqa", "popqa500"}:
            default_prompt_template = "Question:\n\n{question}\n\nModel answer:\n\n{model_answer}"
        elif dataset_name in {
            "mmlupro",
            "mmlu_pro",
            "mmlupro500",
            "mmlu_pro500",
            "gpqa",
            "gpqa_diamond",
            "gpqa_main",
            "gpqa_extended",
        }:
            default_prompt_template = (
                "Multiple-choice question:\n\n{question}\n\nModel answer:\n\n{model_answer}"
            )
        else:
            default_prompt_template = "Problem:\n\n{question}\n\nModel answer:\n\n{model_answer}"
        self.prompt_template = self.cfg.get("prompt_template", default_prompt_template)

        if self.pooling not in {"last", "mean"}:
            raise ValueError(
                f"Unknown pooling={self.pooling!r}. Qwen3ScoreModel supports 'last' or 'mean'."
            )
        if self.max_length <= 0:
            raise ValueError("score_model.max_length must be positive.")
        if self.encode_batch_size <= 0:
            raise ValueError("score_model.encode_batch_size must be positive.")

        self.device = self._resolve_device(self.device_cfg)
        dtype = self._resolve_torch_dtype(self.torch_dtype_cfg)

        print(f"[score_model:qwen3] loading frozen encoder: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True,
            trust_remote_code=self.trust_remote_code,
        )


        self.tokenizer.padding_side = str(self.cfg.get("padding_side", "left"))
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer for {self.model_name} has neither pad_token_id nor eos_token_id."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.attn_implementation not in {None, "", "none"}:
            model_kwargs["attn_implementation"] = str(self.attn_implementation)

        self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.hidden_size = self._infer_hidden_size()
        self._embedding_cache: Dict[str, np.ndarray] = {}

        print(
            "[score_model:qwen3] loaded "
            f"model={self.model_name} "
            f"hidden_size={self.hidden_size} "
            f"pooling={self.pooling} "
            f"max_length={self.max_length} "
            f"batch_size={self.encode_batch_size} "
            f"device={self.device} "
            f"dtype={next(self.model.parameters()).dtype} "
            f"normalize={self.normalize_features} "
            f"cache_embeddings={self.cache_embeddings}"
        )

    @classmethod
    def _resolve_model_name(cls, value: str) -> str:
        key = str(value).strip().lower()
        return cls.MODEL_ALIASES.get(key, str(value).strip())

    @staticmethod
    def _default_batch_size(model_name: str) -> int:
        name = model_name.lower()
        if "4b" in name:
            return 2
        if "1.7b" in name:
            return 4
        return 8

    def _resolve_device(self, value: Any):
        torch = self.torch
        if value is None or str(value).lower() == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(str(value))

    def _resolve_torch_dtype(self, value: Any):
        torch = self.torch

        if value is None:
            return None
        if not isinstance(value, str):
            return value

        value = value.lower()
        if value == "auto":


            return "auto" if self.device.type == "cuda" else torch.float32

        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        if value not in mapping:
            raise ValueError(
                f"Unknown torch dtype {value!r}. Use auto, float16/fp16, "
                "bfloat16/bf16, or float32/fp32."
            )
        return mapping[value]

    def _infer_hidden_size(self) -> int:
        candidates = [
            getattr(self.model.config, "hidden_size", None),
            getattr(self.model.config, "d_model", None),
            getattr(self.model.config, "n_embd", None),
        ]
        for value in candidates:
            if value is not None:
                return int(value)
        raise ValueError(f"Could not infer hidden size for {self.model_name}.")

    def format_text(self, question: str, model_answer: str) -> str:
        return self.prompt_template.format(
            question=str(question),
            model_answer=str(model_answer),
        )

    def encode_rows(self, rows: List[dict]) -> np.ndarray:
        texts = [
            self.format_text(
                question=row.get("question", ""),
                model_answer=row.get("model_answer", ""),
            )
            for row in rows
        ]
        return self.encode_texts(texts)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.hidden_size), dtype=np.float32)

        if not self.cache_embeddings:
            return self._encode_uncached(texts)


        missing = []
        seen_missing = set()
        for text in texts:
            if text not in self._embedding_cache and text not in seen_missing:
                missing.append(text)
                seen_missing.add(text)

        if missing:
            new_features = self._encode_uncached(missing)
            for text, feature in zip(missing, new_features):
                self._embedding_cache[text] = feature.astype(np.float32, copy=True)

        return np.stack([self._embedding_cache[text] for text in texts], axis=0).astype(
            np.float32
        )

    def _encode_uncached(self, texts: List[str]) -> np.ndarray:
        features = []
        batch_size = max(1, int(self.encode_batch_size))
        start = 0

        while start < len(texts):
            batch = texts[start : start + batch_size]
            try:
                batch_features = self._encode_batch(batch)
                features.append(batch_features)
                start += len(batch)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message and batch_size > 1:
                    new_batch_size = max(1, batch_size // 2)
                    print(
                        f"[score_model:qwen3] CUDA OOM with encode_batch_size={batch_size}; "
                        f"retrying with batch_size={new_batch_size}"
                    )
                    self._clear_memory()
                    batch_size = new_batch_size
                    self.encode_batch_size = new_batch_size
                    continue
                raise

        return np.concatenate(features, axis=0).astype(np.float32)

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        torch = self.torch

        inputs = self._tokenize_batch(batch)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs, return_dict=True)
            hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            if self.pooling == "last":
                features = self._last_token_pool(hidden, attention_mask)
            elif self.pooling == "mean":
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                denominator = mask.sum(dim=1).clamp(min=1e-9)
                features = (hidden * mask).sum(dim=1) / denominator
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            features = features.detach().float().cpu().numpy().astype(np.float32)

        if self.normalize_features:
            denominator = np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
            features = features / denominator

        return features.astype(np.float32)

    def _tokenize_batch(self, batch: List[str]):


        reserve_eos = self.append_eos and self.tokenizer.eos_token_id is not None
        content_max_length = self.max_length - 1 if reserve_eos else self.max_length
        content_max_length = max(1, content_max_length)

        encoded = self.tokenizer(
            batch,
            padding=False,
            truncation=True,
            max_length=content_max_length,
            add_special_tokens=True,
        )

        features = []
        eos_id = self.tokenizer.eos_token_id
        for input_ids, attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            ids = list(input_ids)
            mask = list(attention_mask)
            if reserve_eos and (not ids or ids[-1] != eos_id):
                ids.append(int(eos_id))
                mask.append(1)
            features.append({"input_ids": ids, "attention_mask": mask})

        return self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _last_token_pool(self, last_hidden_state, attention_mask):

        if bool((attention_mask[:, -1] == 1).all().item()):
            return last_hidden_state[:, -1]


        sequence_lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = self.torch.arange(
            last_hidden_state.shape[0], device=last_hidden_state.device
        )
        return last_hidden_state[batch_indices, sequence_lengths]

    def _clear_memory(self) -> None:
        gc.collect()
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
