from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

import numpy as np


class MiniLMScoreModel:


    def __init__(self, cfg: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "Install torch and transformers first. In Colab, run: "
                "`pip install -U torch transformers`."
            ) from exc

        self.torch = torch
        self.cfg = dict(cfg)

        self.model_name = self.cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.max_length = int(self.cfg.get("max_length", 512))
        self.encode_batch_size = int(self.cfg.get("encode_batch_size", 64))
        self.pooling = str(self.cfg.get("pooling", "mean")).lower()
        self.normalize_features = bool(self.cfg.get("normalize_features", True))
        self.cache_dir = self.cfg.get("cache_dir", None)
        self.device_cfg = self.cfg.get("device", "auto")
        self.torch_dtype = self.cfg.get("torch_dtype", "auto")

        dataset_name = str(self.cfg.get("dataset_name", "math500")).lower().replace("-", "_")
        if dataset_name in {"popqa", "popqa500"}:
            default_prompt_template = "Question:\n\n{question}\n\nModel answer:\n\n{model_answer}"
        elif dataset_name in {"mmlupro", "mmlu_pro", "mmlupro500", "mmlu_pro500", "gpqa", "gpqa_diamond", "gpqa_main", "gpqa_extended"}:
            default_prompt_template = "Multiple-choice question:\n\n{question}\n\nModel answer:\n\n{model_answer}"
        else:
            default_prompt_template = "Problem:\n\n{question}\n\nModel answer:\n\n{model_answer}"
        self.prompt_template = self.cfg.get("prompt_template", default_prompt_template)

        if self.pooling not in {"mean", "cls", "last"}:
            raise ValueError(f"Unknown pooling='{self.pooling}'. Use 'mean', 'cls', or 'last'.")

        self.device = self._resolve_device(self.device_cfg)
        dtype = self._resolve_torch_dtype(self.torch_dtype)

        print(f"[score_model:minilm] loading encoder: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True,
        )

        model_kwargs: Dict[str, Any] = {"cache_dir": self.cache_dir}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        try:
            self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        except Exception:
            self.hidden_size = 384

        print(
            "[score_model:minilm] loaded "
            f"model={self.model_name} "
            f"pooling={self.pooling} "
            f"max_length={self.max_length} "
            f"batch_size={self.encode_batch_size} "
            f"device={self.device} "
            f"normalize={self.normalize_features}"
        )

    def _resolve_device(self, value: Any):
        torch = self.torch
        if value is None or str(value).lower() == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(str(value))

    def _resolve_torch_dtype(self, value: Any):
        torch = self.torch

        if value is None or value == "auto":
            return None
        if not isinstance(value, str):
            return value

        value = value.lower()
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
                f"Unknown torch dtype '{value}'. "
                "Use one of: auto, float16, fp16, bfloat16, bf16, float32, fp32."
            )
        return mapping[value]

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
        if len(texts) == 0:
            dim = self.hidden_size if self.hidden_size is not None else 384
            return np.empty((0, dim), dtype=np.float32)

        features = []
        batch_size = max(1, int(self.encode_batch_size))
        start = 0

        while start < len(texts):
            batch = texts[start : start + batch_size]
            try:
                batch_features = self._encode_batch(batch)
                features.append(batch_features)
                start += batch_size
            except RuntimeError as exc:
                message = str(exc).lower()
                if "out of memory" in message and batch_size > 1:
                    print(
                        f"[score_model:minilm] CUDA OOM with encode_batch_size={batch_size}; "
                        f"retrying with batch_size={max(1, batch_size // 2)}"
                    )
                    self._clear_memory()
                    batch_size = max(1, batch_size // 2)
                    self.encode_batch_size = batch_size
                    continue
                raise

        return np.concatenate(features, axis=0).astype(np.float32)

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        torch = self.torch

        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs, return_dict=True)
            hidden = out.last_hidden_state
            attention_mask = inputs["attention_mask"]

            if self.pooling == "mean":
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                denom = mask.sum(dim=1).clamp(min=1e-9)
                feats = (hidden * mask).sum(dim=1) / denom

            elif self.pooling == "cls":
                feats = hidden[:, 0]

            elif self.pooling == "last":
                lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
                batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
                feats = hidden[batch_idx, lengths]

            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            feats = feats.detach().float().cpu().numpy().astype(np.float32)

        if self.normalize_features:
            denom = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
            feats = feats / denom

        return feats.astype(np.float32)

    def _clear_memory(self) -> None:
        gc.collect()
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
