import numpy as np

TEMPLATES = {
    "math500": "Problem:\n\n{question}\n\nModel answer:\n\n{model_answer}",
    "popqa": "Question:\n\n{question}\n\nModel answer:\n\n{model_answer}",
    "mmlupro": "Multiple-choice question:\n\n{question}\n\nModel answer:\n\n{model_answer}",
    "gpqa": "Multiple-choice question:\n\n{question}\n\nModel answer:\n\n{model_answer}",
}


class Encoder:
    def __init__(self, model_name, dataset, max_length=512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.template = TEMPLATES[dataset]
        self.max_length = max_length
        self.qwen = "qwen3" in model_name.lower()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        kwargs = {}
        if self.qwen:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            kwargs = {"low_cpu_mem_usage": True, "dtype": "auto" if self.device.type == "cuda" else torch.float32}
            name = model_name.lower()
            self.batch_size = 2 if "4b" in name else 4 if "1.7b" in name else 8
        else:
            self.batch_size = 64
        self.model = AutoModel.from_pretrained(model_name, **kwargs).to(self.device).eval()
        self.cache = {}

    def encode_rows(self, rows):
        texts = [
            self.template.format(question=str(r.get("question", "")), model_answer=str(r.get("model_answer", "")))
            for r in rows
        ]
        if not self.qwen:
            return self._encode(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            for text, feature in zip(missing, self._encode(missing)):
                self.cache[text] = feature.astype(np.float32, copy=True)
        return np.stack([self.cache[t] for t in texts]).astype(np.float32)

    def _encode(self, texts):
        chunks = [self._encode_batch(texts[i : i + self.batch_size]) for i in range(0, len(texts), self.batch_size)]
        return np.concatenate(chunks).astype(np.float32)

    def _tokenize(self, batch):
        if not self.qwen:
            return self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        eos = self.tokenizer.eos_token_id
        encoded = self.tokenizer(batch, padding=False, truncation=True, max_length=self.max_length - 1, add_special_tokens=True)
        features = []
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            ids, mask = list(ids), list(mask)
            if not ids or ids[-1] != eos:
                ids.append(int(eos))
                mask.append(1)
            features.append({"input_ids": ids, "attention_mask": mask})
        return self.tokenizer.pad(features, padding=True, max_length=self.max_length, return_tensors="pt")

    def _encode_batch(self, batch):
        torch = self.torch
        inputs = {k: v.to(self.device) for k, v in self._tokenize(batch).items()}
        with torch.inference_mode():
            hidden = self.model(**inputs, return_dict=True).last_hidden_state
            mask = inputs["attention_mask"]
            if not self.qwen:
                weights = mask.unsqueeze(-1).to(hidden.dtype)
                feats = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
            elif bool((mask[:, -1] == 1).all().item()):
                feats = hidden[:, -1]
            else:
                lengths = mask.sum(dim=1).clamp(min=1) - 1
                feats = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
            feats = feats.float().cpu().numpy().astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        return feats.astype(np.float32)
