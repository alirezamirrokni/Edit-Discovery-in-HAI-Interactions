import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


DEFAULT_SYSTEM_PROMPT = """You are a careful mathematical problem solver.

Hard requirements:
- Solve the problem accurately.
- End with exactly one final line in this format:
#### <answer>

The final line must contain only the marker #### followed by the final answer."""


@dataclass
class GroqConfig:
    provider: str
    model_name: str = "llama-3.3-70b-versatile"
    display_name: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 128
    request_timeout: int = 120
    retry_attempts: int = 8
    retry_sleep: float = 2.0
    min_seconds_between_calls: float = 2.5
    system_prompt: Optional[str] = None
    prompt_version: str = "math500_answer_format_v1"
    reasoning_effort: Optional[str] = None


class GroqLLM:


    def __init__(self, cfg: Dict[str, Any]):
        try:
            from groq import Groq
        except Exception as exc:
            raise ImportError("Install the Groq SDK first. Add `groq>=0.13.0` to requirements.txt.") from exc

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. In Colab, set it with "
                "`os.environ['GROQ_API_KEY'] = getpass(...)`."
            )

        self.cfg = GroqConfig(**cfg)
        self.client = Groq(api_key=api_key, timeout=self.cfg.request_timeout)
        self._last_call_time = 0.0
        self.system_prompt = (
            self.cfg.system_prompt
            if self.cfg.system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT
        )

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call_time
        wait = self.cfg.min_seconds_between_calls - elapsed
        if wait > 0:
            time.sleep(wait)

    @staticmethod
    def _retry_after_seconds(exc: Exception, default: float) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after is not None:
                try:
                    return float(retry_after) + 1.0
                except Exception:
                    pass

        text = str(exc)
        patterns = [
            r"try again in\s*([0-9.]+)\s*s",
            r"retry in\s*([0-9.]+)\s*s",
            r"retry after\s*([0-9.]+)\s*s",
            r"Retry-After[:= ]+([0-9.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1)) + 1.0
                except Exception:
                    pass
        return default

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None

        for attempt in range(1, self.cfg.retry_attempts + 1):
            try:
                self._throttle()
                request_kwargs: Dict[str, Any] = {
                    "model": self.cfg.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": self.cfg.temperature,
                    "max_completion_tokens": self.cfg.max_output_tokens,
                }
                if self.cfg.reasoning_effort is not None:
                    request_kwargs["reasoning_effort"] = self.cfg.reasoning_effort
                if "gpt-oss" in self.cfg.model_name.lower():
                    request_kwargs["include_reasoning"] = False
                response = self.client.chat.completions.create(**request_kwargs)
                self._last_call_time = time.time()
                text = response.choices[0].message.content
                return str(text if text is not None else "").strip()

            except Exception as exc:
                last_err = exc
                base_wait = self.cfg.retry_sleep * attempt
                if "429" in str(exc) or "rate" in str(exc).lower():
                    wait = self._retry_after_seconds(exc, default=max(30.0, base_wait))
                else:
                    wait = base_wait
                print(
                    f"[main_llm:groq] attempt {attempt} failed: {exc}. "
                    f"retrying in {wait:.1f}s"
                )
                time.sleep(wait)

        raise RuntimeError(f"Groq call failed after retries: {last_err}")
