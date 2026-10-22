import os

from openai import OpenAI


class LLM:
    def __init__(self, cfg, system_prompt):
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=120,
            max_retries=8,
        )

    def generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.cfg["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.cfg["max_tokens"],
            extra_body={"reasoning": self.cfg["reasoning"]} if "reasoning" in self.cfg else None,
        )
        return (response.choices[0].message.content or "").strip()
