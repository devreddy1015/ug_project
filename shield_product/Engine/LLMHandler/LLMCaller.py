import json
import os
from typing import Optional
from urllib import request


class LLMCaller:
    def __init__(self) -> None:
        self.api_base = os.environ.get("LLM_API_BASE")
        self.api_key = os.environ.get("LLM_API_KEY")
        self.model = os.environ.get("LLM_MODEL", "")

    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key and self.model)

    def call(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.is_configured():
            return None

        url = self.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
        except Exception:
            return None

        try:
            response = json.loads(body)
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError, TypeError):
            return None
