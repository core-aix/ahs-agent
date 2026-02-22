from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class OllamaClient:
    mode: str
    model: str
    base_url: str
    api_key: str = ""
    max_output_tokens: int = 2048
    allow_insecure_tls: bool = False

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.http = httpx.Client(
            base_url=self.base_url, verify=not self.allow_insecure_tls, timeout=60.0
        )

    def close(self) -> None:
        self.http.close()

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        max_tokens = max(128, min(int(self.max_output_tokens or 2048), 8192))
        if self.mode == "local":
            resp = self.http.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            return str(payload.get("message", {}).get("content", "")).strip()

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self.http.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        if resp.status_code == 404:
            fallback = self.http.post(
                "/api/chat",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                },
            )
            fallback.raise_for_status()
            payload = fallback.json()
            return str(payload.get("message", {}).get("content", "")).strip()

        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        choices = payload.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()
