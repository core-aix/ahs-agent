from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import httpx


@dataclass
class OllamaClient:
    mode: str
    model: str
    base_url: str
    api_key: str = ""
    max_output_tokens: int = 65536
    max_context_tokens: int = 131072
    allow_insecure_tls: bool = False

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_estimated": 0,
            "output_tokens_estimated": 0,
        }
        self.http = httpx.Client(
            base_url=self.base_url, verify=not self.allow_insecure_tls, timeout=180.0
        )

    def close(self) -> None:
        self.http.close()

    def reset_usage(self) -> None:
        self._usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_estimated": 0,
            "output_tokens_estimated": 0,
        }

    def usage(self) -> dict[str, int | str]:
        out: dict[str, int | str] = dict(self._usage)
        if (
            int(out.get("input_tokens", 0) or 0) > 0
            and int(out.get("input_tokens_estimated", 0) or 0) == 0
        ):
            out["input_tokens_estimated"] = "not_needed"
        if (
            int(out.get("output_tokens", 0) or 0) > 0
            and int(out.get("output_tokens_estimated", 0) or 0) == 0
        ):
            out["output_tokens_estimated"] = "not_needed"
        return out

    def _estimate_tokens(self, text: str) -> int:
        raw = str(text or "").strip()
        if not raw:
            return 0
        pieces = re.findall(r"\S+", raw)
        return max(1, int(sum(len(p) for p in pieces) / 4))

    def _estimate_input_tokens(self, messages: list[dict[str, str]]) -> int:
        total = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            total += self._estimate_tokens(str(m.get("content") or ""))
            total += 4
        return max(0, total)

    def _record_usage(
        self,
        messages: list[dict[str, str]],
        output_text: str,
        payload: dict[str, Any] | None,
    ) -> None:
        self._usage["calls"] += 1
        prompt_tokens = 0
        completion_tokens = 0

        if isinstance(payload, dict):
            usage = payload.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or 0)
            if not prompt_tokens:
                prompt_tokens = int(payload.get("prompt_eval_count") or 0)
            if not completion_tokens:
                completion_tokens = int(payload.get("eval_count") or 0)

        if prompt_tokens > 0:
            self._usage["input_tokens"] += prompt_tokens
        else:
            self._usage["input_tokens_estimated"] += self._estimate_input_tokens(
                messages
            )

        if completion_tokens > 0:
            self._usage["output_tokens"] += completion_tokens
        else:
            self._usage["output_tokens_estimated"] += self._estimate_tokens(output_text)

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        max_tokens = max(256, min(int(self.max_output_tokens or 65536), 65536))
        num_ctx = max(4096, min(int(self.max_context_tokens or 131072), 131072))
        if self.mode == "local":
            resp = self.http.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_ctx": num_ctx,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            payload = resp.json()
            content = str(payload.get("message", {}).get("content", "")).strip()
            self._record_usage(
                messages, content, payload if isinstance(payload, dict) else None
            )
            return content

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
                    "options": {
                        "num_ctx": num_ctx,
                        "num_predict": max_tokens,
                    },
                },
            )
            fallback.raise_for_status()
            payload = fallback.json()
            content = str(payload.get("message", {}).get("content", "")).strip()
            self._record_usage(
                messages, content, payload if isinstance(payload, dict) else None
            )
            return content

        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        choices = payload.get("choices", [])
        if not choices:
            self._record_usage(messages, "", payload)
            return ""
        content = str(choices[0].get("message", {}).get("content", "")).strip()
        self._record_usage(messages, content, payload)
        return content
