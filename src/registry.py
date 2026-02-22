from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


def email_from_base(base_email: str, username: str) -> str:
    local, _, domain = base_email.partition("@")
    if not local or not domain:
        raise ValueError("base email must look like name@example.com")
    return f"{local}+{username}@{domain}"


@dataclass
class AgentRegistry:
    file_path: Path

    def __post_init__(self) -> None:
        self.key_file = Path(
            os.getenv(
                "AGENT_REGISTRY_KEY_FILE", str(self.file_path.with_suffix(".key"))
            )
        )

    def _fernet(self) -> Fernet:
        if not self.key_file.exists():
            self.key_file.write_bytes(Fernet.generate_key())
            try:
                self.key_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass
        return Fernet(self.key_file.read_bytes().strip())

    def _encrypt_password(self, password: str) -> str:
        return self._fernet().encrypt(password.encode("utf-8")).decode("utf-8")

    def _decrypt_password(self, token: str) -> str:
        return self._fernet().decrypt(token.encode("utf-8")).decode("utf-8")

    def _load(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {"agents": {}}
        try:
            data = json.loads(self.file_path.read_text())
            if not isinstance(data, dict):
                return {"agents": {}}
            if "agents" not in data or not isinstance(data["agents"], dict):
                data["agents"] = {}
            return data
        except Exception:
            return {"agents": {}}

    def _save(self, data: dict[str, Any]) -> None:
        self.file_path.write_text(json.dumps(data, indent=2))

    def add_agent(self, username: str, email: str, password: str) -> None:
        data = self._load()
        existing = (
            data["agents"].get(username)
            if isinstance(data["agents"].get(username), dict)
            else {}
        )
        data["agents"][username] = {
            "username": username,
            "email": email,
            "password_enc": self._encrypt_password(password),
            "persona": existing.get("persona", ""),
            "llm": existing.get("llm", {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save(data)

    def set_persona(self, username: str, persona: str) -> None:
        data = self._load()
        agent = data["agents"].get(username)
        if not isinstance(agent, dict):
            raise KeyError(f"Agent '{username}' not found")
        agent["persona"] = persona.strip()
        data["agents"][username] = agent
        self._save(data)

    def set_llm_config(self, username: str, llm: dict[str, Any]) -> None:
        data = self._load()
        agent = data["agents"].get(username)
        if not isinstance(agent, dict):
            raise KeyError(f"Agent '{username}' not found")
        agent["llm"] = llm
        data["agents"][username] = agent
        self._save(data)

    def get_agent(self, username: str) -> dict[str, Any] | None:
        data = self._load()
        value = data["agents"].get(username)
        if isinstance(value, dict):
            out = dict(value)
            if isinstance(out.get("password"), str):
                return out
            if isinstance(out.get("password_enc"), str):
                try:
                    out["password"] = self._decrypt_password(out["password_enc"])
                except Exception:
                    out["password"] = ""
            return out
        return None

    def list_agents(self) -> list[dict[str, Any]]:
        data = self._load()
        values: list[dict[str, Any]] = []
        for _, value in sorted(data["agents"].items()):
            if isinstance(value, dict):
                item = dict(value)
                item.pop("password", None)
                item.pop("password_enc", None)
                item["has_password"] = True
                values.append(item)
        return values
