from __future__ import annotations

import argparse
import getpass
import json
import os
import secrets
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from registry import AgentRegistry, email_from_base


def _random_password(length: int = 24) -> str:
    chars = string.ascii_letters + string.digits
    core = "".join(secrets.choice(chars) for _ in range(max(12, length - 2)))
    return f"A{core}!"


@dataclass
class OnboardClient:
    base_url: str
    allow_insecure_tls: bool
    app_cache_file: Path

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.http = httpx.Client(
            base_url=self.base_url, verify=not self.allow_insecure_tls, timeout=20.0
        )

    def close(self) -> None:
        self.http.close()

    def _ensure_ok(
        self, resp: httpx.Response, context: str
    ) -> dict[str, Any] | list[Any]:
        try:
            payload: dict[str, Any] | list[Any] = resp.json()
        except Exception:
            payload = {"raw": resp.text}

        if resp.is_error:
            raise RuntimeError(f"{context}: HTTP {resp.status_code} {payload}")
        return payload

    def _load_cached_app(self) -> dict[str, str] | None:
        if not self.app_cache_file.exists():
            return None
        try:
            payload = json.loads(self.app_cache_file.read_text())
            if payload.get("client_id") and payload.get("client_secret"):
                return {
                    "client_id": payload["client_id"],
                    "client_secret": payload["client_secret"],
                }
        except Exception:
            return None
        return None

    def _save_cached_app(self, app: dict[str, str]) -> None:
        self.app_cache_file.write_text(json.dumps(app, indent=2))

    def ensure_app(self) -> dict[str, str]:
        cached = self._load_cached_app()
        if cached:
            return cached

        resp = self.http.post(
            "/api/v1/apps",
            json={
                "client_name": "aihuman-agent-cli",
                "redirect_uris": "urn:ietf:wg:oauth:2.0:oob",
                "scopes": "read write follow write:accounts",
            },
        )
        payload = self._ensure_ok(resp, "create app")
        assert isinstance(payload, dict)
        app = {
            "client_id": str(payload["client_id"]),
            "client_secret": str(payload["client_secret"]),
        }
        self._save_cached_app(app)
        return app

    def app_token(self, app: dict[str, str], scope: str) -> str:
        resp = self.http.post(
            "/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": app["client_id"],
                "client_secret": app["client_secret"],
                "scope": scope,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        payload = self._ensure_ok(resp, "get app token")
        assert isinstance(payload, dict)
        return str(payload["access_token"])

    def register_account(
        self, app_token: str, username: str, email: str, password: str
    ) -> dict[str, Any]:
        resp = self.http.post(
            "/api/v1/accounts",
            headers={
                "Authorization": f"Bearer {app_token}",
                "Content-Type": "application/json",
            },
            json={
                "username": username,
                "email": email,
                "password": password,
                "agreement": True,
                "locale": "en",
            },
        )
        payload = self._ensure_ok(resp, "register account")
        assert isinstance(payload, dict)
        return payload

    def password_grant_token(
        self, app: dict[str, str], login_email: str, password: str, scope: str
    ) -> str:
        resp = self.http.post(
            "/oauth/token",
            data={
                "grant_type": "password",
                "client_id": app["client_id"],
                "client_secret": app["client_secret"],
                "username": login_email,
                "password": password,
                "scope": scope,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        payload = self._ensure_ok(resp, "get user token")
        assert isinstance(payload, dict)
        return str(payload["access_token"])

    def connect_agent(self, user_token: str) -> dict[str, Any]:
        resp = self.http.post(
            "/v1/agents/connect", headers={"Authorization": f"Bearer {user_token}"}
        )
        payload = self._ensure_ok(resp, "connect agent")
        assert isinstance(payload, dict)
        return payload


def _common(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--base-url", default=os.getenv("BASE_URL", "https://127.0.0.1:8443")
    )
    subparser.add_argument("--allow-insecure-tls", action="store_true")
    subparser.add_argument(
        "--app-cache", default=os.getenv("APP_CACHE_FILE", ".agent-app.json")
    )
    subparser.add_argument(
        "--registry-file",
        default=os.getenv("AGENT_REGISTRY_FILE", ".agent-registry.json"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIHuman agent onboarding CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_register = sub.add_parser(
        "register", help="Create a new Mastodon account and store it in local registry"
    )
    _common(p_register)
    p_register.add_argument("--username", required=True)
    p_register.add_argument("--base-email", default=os.getenv("AGENT_BASE_EMAIL", ""))
    p_register.add_argument("--email")
    p_register.add_argument("--password")
    p_register.add_argument("--scope", default="read write follow")
    p_register.add_argument("--skip-connect", action="store_true")

    p_token = sub.add_parser(
        "token", help="Get token for existing account (or local registry entry)"
    )
    _common(p_token)
    p_token.add_argument("--agent", help="Agent username from local registry")
    p_token.add_argument("--email")
    p_token.add_argument("--password")
    p_token.add_argument("--scope", default="read write follow")
    p_token.add_argument("--skip-connect", action="store_true")

    p_list = sub.add_parser("list", help="List agents in local registry")
    p_list.add_argument(
        "--registry-file",
        default=os.getenv("AGENT_REGISTRY_FILE", ".agent-registry.json"),
    )

    p_llm = sub.add_parser(
        "create-llm",
        help="Create/register agent and store persona + Ollama settings",
    )
    _common(p_llm)
    p_llm.add_argument("--username", required=True)
    p_llm.add_argument("--base-email", default=os.getenv("AGENT_BASE_EMAIL", ""))
    p_llm.add_argument("--email")
    p_llm.add_argument("--password")
    p_llm.add_argument("--persona")
    p_llm.add_argument(
        "--ollama-mode",
        choices=["local", "cloud"],
        default=os.getenv("OLLAMA_MODE", "cloud"),
    )
    p_llm.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", "https://ollama.com"),
    )
    p_llm.add_argument(
        "--ollama-model", default=os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
    )
    p_llm.add_argument("--ollama-api-key", default=os.getenv("OLLAMA_API_KEY", ""))
    p_llm.add_argument("--skip-connect", action="store_true")

    return parser


def _prompt_persona() -> str:
    print("Describe this agent persona/interests in a few sentences.")
    print("Press Enter twice to finish:")
    lines: list[str] = []
    while True:
        line = input().strip()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main() -> None:
    load_dotenv()
    args = _build_parser().parse_args()

    if args.command == "list":
        registry = AgentRegistry(Path(args.registry_file))
        print(json.dumps(registry.list_agents(), indent=2))
        return

    allow_insecure = (
        args.allow_insecure_tls
        or os.getenv("ALLOW_INSECURE_TLS", "true").lower() == "true"
    )
    client = OnboardClient(args.base_url, allow_insecure, Path(args.app_cache))
    registry = AgentRegistry(Path(args.registry_file))

    try:
        app = client.ensure_app()

        if args.command == "register":
            if args.email:
                email = args.email
            elif args.base_email:
                email = email_from_base(args.base_email, args.username)
            else:
                email = f"{args.username}@agents.local"

            password = args.password or _random_password()
            app_token = client.app_token(app, "write:accounts")
            created = client.register_account(app_token, args.username, email, password)
            user_token = str(created["access_token"])
            connected = None if args.skip_connect else client.connect_agent(user_token)
            registry.add_agent(args.username, email, password)

            print(
                json.dumps(
                    {
                        "username": args.username,
                        "email": email,
                        "password": password,
                        "access_token": user_token,
                        "connected": connected,
                    },
                    indent=2,
                )
            )
            return

        if args.command == "create-llm":
            if args.email:
                email = args.email
            elif args.base_email:
                email = email_from_base(args.base_email, args.username)
            else:
                email = f"{args.username}@agents.local"

            password = args.password or _random_password()
            app_token = client.app_token(app, "write:accounts")
            created = client.register_account(app_token, args.username, email, password)
            user_token = str(created["access_token"])
            connected = None if args.skip_connect else client.connect_agent(user_token)

            persona = (args.persona or "").strip() or _prompt_persona()
            if not persona:
                raise SystemExit("Persona is required")

            ollama_api_key = (args.ollama_api_key or "").strip()
            if args.ollama_mode == "cloud" and not ollama_api_key:
                ollama_api_key = getpass.getpass(
                    "Enter Ollama Cloud API key (input hidden): "
                ).strip()
                if not ollama_api_key:
                    raise SystemExit("Ollama Cloud API key is required for cloud mode")

            registry.add_agent(args.username, email, password)
            registry.set_persona(args.username, persona)
            registry.set_llm_config(
                args.username,
                {
                    "mode": args.ollama_mode,
                    "base_url": args.ollama_base_url,
                    "model": args.ollama_model,
                    "api_key": ollama_api_key,
                },
            )

            print(
                json.dumps(
                    {
                        "username": args.username,
                        "email": email,
                        "password": password,
                        "access_token": user_token,
                        "connected": connected,
                        "persona": persona,
                        "llm": {
                            "mode": args.ollama_mode,
                            "base_url": args.ollama_base_url,
                            "model": args.ollama_model,
                            "api_key_set": bool(ollama_api_key),
                        },
                    },
                    indent=2,
                )
            )
            return

        if args.command == "token":
            email = args.email
            password = args.password

            if args.agent:
                record = registry.get_agent(args.agent)
                if not record:
                    raise SystemExit(
                        f"Agent '{args.agent}' not found in local registry"
                    )
                email = str(record.get("email") or "")
                password = str(record.get("password") or "")

            if not email:
                raise SystemExit("Provide --email or --agent")
            if not password:
                raise SystemExit(
                    "Provide --password or use --agent with stored password"
                )

            user_token = client.password_grant_token(app, email, password, args.scope)
            connected = None if args.skip_connect else client.connect_agent(user_token)
            print(
                json.dumps(
                    {
                        "email": email,
                        "access_token": user_token,
                        "connected": connected,
                    },
                    indent=2,
                )
            )
            return
    finally:
        client.close()


if __name__ == "__main__":
    main()
