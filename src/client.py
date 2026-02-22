from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


@dataclass
class AgentClient:
    base_url: str
    token: str
    allow_insecure_tls: bool = False

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._max_post_chars_cache: int | None = None
        self._http = httpx.Client(
            base_url=self.base_url,
            verify=not self.allow_insecure_tls,
            timeout=20.0,
            headers={"Authorization": f"Bearer {self.token}"},
        )

    def close(self) -> None:
        self._http.close()

    def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._http.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def connect_agent(self) -> dict[str, Any]:
        return dict(self._json("POST", "/v1/agents/connect"))

    def verify_credentials(self) -> dict[str, Any]:
        payload = self._json("GET", "/api/v1/accounts/verify_credentials")
        return dict(payload) if isinstance(payload, dict) else {}

    def agent_status(self) -> dict[str, Any]:
        return dict(self._json("GET", "/v1/agents/status"))

    def post(
        self,
        text: str,
        in_reply_to_id: str | None = None,
        visibility: str | None = None,
    ) -> dict[str, Any]:
        data = {"status": text}
        if in_reply_to_id:
            data["in_reply_to_id"] = in_reply_to_id
        if visibility:
            data["visibility"] = visibility
        return dict(
            self._json(
                "POST",
                "/api/v1/statuses",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        )

    def read_home(self, limit: int = 20) -> list[Any]:
        payload = self._json(
            "GET", f"/api/v1/timelines/home?limit={max(1, min(limit, 80))}"
        )
        return payload if isinstance(payload, list) else []

    def account_statuses(self, account_id: str, limit: int = 20) -> list[Any]:
        payload = self._json(
            "GET",
            f"/api/v1/accounts/{quote(str(account_id))}/statuses?limit={max(1, min(limit, 80))}&exclude_reblogs=true",
        )
        return payload if isinstance(payload, list) else []

    def follow(self, acct: str) -> dict[str, Any]:
        found = self._json(
            "GET",
            f"/api/v1/accounts/search?q={quote(acct)}&resolve=true&limit=1",
        )
        if not isinstance(found, list) or not found:
            raise RuntimeError(f"Account not found: {acct}")
        account_id = found[0]["id"]
        return dict(self._json("POST", f"/api/v1/accounts/{account_id}/follow"))

    def dm(self, to: str, text: str) -> dict[str, Any]:
        return dict(
            self._json(
                "POST",
                "/api/v1/statuses",
                data={"status": f"@{to.lstrip('@')} {text}", "visibility": "direct"},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        )

    def notifications(self, limit: int = 20) -> list[Any]:
        payload = self._json(
            "GET", f"/api/v1/notifications?limit={max(1, min(limit, 80))}"
        )
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(
            payload.get("notification_groups"), list
        ):
            groups = payload.get("notification_groups") or []
            statuses = payload.get("statuses") or {}
            accounts = payload.get("accounts") or {}
            normalized: list[dict[str, Any]] = []
            for g in groups:
                if not isinstance(g, dict):
                    continue
                sample = g.get("sample_account_ids") or []
                account = accounts.get(str(sample[0]), {}) if sample else {}
                status = (
                    statuses.get(str(g.get("status_id")), {})
                    if g.get("status_id")
                    else {}
                )
                normalized.append(
                    {
                        "type": g.get("type"),
                        "id": g.get("notification_group_key") or g.get("id"),
                        "account": account if isinstance(account, dict) else {},
                        "status": status if isinstance(status, dict) else {},
                    }
                )
            return normalized
        return []

    def mention_notifications_all(self, page_limit: int = 80) -> list[dict[str, Any]]:
        per_page = max(1, min(page_limit, 80))
        max_id: str | None = None
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for _ in range(25):
            path = f"/api/v1/notifications?types[]=mention&limit={per_page}"
            if max_id:
                path += f"&max_id={quote(max_id)}"

            payload = self._json("GET", path)
            if not isinstance(payload, list) or not payload:
                break

            page_items: list[dict[str, Any]] = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                nid = str(item.get("id") or "")
                if not nid or nid in seen_ids:
                    continue
                seen_ids.add(nid)
                page_items.append(item)

            if not page_items:
                break

            out.extend(page_items)

            last_id = str(page_items[-1].get("id") or "")
            if not last_id:
                break
            max_id = last_id

            if len(page_items) < per_page:
                break

        return out

    def mention_notifications_since(
        self, since_id: str | None, page_limit: int = 80
    ) -> list[dict[str, Any]]:
        per_page = max(1, min(page_limit, 80))
        max_id: str | None = None
        out: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        since_num: int | None = None
        try:
            since_num = int(str(since_id)) if since_id else None
        except Exception:
            since_num = None

        for _ in range(25):
            path = f"/api/v1/notifications?types[]=mention&limit={per_page}"
            if max_id:
                path += f"&max_id={quote(max_id)}"

            payload = self._json("GET", path)
            if not isinstance(payload, list) or not payload:
                break

            page_items: list[dict[str, Any]] = []
            reached_seen_boundary = False
            for item in payload:
                if not isinstance(item, dict):
                    continue
                nid = str(item.get("id") or "")
                if not nid or nid in seen_ids:
                    continue

                if since_num is not None:
                    try:
                        if int(nid) <= since_num:
                            reached_seen_boundary = True
                            continue
                    except Exception:
                        pass

                seen_ids.add(nid)
                page_items.append(item)

            if page_items:
                out.extend(page_items)

            last_id = str((payload[-1] or {}).get("id") or "")
            if not last_id:
                break
            max_id = last_id

            if len(payload) < per_page or reached_seen_boundary:
                break

        return out

    def dismiss_notification(self, notification_id: str) -> dict[str, Any]:
        payload = self._json(
            "POST", f"/api/v1/notifications/{quote(str(notification_id))}/dismiss"
        )
        return dict(payload) if isinstance(payload, dict) else {}

    def trends_statuses(self, limit: int = 20) -> list[Any]:
        payload = self._json(
            "GET", f"/api/v1/trends/statuses?limit={max(1, min(limit, 40))}"
        )
        return payload if isinstance(payload, list) else []

    def status_context(self, status_id: str) -> dict[str, Any]:
        payload = self._json("GET", f"/api/v1/statuses/{quote(str(status_id))}/context")
        return dict(payload) if isinstance(payload, dict) else {}

    def status(self, status_id: str) -> dict[str, Any]:
        payload = self._json("GET", f"/api/v1/statuses/{quote(str(status_id))}")
        return dict(payload) if isinstance(payload, dict) else {}

    def favourite(self, status_id: str) -> dict[str, Any]:
        payload = self._json(
            "POST", f"/api/v1/statuses/{quote(str(status_id))}/favourite"
        )
        return dict(payload) if isinstance(payload, dict) else {}

    def boost(self, status_id: str) -> dict[str, Any]:
        payload = self._json("POST", f"/api/v1/statuses/{quote(str(status_id))}/reblog")
        return dict(payload) if isinstance(payload, dict) else {}

    def search_accounts(self, acct_query: str, limit: int = 5) -> list[Any]:
        payload = self._json(
            "GET",
            f"/api/v1/accounts/search?q={quote(acct_query)}&resolve=true&limit={max(1, min(limit, 80))}",
        )
        return payload if isinstance(payload, list) else []

    def max_post_characters(self) -> int:
        if self._max_post_chars_cache is not None:
            return self._max_post_chars_cache

        limit = 500
        try:
            payload = self._json("GET", "/api/v2/instance")
            conf = payload.get("configuration", {}) if isinstance(payload, dict) else {}
            statuses = conf.get("statuses", {}) if isinstance(conf, dict) else {}
            value = (
                statuses.get("max_characters") if isinstance(statuses, dict) else None
            )
            if isinstance(value, int) and value > 0:
                limit = value
        except Exception:
            pass

        self._max_post_chars_cache = limit
        return limit
