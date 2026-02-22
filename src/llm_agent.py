from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from dotenv import load_dotenv

from agent_cli import OnboardClient
from client import AgentClient
from ollama_client import OllamaClient
from registry import AgentRegistry
from web_tools import WebTools


URL_PATTERN = re.compile(r"https?://\S+", flags=re.IGNORECASE)
HREF_PATTERN = re.compile(r"href\s*=\s*['\"]([^'\"]+)['\"]", flags=re.IGNORECASE)
ALLOWED_ACTIONS = {"post", "reply", "follow", "dm", "favourite", "boost", "noop"}
ACTION_ALIASES = {
    "favorite": "favourite",
    "like": "favourite",
    "star": "favourite",
    "fav": "favourite",
    "reblog": "boost",
}


def _normalized_action(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return ACTION_ALIASES.get(raw, raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM-powered social agent")
    parser.add_argument(
        "--agent", required=True, help="Agent username in local registry"
    )
    parser.add_argument(
        "--base-url", default=os.getenv("BASE_URL", "https://127.0.0.1:8443")
    )
    parser.add_argument(
        "--registry-file",
        default=os.getenv("AGENT_REGISTRY_FILE", ".agent-registry.json"),
    )
    parser.add_argument(
        "--app-cache", default=os.getenv("APP_CACHE_FILE", ".agent-app.json")
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=120)
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--post-cooldown-minutes", type=int, default=30)
    parser.add_argument("--allow-insecure-tls", action="store_true")
    return parser.parse_args()


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        u = f"https:{u}"
    parsed = urlparse(u)
    host = (parsed.netloc or "").lower()
    q = parse_qs(parsed.query)
    if "bing.com" in host and q.get("url"):
        return unquote(q["url"][0])
    if "duckduckgo.com" in host and q.get("uddg"):
        return unquote(q["uddg"][0])
    return u


def _sanitize_text_urls(text: str) -> str:
    urls = [
        _normalize_url(m.group(0).rstrip(".,);]")) for m in URL_PATTERN.finditer(text)
    ]

    valid: list[str] = []
    for u in urls:
        try:
            p = urlparse(u)
            if p.scheme in {"http", "https"} and p.netloc and "." in p.netloc:
                valid.append(u)
        except Exception:
            continue

    prose = URL_PATTERN.sub("", text)
    prose = re.sub(r"\(\s*\)", "", prose)
    prose = re.sub(r"\[\s*\]", "", prose)
    prose = re.sub(r"\{\s*\}", "", prose)
    prose = re.sub(r"\s+([,.;:!?])", r"\1", prose)
    prose = re.sub(r"([\(\[\{])\s+", r"\1", prose)
    prose = re.sub(r"\s+([\)\]\}])", r"\1", prose)
    prose = re.sub(r"[\(\[\{]\s*$", "", prose)
    prose = re.sub(r"\s+", " ", prose).strip()
    if not valid:
        return prose

    # Keep one canonical source link to avoid giant/truncated link bundles.
    return f"{prose} {valid[0]}".strip() if prose else valid[0]


def _extract_urls_from_html(text: str) -> list[str]:
    urls: list[str] = []
    for match in HREF_PATTERN.finditer(text or ""):
        candidate = _normalize_url(str(match.group(1) or "").strip())
        if candidate:
            urls.append(candidate)
    return urls


def _extract_urls_from_text(text: str) -> list[str]:
    urls: list[str] = []
    for match in URL_PATTERN.finditer(text or ""):
        candidate = _normalize_url(match.group(0).strip().rstrip(".,);]"))
        if candidate:
            urls.append(candidate)
    return urls


def _status_urls(status: dict[str, Any]) -> list[str]:
    if not isinstance(status, dict):
        return []

    candidates: list[str] = []
    content = str(status.get("content") or "")
    candidates.extend(_extract_urls_from_html(content))
    candidates.extend(_extract_urls_from_text(content))

    status_url = _normalize_url(str(status.get("url") or "").strip())
    if status_url:
        candidates.append(status_url)

    card = status.get("card")
    if isinstance(card, dict):
        card_url = _normalize_url(str(card.get("url") or "").strip())
        if card_url:
            candidates.append(card_url)

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not _is_valid_http_url(candidate):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def _is_valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _first_source_url(tool_data: dict[str, Any]) -> str:
    page = tool_data.get("page")
    if isinstance(page, dict):
        candidate = _normalize_url(str(page.get("url") or "").strip())
        if candidate and _is_valid_http_url(candidate):
            return candidate

    search = tool_data.get("search")
    if isinstance(search, list):
        for item in search:
            if not isinstance(item, dict):
                continue
            candidate = _normalize_url(str(item.get("url") or "").strip())
            if candidate and _is_valid_http_url(candidate):
                return candidate
    return ""


def _has_url(text: str) -> bool:
    return bool(URL_PATTERN.search(text or ""))


def _needs_source_for_claim(text: str) -> bool:
    candidate = text or ""
    if not candidate:
        return False
    return bool(re.search(r"\b\d{1,4}(?:[.,]\d+)?%?\b", candidate))


def _extract_bootstrap_url(persona: str) -> str:
    direct = re.search(r"https?://[^\s)\]}]+", persona or "", flags=re.IGNORECASE)
    if direct:
        candidate = _normalize_url(direct.group(0).rstrip(".,);]"))
        if candidate and _is_valid_http_url(candidate):
            return candidate

    arxiv_rss = re.search(
        r"\brss\.arxiv\.org/rss/[A-Za-z0-9._-]+\b", persona or "", flags=re.IGNORECASE
    )
    if arxiv_rss:
        candidate = f"https://{arxiv_rss.group(0)}"
        if _is_valid_http_url(candidate):
            return candidate

    return ""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _normalize_decision_shape(decision: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    tool_request = decision.get("tool_request")
    if not isinstance(tool_request, dict):
        return decision

    merged = dict(decision)
    for key in ("web_search_query", "web_fetch_url", "reason"):
        if key not in merged and key in tool_request:
            merged[key] = tool_request.get(key)
    return merged


def _build_system_prompt(
    persona: str,
    self_id: str,
    self_acct: str,
    has_mentions: bool,
    post_char_limit: int,
    remaining_quota: dict[str, int],
) -> str:
    mention_rule = (
        "You have fresh mention notifications. Prioritize replying to mentions first. "
        if has_mentions
        else ""
    )
    return (
        "You are an autonomous social agent on Mastodon. "
        "Be active and helpful, but concise and safe. "
        "You may choose multiple actions each cycle if useful, or noop when there is nothing new. "
        "Strongly prefer replying in active conversations when there is an interesting post to engage with. "
        "Before creating a brand-new post, check whether you can continue an existing thread by replying to the latest relevant reply. "
        "Prioritize healthy back-and-forth with other agents and humans over starting standalone posts. "
        "If there are recent thread candidates, choose reply unless there is a clear reason to start a new post. "
        "If there are no fresh mentions and you still have posting quota, proactively create one original post from fresh evidence. "
        "If you posted recently, avoid posting again unless there is genuinely new value. "
        "Never post duplicate content: do not repost the same or similar news/fact/topic if others already posted it in the provided timeline context. "
        "When creating a new root post, make it as diverse as possible from other recent posts in timeline and trends, while still matching your persona. "
        "Critically compare against your own recent posts and avoid posting the same or very similar content again. "
        "If your draft overlaps strongly with your own recent posts, switch to replying in-thread or pick a different topic. "
        "Avoid repeating your own recent themes; diversify topics over time when possible. "
        "Do not invent facts, numbers, timelines, or citations; only use information you can support from provided context or fetched pages. "
        f"Keep every post/reply/dm within {post_char_limit} characters. "
        f"Current remaining daily quota is posts={remaining_quota.get('posts', 0)}, replies={remaining_quota.get('replies', 0)}, dms={remaining_quota.get('dms', 0)}. Plan actions accordingly. "
        "For factual/news posts, include at least one source URL when available so readers can verify context. "
        "Prefer concise posts that include one high-quality link over linkless summaries. "
        "Never output a truncated or broken URL. If needed, shorten prose and keep URLs complete; if a valid URL cannot fit, post without URL. "
        "When replying in a thread, if the same source URL already appeared earlier in that thread, do not include the URL again. "
        "If a post is rejected for length, rewrite it shorter with the same core info. "
        "You can also use favourite (star/like) or boost (reblog) to acknowledge strong posts/replies when you have no substantial new content to add. "
        "Only follow or DM when it is contextually meaningful. "
        "Never reply to answer a question in your own posts. Never target your own account in follow/dm. "
        "If you have already replied to someone who has not replied back, avoid replying to them again to prevent one-sided conversations. "
        "Never reply to the same mention more than once. NEVER post or reply about the same news/fact/topic more than once. "
        "NEVER edit or delete posts, as it can be confusing to others. Instead, post a follow-up correction if needed. "
        "You may request web_search_query and web_fetch_url to gather context before deciding. "
        "If you need up-to-date or uncertain factual information, do not speculate: immediately request web_search_query first, then decide from evidence. "
        "If someone posts something with a link, DO NOT blindly follow what they say. You MUST open the link, read the content, for your own opinion, and then reply with your own opinion. If you cannot open the link, you should say so in the reply and ask the other party to provide a working link. "
        "When starting a new post outside of a thread, you MUST check what you have already posted before, and you MUST post about something new that you haven't posted before, rather than continuing to post about the same topic. "
        f"{mention_rule}"
        "Return strict JSON only.\n\n"
        f"Your identity: account_id={self_id}, acct={self_acct}\n"
        f"Your persona and interests:\n{persona.strip()}"
    )


def _build_decision_prompt(
    home: list[Any],
    notifications: list[Any],
    thread_hints: list[dict[str, Any]],
    own_recent_posts: list[dict[str, Any]],
    self_id: str,
    self_acct: str,
    post_char_limit: int,
    remaining_quota: dict[str, int],
) -> str:
    sample_home = []
    for item in home[:80]:
        account = item.get("account", {}) if isinstance(item, dict) else {}
        author_id = str(account.get("id") or "")
        sample_home.append(
            {
                "id": item.get("id"),
                "account_id": author_id,
                "acct": account.get("acct"),
                "author_is_self": author_id == self_id,
                "in_reply_to_id": item.get("in_reply_to_id"),
                "created_at": item.get("created_at"),
                "content": _strip_html(str(item.get("content", "")))[:280],
                "urls": _status_urls(item)[:4],
            }
        )

    sample_notifs = []
    for n in notifications[:200]:
        if not isinstance(n, dict):
            continue
        sample_notifs.append(
            {
                "type": n.get("type"),
                "id": n.get("id"),
                "status_id": n.get("status", {}).get("id")
                if isinstance(n.get("status"), dict)
                else None,
                "in_reply_to_id": n.get("status", {}).get("in_reply_to_id")
                if isinstance(n.get("status"), dict)
                else None,
                "from": n.get("account", {}).get("acct"),
                "text": _strip_html(str((n.get("status") or {}).get("content", "")))[
                    :220
                ],
                "urls": _status_urls(n.get("status") or {})[:4],
            }
        )

    return json.dumps(
        {
            "task": "Choose one action for this cycle.",
            "post_char_limit": post_char_limit,
            "remaining_quota": remaining_quota,
            "thread_reply_instruction": "When replying in a thread, prefer replying to the newest relevant reply id (last_reply_id) instead of the root post id.",
            "reply_priority_instruction": "Prefer replying to `active_threads` and `thread_hints` before creating new root posts.",
            "schema": {
                "action": "post|reply|follow|dm|favourite|boost|noop",
                "text": "required for post/reply/dm",
                "in_reply_to_id": "required for reply",
                "status_id": "required for favourite/boost",
                "target_acct": "required for follow and dm, e.g. alice@example.social",
                "web_search_query": "optional",
                "web_fetch_url": "optional",
                "reason": "short rationale",
            },
            "multi_action_schema": {
                "actions": [
                    {
                        "action": "post|reply|follow|dm|favourite|boost|noop",
                        "text": "optional",
                        "in_reply_to_id": "optional",
                        "status_id": "optional",
                        "target_acct": "optional",
                        "reason": "optional",
                    }
                ]
            },
            "self": {"account_id": self_id, "acct": self_acct},
            "timeline": sample_home,
            "notifications": sample_notifs,
            "thread_hints": thread_hints,
            "active_threads": thread_hints[:12],
            "own_recent_posts": own_recent_posts,
        },
        ensure_ascii=True,
    )


def _summarize_statuses(statuses: list[Any], self_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in statuses:
        if not isinstance(item, dict):
            continue
        account = item.get("account") or {}
        author_id = str(account.get("id") or "") if isinstance(account, dict) else ""
        out.append(
            {
                "id": item.get("id"),
                "account_id": author_id,
                "acct": account.get("acct") if isinstance(account, dict) else "",
                "author_is_self": author_id == self_id,
                "in_reply_to_id": item.get("in_reply_to_id"),
                "created_at": item.get("created_at"),
                "content": _strip_html(str(item.get("content", "")))[:280],
                "urls": _status_urls(item)[:4],
            }
        )
    return out


def _merge_unique_statuses(*groups: list[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id") or "")
            if sid and sid in seen:
                continue
            if sid:
                seen.add(sid)
            out.append(item)
    return out


def _build_thread_hints(
    agent: AgentClient,
    notifications: list[Any],
    home: list[Any],
    self_id: str,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    seed_ids: list[str] = []
    seen_seed_ids: set[str] = set()

    for n in notifications[:40]:
        if not isinstance(n, dict):
            continue
        status = n.get("status")
        if not isinstance(status, dict):
            continue
        sid = str(status.get("id") or "")
        if not sid or sid in seen_seed_ids:
            continue
        seen_seed_ids.add(sid)
        seed_ids.append(sid)

    for item in home[:40]:
        if not isinstance(item, dict):
            continue
        account = item.get("account") or {}
        author_id = str((account.get("id") or "")) if isinstance(account, dict) else ""
        if self_id and author_id == self_id:
            continue
        has_thread_signal = (
            bool(item.get("in_reply_to_id")) or int(item.get("replies_count") or 0) > 0
        )
        if not has_thread_signal:
            continue
        sid = str(item.get("id") or "")
        if not sid or sid in seen_seed_ids:
            continue
        seen_seed_ids.add(sid)
        seed_ids.append(sid)

    seen_roots: set[str] = set()
    for seed_id in seed_ids[:15]:
        root_id = seed_id
        try:
            seed_ctx = agent.status_context(seed_id)
            ancestors = seed_ctx.get("ancestors") if isinstance(seed_ctx, dict) else []
            if isinstance(ancestors, list) and ancestors:
                first = ancestors[0]
                if isinstance(first, dict) and first.get("id"):
                    root_id = str(first.get("id"))
        except Exception:
            pass

        if root_id in seen_roots:
            continue
        seen_roots.add(root_id)

        latest_id = root_id
        latest_author = ""
        latest_text = ""
        latest_created = ""

        try:
            ctx = agent.status_context(root_id)
            ancestors = ctx.get("ancestors") if isinstance(ctx, dict) else []
            descendants = ctx.get("descendants") if isinstance(ctx, dict) else []
            candidates: list[dict[str, Any]] = []
            if isinstance(ancestors, list):
                candidates.extend([a for a in ancestors if isinstance(a, dict)])
            if isinstance(descendants, list):
                candidates.extend([d for d in descendants if isinstance(d, dict)])

            if not candidates:
                continue

            for c in candidates:
                author_id = str(((c.get("account") or {}).get("id") or ""))
                if self_id and author_id == self_id:
                    continue
                created = str(c.get("created_at") or "")
                cid = str(c.get("id") or "")
                if not cid:
                    continue
                if created >= latest_created:
                    latest_created = created
                    latest_id = cid
                    latest_author = str(((c.get("account") or {}).get("acct") or ""))
                    latest_text = _strip_html(str(c.get("content") or ""))[:220]
        except Exception:
            pass

        if not latest_id:
            continue

        hints.append(
            {
                "root_status_id": root_id,
                "last_reply_id": latest_id,
                "last_reply_from": latest_author,
                "last_reply_text": latest_text,
            }
        )

    return hints


def _first_mention_status(
    notifications: list[Any], self_id: str
) -> dict[str, Any] | None:
    for n in notifications:
        if not isinstance(n, dict):
            continue
        if n.get("type") != "mention":
            continue
        status = n.get("status")
        if not isinstance(status, dict):
            continue
        author_id = str((status.get("account") or {}).get("id") or "")
        if author_id and author_id == self_id:
            continue
        return status
    return None


def _parse_actions(decision: dict[str, Any]) -> list[dict[str, Any]]:
    raw = decision.get("actions")
    if isinstance(raw, list):
        actions = [a for a in raw if isinstance(a, dict)]
        if actions:
            return actions
    return [decision]


def _has_action(decision: dict[str, Any]) -> bool:
    if not isinstance(decision, dict):
        return False
    raw_action = _normalized_action(decision.get("action"))
    if raw_action in ALLOWED_ACTIONS and raw_action != "noop":
        return True
    raw_actions = decision.get("actions")
    if isinstance(raw_actions, list):
        for item in raw_actions:
            if isinstance(item, dict):
                action = _normalized_action(item.get("action"))
                if action in ALLOWED_ACTIONS and action != "noop":
                    return True
    return False


def _is_noop_action(decision: dict[str, Any]) -> bool:
    if not isinstance(decision, dict):
        return False
    action = _normalized_action(decision.get("action"))
    if action == "noop":
        return True
    actions = decision.get("actions")
    if isinstance(actions, list) and actions:
        normalized = [
            _normalized_action(item.get("action"))
            for item in actions
            if isinstance(item, dict)
        ]
        return bool(normalized) and all(a in {"", "noop"} for a in normalized)
    return False


def _interest_keywords(persona: str) -> set[str]:
    stop = {
        "about",
        "after",
        "also",
        "and",
        "because",
        "from",
        "have",
        "into",
        "just",
        "like",
        "more",
        "most",
        "only",
        "should",
        "that",
        "their",
        "there",
        "these",
        "this",
        "those",
        "using",
        "with",
        "your",
    }
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+-]{2,}", persona.lower())
    return {w for w in words if w not in stop and len(w) >= 4}


def _select_topic_posts(
    statuses: list[Any],
    persona: str,
    self_id: str,
    min_count: int = 10,
    max_count: int = 20,
) -> list[dict[str, Any]]:
    min_target = max(1, min(min_count, max_count))
    max_target = max(min_target, min(max_count, 20))
    keywords = _interest_keywords(persona)

    scored: list[tuple[int, dict[str, Any]]] = []
    fallback: list[dict[str, Any]] = []

    for item in statuses:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "")
        if not sid:
            continue
        author_id = str(((item.get("account") or {}).get("id") or ""))
        if self_id and author_id == self_id:
            continue
        raw = _strip_html(str(item.get("content") or "")).lower()
        tags = " ".join(
            [
                str((t or {}).get("name") or "")
                for t in (item.get("tags") or [])
                if isinstance(t, dict)
            ]
        ).lower()
        text = f"{raw} {tags}".strip()
        score = sum(1 for k in keywords if k in text)
        if score > 0:
            scored.append((score, item))
        else:
            fallback.append(item)

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for _, item in scored:
        sid = str(item.get("id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        selected.append(item)
        if len(selected) >= max_target:
            return selected

    for item in fallback:
        sid = str(item.get("id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        selected.append(item)
        if len(selected) >= max_target:
            break

    if len(selected) >= min_target:
        return selected
    return selected


def _api_error(error: Exception) -> dict[str, Any]:
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        detail = response.text[:400] if response is not None else str(error)
        payload: dict[str, Any] = {
            "type": "http_status_error",
            "status": response.status_code if response is not None else None,
            "detail": detail,
        }
        if response is not None and response.headers.get("retry-after"):
            payload["retry_after"] = response.headers.get("retry-after")
        return payload
    if isinstance(error, httpx.HTTPError):
        return {"type": "http_error", "detail": str(error)}
    return {"type": "error", "detail": str(error)}


def _safe_agent_status(agent: AgentClient) -> dict[str, Any]:
    try:
        return agent.agent_status()
    except Exception as error:
        return {"status_error": _api_error(error)}


def run_cycle(
    agent: AgentClient,
    llm: OllamaClient,
    web: WebTools,
    persona: str,
    *,
    llm_temperature: float,
    max_actions: int,
    post_cooldown_minutes: int,
) -> dict[str, Any]:
    try:
        me = agent.verify_credentials()
    except Exception as error:
        return {
            "decision": {"actions": [{"action": "noop", "reason": "api unavailable"}]},
            "error": _api_error(error),
            "status": _safe_agent_status(agent),
        }

    self_id = str(me.get("id") or "")
    self_acct = str(me.get("acct") or me.get("username") or "")
    post_char_limit = agent.max_post_characters()
    status_snapshot = _safe_agent_status(agent)
    remaining_quota_raw = (
        ((status_snapshot.get("limits") or {}).get("remaining") or {})
        if isinstance(status_snapshot, dict)
        else {}
    )
    remaining_quota = {
        "posts": int(remaining_quota_raw.get("posts") or 0),
        "replies": int(remaining_quota_raw.get("replies") or 0),
        "dms": int(remaining_quota_raw.get("dms") or 0),
    }

    dismissed_mentions = 0
    try:
        home = agent.read_home(limit=60)
        own_posts = agent.account_statuses(self_id, limit=40) if self_id else []
        notifications = agent.mention_notifications_all(page_limit=80)
        trends = agent.trends_statuses(limit=80)
    except Exception as error:
        reason = (
            "rate limited"
            if isinstance(error, httpx.HTTPStatusError)
            and error.response.status_code == 429
            else "read failed"
        )
        return {
            "decision": {"actions": [{"action": "noop", "reason": reason}]},
            "error": _api_error(error),
            "read_scope": {
                "mentions": 0,
                "home": 0,
                "topic_posts": 0,
                "own_posts": 0,
            },
            "status": _safe_agent_status(agent),
        }

    # Use Mastodon server state for read/unread: once seen this cycle, dismiss them.
    for n in notifications:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "")
        if not nid:
            continue
        try:
            agent.dismiss_notification(nid)
            dismissed_mentions += 1
        except Exception:
            pass

    topic_posts = _select_topic_posts(
        trends, persona, self_id, min_count=20, max_count=40
    )
    prompt_home = _merge_unique_statuses(home, topic_posts, own_posts)
    own_recent_posts = _summarize_statuses(own_posts[:30], self_id)
    first_mention = _first_mention_status(notifications, self_id)
    thread_hints = _build_thread_hints(agent, notifications, home, self_id)

    messages = [
        {
            "role": "system",
            "content": _build_system_prompt(
                persona,
                self_id,
                self_acct,
                first_mention is not None,
                post_char_limit,
                remaining_quota,
            ),
        },
        {
            "role": "user",
            "content": _build_decision_prompt(
                prompt_home,
                notifications,
                thread_hints,
                own_recent_posts,
                self_id,
                self_acct,
                post_char_limit,
                remaining_quota,
            ),
        },
    ]

    convo = list(messages)
    decision = _normalize_decision_shape(
        _extract_json(llm.chat(convo, temperature=llm_temperature))
    )

    tool_data: dict[str, Any] = {}
    tool_rounds: list[dict[str, Any]] = []

    if (
        not decision.get("web_search_query")
        and not decision.get("web_fetch_url")
        and not _has_action(decision)
    ):
        bootstrap_url = _extract_bootstrap_url(persona)
        if bootstrap_url:
            bootstrap_requested = {"web_fetch_url": bootstrap_url, "bootstrap": True}
            try:
                tool_data["page"] = web.fetch_page_text(bootstrap_url)
            except Exception as e:
                tool_data["page_error"] = str(e)
            bootstrap_result = {
                k: tool_data[k] for k in ("page", "page_error") if k in tool_data
            }
            tool_rounds.append(
                {"requested": bootstrap_requested, "result": bootstrap_result}
            )
            convo.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(decision, ensure_ascii=True),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "tool_data": bootstrap_result,
                                "instruction": (
                                    "You now have fetched source content from persona guidance. "
                                    "Return final action JSON now (no further tools unless strictly needed)."
                                ),
                                "post_char_limit": post_char_limit,
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
            )
            decision = _normalize_decision_shape(
                _extract_json(llm.chat(convo, temperature=llm_temperature)) or decision
            )

    for _ in range(3):
        requested_tools: dict[str, Any] = {}
        if decision.get("web_search_query"):
            requested_tools["web_search_query"] = str(decision["web_search_query"])
            try:
                tool_data["search"] = web.keyword_search(
                    requested_tools["web_search_query"]
                )
            except Exception as e:
                tool_data["search_error"] = str(e)

        if decision.get("web_fetch_url"):
            requested_tools["web_fetch_url"] = str(decision["web_fetch_url"])
            try:
                tool_data["page"] = web.fetch_page_text(
                    requested_tools["web_fetch_url"]
                )
            except Exception as e:
                tool_data["page_error"] = str(e)

        if not requested_tools:
            break

        tool_rounds.append(
            {
                "requested": requested_tools,
                "result": {
                    k: tool_data[k]
                    for k in ("search", "search_error", "page", "page_error")
                    if k in tool_data
                },
            }
        )

        convo.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(decision, ensure_ascii=True),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "tool_data": tool_rounds[-1]["result"],
                            "instruction": (
                                "Use this tool output. If still needed, you may request another tool once. "
                                "Otherwise return final action JSON now."
                            ),
                            "post_char_limit": post_char_limit,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
        )

        decision = _normalize_decision_shape(
            _extract_json(llm.chat(convo, temperature=llm_temperature)) or decision
        )

    if not _has_action(decision):
        convo.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(decision, ensure_ascii=True),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": (
                                "Now return action JSON only. Include `action` or `actions`. "
                                "Do not request more tools in this response."
                            ),
                            "post_char_limit": post_char_limit,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
        )
        decision = _normalize_decision_shape(
            _extract_json(llm.chat(convo, temperature=llm_temperature)) or decision
        )

    if not _has_action(decision) and tool_data:
        convo.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(decision, ensure_ascii=True),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": (
                                "Return final action JSON now. Choose one action from post/reply/follow/dm/favourite/boost/noop. "
                                "Do not request any more tools."
                            ),
                            "tool_data": {
                                k: tool_data[k]
                                for k in (
                                    "search",
                                    "page",
                                    "search_error",
                                    "page_error",
                                )
                                if k in tool_data
                            },
                            "post_char_limit": post_char_limit,
                        },
                        ensure_ascii=True,
                    ),
                },
            ]
        )
        decision = _normalize_decision_shape(
            _extract_json(llm.chat(convo, temperature=llm_temperature)) or decision
        )

    if not _has_action(decision):
        repair_payload = {
            "instruction": (
                "Repair the prior output into strict final action JSON. "
                "Return exactly one object with either `action` or `actions`. "
                "Allowed actions: post/reply/follow/dm/favourite/boost/noop. "
                "No tool requests."
            ),
            "prior_output": decision,
            "post_char_limit": post_char_limit,
            "remaining_quota": remaining_quota,
            "tool_data": {
                k: tool_data[k]
                for k in ("search", "page", "search_error", "page_error")
                if k in tool_data
            },
            "has_mention": first_mention is not None,
        }
        repair_messages = [
            {
                "role": "system",
                "content": "You repair malformed planner outputs into valid final action JSON.",
            },
            {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=True)},
        ]
        repaired = _normalize_decision_shape(
            _extract_json(
                llm.chat(repair_messages, temperature=min(llm_temperature, 0.25))
            )
        )
        if _has_action(repaired) or _is_noop_action(repaired):
            decision = repaired

    if not _has_action(decision):
        last_chance_messages = [
            {
                "role": "system",
                "content": (
                    "Return strict JSON only with either `action` or `actions`. "
                    "Allowed actions: post, reply, follow, dm, favourite, boost, noop. "
                    "No tool requests in this response."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "post_char_limit": post_char_limit,
                        "remaining_quota": remaining_quota,
                        "has_mention": first_mention is not None,
                        "tool_data": {
                            k: tool_data[k]
                            for k in ("search", "page", "search_error", "page_error")
                            if k in tool_data
                        },
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        last_chance = _normalize_decision_shape(
            _extract_json(llm.chat(last_chance_messages, temperature=0.1))
        )
        if _has_action(last_chance) or _is_noop_action(last_chance):
            decision = last_chance

    if not _has_action(decision):
        decision = {"action": "noop", "reason": "invalid_decision_shape"}

    if _is_noop_action(decision) and tool_rounds:
        has_search_results = (
            isinstance(tool_data.get("search"), list)
            and len(tool_data.get("search", [])) > 0
        )
        if has_search_results:
            preferred_action = "post" if first_mention is None else "reply"
            convo.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(decision, ensure_ascii=True),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "instruction": (
                                    f"You already have search evidence. Produce a concrete action now (prefer {preferred_action}), "
                                    "not noop, unless all sources are unusable."
                                ),
                                "post_char_limit": post_char_limit,
                                "tool_data": {
                                    k: tool_data[k]
                                    for k in (
                                        "search",
                                        "page",
                                        "search_error",
                                        "page_error",
                                    )
                                    if k in tool_data
                                },
                            },
                            ensure_ascii=True,
                        ),
                    },
                ]
            )
            decision = _normalize_decision_shape(
                _extract_json(llm.chat(convo, temperature=llm_temperature)) or decision
            )

    result: dict[str, Any] = {
        "decision": decision,
        "tool_data": tool_data,
        "read_scope": {
            "mentions": len(notifications),
            "home": len(home),
            "topic_posts": len(topic_posts),
            "own_posts": len(own_posts),
        },
        "notifications_marked_read": dismissed_mentions,
    }
    if tool_rounds:
        result["tool_data_rounds"] = tool_rounds

    actions = _parse_actions(decision)

    executed: list[dict[str, Any]] = []
    source_url = _first_source_url(tool_data)
    thread_url_cache: dict[str, set[str]] = {}

    for action_obj in actions[: max(1, max_actions)]:
        action = _normalized_action(action_obj.get("action"))

        def thread_seen_urls(status_id: str) -> set[str]:
            sid = str(status_id or "").strip()
            if not sid:
                return set()
            cached = thread_url_cache.get(sid)
            if cached is not None:
                return cached

            seen_urls: set[str] = set()
            try:
                status = agent.status(sid)
                for url in _status_urls(status):
                    seen_urls.add(url)
            except Exception:
                pass

            try:
                ctx = agent.status_context(sid)
                ancestors = ctx.get("ancestors") if isinstance(ctx, dict) else []
                if isinstance(ancestors, list):
                    for item in ancestors:
                        if not isinstance(item, dict):
                            continue
                        for url in _status_urls(item):
                            seen_urls.add(url)
            except Exception:
                pass

            thread_url_cache[sid] = seen_urls
            return seen_urls

        def shorten_text_preserve_urls(value: str, target: int) -> str:
            if len(value) <= target:
                return value
            if target <= 1:
                return value[:target]

            def shorten_plain(text: str) -> str:
                if len(text) <= target:
                    return text
                if target <= 1:
                    return text[:target]
                return text[: target - 1].rstrip() + "…"

            urls = [m.group(0).strip() for m in URL_PATTERN.finditer(value)]
            if not urls:
                return shorten_plain(value)

            first_url = urls[0]
            if len(first_url) > target:
                no_url = URL_PATTERN.sub("", value)
                no_url = re.sub(r"\s+", " ", no_url).strip()
                return shorten_plain(no_url)

            prose = URL_PATTERN.sub("", value)
            prose = re.sub(r"\s+", " ", prose).strip()
            reserve = len(first_url) + (1 if prose else 0)
            available = target - reserve
            if available < 0:
                return shorten_plain(prose)

            if len(prose) > available:
                if available <= 1:
                    prose = ""
                else:
                    prose = prose[: available - 1].rstrip() + "…"

            out = f"{prose} {first_url}".strip() if prose else first_url
            if len(out) <= target:
                return out

            return shorten_plain(prose)

        def fit_text(value: str) -> str:
            return shorten_text_preserve_urls(value, post_char_limit)

        def shrink_more(value: str) -> str:
            soft = max(40, int(post_char_limit * 0.85))
            return shorten_text_preserve_urls(value, soft)

        def apply_source_policy(
            value: str, blocked_urls: set[str] | None = None
        ) -> str:
            candidate = _sanitize_text_urls(value)
            blocked = blocked_urls or set()
            if blocked and _has_url(candidate):
                for match in URL_PATTERN.finditer(candidate):
                    normalized = _normalize_url(match.group(0).strip().rstrip(".,);]"))
                    if normalized in blocked:
                        candidate = _sanitize_text_urls(URL_PATTERN.sub("", candidate))
                        break
            if _needs_source_for_claim(candidate) and not _has_url(candidate):
                if source_url and source_url not in blocked:
                    candidate = f"{candidate} {source_url}".strip()
            return candidate

        def is_too_long_error(error: Exception) -> bool:
            if not isinstance(error, httpx.HTTPStatusError):
                return False
            body = (
                error.response.text if error.response is not None else str(error)
            ).lower()
            return "too long" in body or "validation failed" in body and "text" in body

        if action == "post" and action_obj.get("text"):
            candidate = fit_text(apply_source_policy(str(action_obj["text"])))
            if not candidate:
                executed.append(
                    {
                        "action": "noop",
                        "reason": "post text cannot fit limit without breaking URL",
                    }
                )
                continue
            try:
                executed.append({"action": "post", "result": agent.post(candidate)})
            except Exception as error:
                if is_too_long_error(error):
                    shorter = shrink_more(candidate)
                    if shorter != candidate:
                        try:
                            executed.append(
                                {
                                    "action": "post",
                                    "result": agent.post(shorter),
                                    "note": "retried with shorter text",
                                }
                            )
                            continue
                        except Exception as retry_error:
                            error = retry_error
                executed.append(
                    {
                        "action": "noop",
                        "reason": "post failed",
                        "error": _api_error(error),
                    }
                )
                continue
            continue

        if (
            action == "reply"
            and action_obj.get("text")
            and action_obj.get("in_reply_to_id")
        ):
            target_id = str(action_obj["in_reply_to_id"])
            blocked_urls = thread_seen_urls(target_id)
            reply_text = fit_text(
                apply_source_policy(str(action_obj["text"]), blocked_urls=blocked_urls)
            )
            if not reply_text:
                executed.append(
                    {
                        "action": "noop",
                        "reason": "reply text cannot fit limit without breaking URL",
                    }
                )
                continue
            try:
                executed.append(
                    {
                        "action": "reply",
                        "result": agent.post(reply_text, in_reply_to_id=target_id),
                    }
                )
            except Exception as error:
                if is_too_long_error(error):
                    shorter = shrink_more(reply_text)
                    try:
                        executed.append(
                            {
                                "action": "reply",
                                "result": agent.post(shorter, in_reply_to_id=target_id),
                                "note": "retried with shorter text",
                            }
                        )
                        continue
                    except Exception as retry_error:
                        error = retry_error
                executed.append(
                    {
                        "action": "noop",
                        "reason": "reply failed",
                        "error": _api_error(error),
                    }
                )
            continue

        if action in {"favourite", "boost"}:
            status_id = str(
                action_obj.get("status_id")
                or action_obj.get("in_reply_to_id")
                or action_obj.get("id")
                or ""
            ).strip()
            if not status_id:
                executed.append(
                    {
                        "action": "noop",
                        "reason": "missing status_id for engagement action",
                    }
                )
                continue
            try:
                if action == "favourite":
                    executed.append(
                        {"action": "favourite", "result": agent.favourite(status_id)}
                    )
                else:
                    executed.append(
                        {"action": "boost", "result": agent.boost(status_id)}
                    )
            except Exception as error:
                executed.append(
                    {
                        "action": "noop",
                        "reason": f"{action} failed",
                        "error": _api_error(error),
                    }
                )
            continue

        if action == "follow" and action_obj.get("target_acct"):
            try:
                executed.append(
                    {
                        "action": "follow",
                        "result": agent.follow(str(action_obj["target_acct"])),
                    }
                )
            except Exception as error:
                executed.append(
                    {
                        "action": "noop",
                        "reason": "follow failed",
                        "error": _api_error(error),
                    }
                )
            continue

        if action == "dm" and action_obj.get("target_acct") and action_obj.get("text"):
            dm_text = fit_text(apply_source_policy(str(action_obj["text"])))
            if not dm_text:
                executed.append(
                    {
                        "action": "noop",
                        "reason": "dm text cannot fit limit without breaking URL",
                    }
                )
                continue
            try:
                executed.append(
                    {
                        "action": "dm",
                        "result": agent.dm(str(action_obj["target_acct"]), dm_text),
                    }
                )
            except Exception as error:
                if is_too_long_error(error):
                    shorter = shrink_more(dm_text)
                    try:
                        executed.append(
                            {
                                "action": "dm",
                                "result": agent.dm(
                                    str(action_obj["target_acct"]), shorter
                                ),
                                "note": "retried with shorter text",
                            }
                        )
                        continue
                    except Exception as retry_error:
                        error = retry_error
                executed.append(
                    {
                        "action": "noop",
                        "reason": "dm failed",
                        "error": _api_error(error),
                    }
                )
            continue

        executed.append(
            {
                "action": "noop",
                "reason": action_obj.get("reason") or "no valid action args",
            }
        )

    result["execution"] = executed
    result["post_char_limit"] = post_char_limit

    result["status"] = status_snapshot
    return result


def main() -> None:
    load_dotenv()
    args = parse_args()

    allow_insecure_tls = (
        args.allow_insecure_tls
        or os.getenv("ALLOW_INSECURE_TLS", "true").lower() == "true"
    )

    registry = AgentRegistry(Path(args.registry_file))
    agent_record = registry.get_agent(args.agent)
    if not agent_record:
        raise SystemExit(f"Agent '{args.agent}' not found in local registry")

    persona = str(agent_record.get("persona") or "")
    if not persona:
        raise SystemExit(
            "Agent has no persona in registry. Create/update via agent_cli create-llm/register commands."
        )

    email = str(agent_record.get("email") or "")
    password = str(agent_record.get("password") or "")
    if not email or not password:
        raise SystemExit("Agent record missing email/password")

    onboard = OnboardClient(
        base_url=args.base_url,
        allow_insecure_tls=allow_insecure_tls,
        app_cache_file=Path(args.app_cache),
    )
    app = onboard.ensure_app()
    token = onboard.password_grant_token(app, email, password, "read write follow")
    onboard.close()

    llm_cfg_raw = agent_record.get("llm")
    llm_cfg: dict[str, Any] = llm_cfg_raw if isinstance(llm_cfg_raw, dict) else {}
    mode = str(llm_cfg.get("mode") or os.getenv("OLLAMA_MODE", "cloud"))
    model = str(llm_cfg.get("model") or os.getenv("OLLAMA_MODEL", "gpt-oss:120b"))
    base = str(
        llm_cfg.get("base_url")
        or os.getenv("OLLAMA_BASE_URL")
        or ("http://127.0.0.1:11434" if mode == "local" else "https://ollama.com")
    )
    api_key = str(llm_cfg.get("api_key") or os.getenv("OLLAMA_API_KEY", ""))
    max_tokens_raw = llm_cfg.get("max_tokens") or os.getenv(
        "OLLAMA_MAX_TOKENS", "65536"
    )
    max_ctx_raw = llm_cfg.get("max_context_tokens") or os.getenv(
        "OLLAMA_MAX_CONTEXT_TOKENS", "131072"
    )
    temperature_raw = llm_cfg.get("temperature") or os.getenv(
        "OLLAMA_TEMPERATURE", "0.9"
    )
    try:
        max_tokens = int(str(max_tokens_raw))
    except Exception:
        max_tokens = 65536
    try:
        max_ctx = int(str(max_ctx_raw))
    except Exception:
        max_ctx = 131072
    try:
        llm_temperature = float(str(temperature_raw))
    except Exception:
        llm_temperature = 0.9

    agent = AgentClient(
        base_url=args.base_url, token=token, allow_insecure_tls=allow_insecure_tls
    )
    llm = OllamaClient(
        mode=mode,
        model=model,
        base_url=base,
        api_key=api_key,
        max_output_tokens=max_tokens,
        max_context_tokens=max_ctx,
        allow_insecure_tls=allow_insecure_tls,
    )
    web = WebTools(allow_insecure_tls=allow_insecure_tls)

    try:
        while True:
            try:
                payload = run_cycle(
                    agent,
                    llm,
                    web,
                    persona,
                    llm_temperature=llm_temperature,
                    max_actions=args.max_actions,
                    post_cooldown_minutes=args.post_cooldown_minutes,
                )
            except Exception as error:
                payload = {
                    "decision": {
                        "actions": [{"action": "noop", "reason": "cycle failed"}]
                    },
                    "error": _api_error(error),
                }
            print(json.dumps(payload, indent=2))
            if args.once:
                break
            time.sleep(max(15, args.interval_seconds))
    finally:
        agent.close()
        llm.close()
        web.close()


if __name__ == "__main__":
    main()
