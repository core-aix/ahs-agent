from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_cli import OnboardClient
from client import AgentClient
from ollama_client import OllamaClient
from registry import AgentRegistry
from web_tools import WebTools


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


def _build_system_prompt(
    persona: str, self_id: str, self_acct: str, has_mentions: bool
) -> str:
    mention_rule = (
        "You have fresh mention notifications. Prioritize replying to mentions first. "
        if has_mentions
        else ""
    )
    return (
        "You are an autonomous social AI agent on Mastodon. "
        "Be active and helpful, but concise and safe. "
        "You may choose multiple actions each cycle if useful, or noop when there is nothing new. "
        "Prefer reply when there is an interesting post to engage with. "
        "If you posted recently, avoid posting again unless there is genuinely new value. "
        "Only follow or DM when it is contextually meaningful. "
        "Never reply to your own posts. Never target your own account in follow/dm. "
        "You may request web_search_query and web_fetch_url to gather context before deciding. "
        "If you need up-to-date or uncertain factual information, do not speculate: immediately request web_search_query first, then decide from evidence. "
        f"{mention_rule}"
        "Return strict JSON only.\n\n"
        f"Your identity: account_id={self_id}, acct={self_acct}\n"
        f"Persona and interests:\n{persona.strip()}"
    )


def _build_decision_prompt(
    home: list[Any], notifications: list[Any], self_id: str, self_acct: str
) -> str:
    sample_home = []
    for item in home[:40]:
        account = item.get("account", {}) if isinstance(item, dict) else {}
        author_id = str(account.get("id") or "")
        sample_home.append(
            {
                "id": item.get("id"),
                "account_id": author_id,
                "acct": account.get("acct"),
                "author_is_self": author_id == self_id,
                "content": _strip_html(str(item.get("content", "")))[:280],
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
                "from": n.get("account", {}).get("acct"),
                "text": _strip_html(str((n.get("status") or {}).get("content", "")))[
                    :220
                ],
            }
        )

    return json.dumps(
        {
            "task": "Choose one action for this cycle.",
            "schema": {
                "action": "post|reply|follow|dm|noop",
                "text": "required for post/reply/dm",
                "in_reply_to_id": "required for reply",
                "target_acct": "required for follow and dm, e.g. alice@example.social",
                "web_search_query": "optional",
                "web_fetch_url": "optional",
                "reason": "short rationale",
            },
            "multi_action_schema": {
                "actions": [
                    {
                        "action": "post|reply|follow|dm|noop",
                        "text": "optional",
                        "in_reply_to_id": "optional",
                        "target_acct": "optional",
                        "reason": "optional",
                    }
                ]
            },
            "self": {"account_id": self_id, "acct": self_acct},
            "timeline": sample_home,
            "notifications": sample_notifs,
        },
        ensure_ascii=True,
    )


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


def _build_mention_reply(
    llm: OllamaClient, persona: str, self_acct: str, status: dict[str, Any]
) -> str:
    author = (status.get("account") or {}).get("acct") or ""
    text = _strip_html(str(status.get("content", "")))[:320]
    prompt = (
        "Write one short Mastodon reply (max 280 chars). Keep it friendly, useful, and in persona. "
        "Do not mention yourself. Return plain text only, no JSON.\n"
        f"Your acct: {self_acct}\n"
        f"Persona: {persona}\n"
        f"Incoming mention from @{author}: {text}"
    )
    out = llm.chat(
        [
            {"role": "system", "content": "You write concise social replies."},
            {"role": "user", "content": prompt},
        ]
    )
    out = out.strip().replace("\n", " ")
    return out[:280] if out else "Thanks for the mention!"


def _parse_actions(decision: dict[str, Any]) -> list[dict[str, Any]]:
    raw = decision.get("actions")
    if isinstance(raw, list):
        actions = [a for a in raw if isinstance(a, dict)]
        if actions:
            return actions
    return [decision]


def _recent_self_post_exists(
    home: list[Any], self_id: str, cooldown_minutes: int
) -> bool:
    if not self_id:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, cooldown_minutes))
    for item in home:
        if not isinstance(item, dict):
            continue
        author_id = str(((item.get("account") or {}).get("id") or ""))
        if author_id != self_id:
            continue
        created_raw = str(item.get("created_at") or "")
        if not created_raw:
            continue
        try:
            created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if created >= cutoff:
            return True
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


def run_cycle(
    agent: AgentClient,
    llm: OllamaClient,
    web: WebTools,
    persona: str,
    *,
    max_actions: int,
    post_cooldown_minutes: int,
) -> dict[str, Any]:
    me = agent.verify_credentials()
    self_id = str(me.get("id") or "")
    self_acct = str(me.get("acct") or me.get("username") or "")

    home = agent.read_home(limit=20)
    notifications = agent.mention_notifications_all(page_limit=80)
    trends = agent.trends_statuses(limit=40)
    topic_posts = _select_topic_posts(
        trends, persona, self_id, min_count=10, max_count=20
    )
    prompt_home = topic_posts + home
    first_mention = _first_mention_status(notifications, self_id)

    messages = [
        {
            "role": "system",
            "content": _build_system_prompt(
                persona, self_id, self_acct, first_mention is not None
            ),
        },
        {
            "role": "user",
            "content": _build_decision_prompt(
                prompt_home, notifications, self_id, self_acct
            ),
        },
    ]

    initial = llm.chat(messages)
    decision = _extract_json(initial)

    tool_data: dict[str, Any] = {}
    if decision.get("web_search_query"):
        try:
            tool_data["search"] = web.keyword_search(str(decision["web_search_query"]))
        except Exception as e:
            tool_data["search_error"] = str(e)

    if decision.get("web_fetch_url"):
        try:
            tool_data["page"] = web.fetch_page_text(str(decision["web_fetch_url"]))
        except Exception as e:
            tool_data["page_error"] = str(e)

    if tool_data:
        followup_messages = messages + [
            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=True)},
            {
                "role": "user",
                "content": json.dumps({"tool_data": tool_data}, ensure_ascii=True),
            },
        ]
        decision = _extract_json(llm.chat(followup_messages)) or decision

    result: dict[str, Any] = {
        "decision": decision,
        "tool_data": tool_data,
        "read_scope": {
            "mentions": len(notifications),
            "home": len(home),
            "topic_posts": len(topic_posts),
        },
    }

    status_by_id: dict[str, dict[str, Any]] = {}
    for item in home:
        if isinstance(item, dict) and item.get("id"):
            status_by_id[str(item["id"])] = item

    actions = _parse_actions(decision)

    if first_mention is not None and not any(
        str(a.get("action", "noop")) == "reply" for a in actions
    ):
        reply_text = _build_mention_reply(llm, persona, self_acct, first_mention)
        mention_action = {
            "action": "reply",
            "text": reply_text,
            "in_reply_to_id": first_mention.get("id"),
            "reason": "prioritize mention response",
        }
        actions = [mention_action] + actions
        result["decision"] = {"actions": actions, "reason": "mention prioritized"}

    recent_post = _recent_self_post_exists(home, self_id, post_cooldown_minutes)
    executed: list[dict[str, Any]] = []

    for action_obj in actions[: max(1, max_actions)]:
        action = str(action_obj.get("action", "noop"))

        if action == "post" and action_obj.get("text"):
            if recent_post:
                executed.append(
                    {"action": "noop", "reason": "recent self post cooldown"}
                )
                continue
            executed.append(
                {"action": "post", "result": agent.post(str(action_obj["text"]))}
            )
            recent_post = True
            continue

        if (
            action == "reply"
            and action_obj.get("text")
            and action_obj.get("in_reply_to_id")
        ):
            target_id = str(action_obj["in_reply_to_id"])
            target_status = status_by_id.get(target_id)
            target_author_id = str(
                ((target_status or {}).get("account") or {}).get("id") or ""
            )
            if self_id and target_author_id == self_id:
                executed.append({"action": "noop", "reason": "blocked self-reply"})
                continue
            executed.append(
                {
                    "action": "reply",
                    "result": agent.post(
                        str(action_obj["text"]), in_reply_to_id=target_id
                    ),
                }
            )
            continue

        if action == "follow" and action_obj.get("target_acct"):
            target = str(action_obj["target_acct"]).lstrip("@")
            if target == self_acct.lstrip("@"):
                executed.append({"action": "noop", "reason": "blocked self-follow"})
                continue
            executed.append(
                {
                    "action": "follow",
                    "result": agent.follow(str(action_obj["target_acct"])),
                }
            )
            continue

        if action == "dm" and action_obj.get("target_acct") and action_obj.get("text"):
            target = str(action_obj["target_acct"]).lstrip("@")
            if target == self_acct.lstrip("@"):
                executed.append({"action": "noop", "reason": "blocked self-dm"})
                continue
            executed.append(
                {
                    "action": "dm",
                    "result": agent.dm(
                        str(action_obj["target_acct"]), str(action_obj["text"])
                    ),
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

    result["status"] = agent.agent_status()
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

    agent = AgentClient(
        base_url=args.base_url, token=token, allow_insecure_tls=allow_insecure_tls
    )
    llm = OllamaClient(
        mode=mode,
        model=model,
        base_url=base,
        api_key=api_key,
        allow_insecure_tls=allow_insecure_tls,
    )
    web = WebTools(allow_insecure_tls=allow_insecure_tls)

    try:
        while True:
            payload = run_cycle(
                agent,
                llm,
                web,
                persona,
                max_actions=args.max_actions,
                post_cooldown_minutes=args.post_cooldown_minutes,
            )
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
