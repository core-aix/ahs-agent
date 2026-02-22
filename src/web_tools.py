from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


def _strip_tags(raw_html: str) -> str:
    text = re.sub(r"<script.*?</script>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if "}" in tag else tag.lower()


def _looks_like_feed(url: str, content_type: str) -> bool:
    ct = (content_type or "").lower()
    u = url.lower()
    if any(token in ct for token in ("application/rss+xml", "application/atom+xml")):
        return True
    if "xml" in ct:
        return True
    return any(token in u for token in ("/rss", ".rss", ".xml", "atom"))


def _extract_feed_text(raw_xml: str, max_items: int = 12) -> tuple[str, int]:
    root = ET.fromstring(raw_xml)
    root_name = _tag_name(root.tag)
    lines: list[str] = []

    def text_of(el: ET.Element | None) -> str:
        if el is None:
            return ""
        return " ".join("".join(el.itertext()).split()).strip()

    def child(el: ET.Element, name: str) -> ET.Element | None:
        for c in list(el):
            if _tag_name(c.tag) == name:
                return c
        return None

    feed_title = ""
    items: list[dict[str, str]] = []

    if root_name == "rss":
        channel = child(root, "channel")
        if channel is not None:
            feed_title = text_of(child(channel, "title"))
            for it in [c for c in list(channel) if _tag_name(c.tag) == "item"][
                :max_items
            ]:
                items.append(
                    {
                        "title": text_of(child(it, "title")),
                        "link": text_of(child(it, "link")),
                        "date": text_of(child(it, "pubdate")),
                        "summary": text_of(child(it, "description")),
                    }
                )
    elif root_name == "feed":
        feed_title = text_of(child(root, "title"))
        for entry in [c for c in list(root) if _tag_name(c.tag) == "entry"][:max_items]:
            link = ""
            for lc in list(entry):
                if _tag_name(lc.tag) != "link":
                    continue
                rel = (lc.attrib.get("rel") or "alternate").lower()
                href = (lc.attrib.get("href") or "").strip()
                if rel == "alternate" and href:
                    link = href
                    break
                if not link and href:
                    link = href
            items.append(
                {
                    "title": text_of(child(entry, "title")),
                    "link": link,
                    "date": text_of(child(entry, "updated"))
                    or text_of(child(entry, "published")),
                    "summary": text_of(child(entry, "summary"))
                    or text_of(child(entry, "content")),
                }
            )

    if feed_title:
        lines.append(f"Feed: {feed_title}")

    for idx, item in enumerate(items, 1):
        title = item.get("title") or "(untitled)"
        date = item.get("date") or ""
        link = item.get("link") or ""
        summary = item.get("summary") or ""
        summary = _strip_tags(summary) if summary else ""
        parts = [f"{idx}. {title}"]
        if date:
            parts.append(f"[{date}]")
        if link:
            parts.append(link)
        lines.append(" ".join(parts))
        if summary:
            lines.append(f"   {summary[:280]}")

    return "\n".join(lines).strip(), len(items)


@dataclass
class WebTools:
    allow_insecure_tls: bool = False

    def __post_init__(self) -> None:
        self.user_agent = "aihuman-agent/1.0 (contact: admin@ai-human.social)"
        self.http = httpx.Client(
            timeout=20.0,
            verify=not self.allow_insecure_tls,
            headers={"User-Agent": self.user_agent},
        )

    def close(self) -> None:
        self.http.close()

    def _search_duckduckgo(self, query: str, limit: int) -> list[dict[str, str]]:
        url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
        resp = self.http.get(url, follow_redirects=True)
        resp.raise_for_status()
        body = resp.text

        matches = re.findall(
            r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results: list[dict[str, str]] = []
        for href, title_html in matches[: max(1, min(limit, 10))]:
            title = _strip_tags(title_html)
            results.append(
                {"engine": "duckduckgo", "title": title, "url": html.unescape(href)}
            )
        return results

    def _search_bing(self, query: str, limit: int) -> list[dict[str, str]]:
        rss_url = f"https://www.bing.com/news/search?q={quote(query)}&format=rss"
        rss_resp = self.http.get(rss_url, follow_redirects=True)
        rss_resp.raise_for_status()

        results: list[dict[str, str]] = []
        try:
            root = ET.fromstring(rss_resp.text)
            for item in root.findall("./channel/item")[: max(1, min(limit, 10))]:
                title = " ".join(
                    "".join((item.findtext("title") or "")).split()
                ).strip()
                link = " ".join("".join((item.findtext("link") or "")).split()).strip()
                if not title or not link:
                    continue
                results.append({"engine": "bing", "title": title, "url": link})
        except Exception:
            results = []

        if results:
            return results

        url = f"https://www.bing.com/search?q={quote(query)}"
        resp = self.http.get(url, follow_redirects=True)
        resp.raise_for_status()
        body = resp.text

        matches = re.findall(
            r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )

        for href, title_html in matches[: max(1, min(limit, 10))]:
            title = _strip_tags(title_html)
            if not title:
                continue
            results.append(
                {"engine": "bing", "title": title, "url": html.unescape(href)}
            )
        return results

    def _search_wikipedia(self, query: str, limit: int) -> list[dict[str, str]]:
        url = "https://en.wikipedia.org/w/api.php"
        resp = self.http.get(
            url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max(1, min(limit, 10)),
                "format": "json",
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        payload = resp.json()
        search_items = (payload.get("query") or {}).get("search") or []
        out: list[dict[str, str]] = []
        for item in search_items:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            out.append(
                {
                    "engine": "wikipedia",
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                }
            )
        return out

    def keyword_search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        target = max(1, min(limit, 10))
        results: list[dict[str, str]] = []

        try:
            results.extend(self._search_bing(query, target))
        except Exception:
            pass

        try:
            results.extend(self._search_duckduckgo(query, target))
        except Exception:
            pass

        try:
            results.extend(self._search_wikipedia(query, target))
        except Exception:
            pass

        dedup: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in results:
            url = str(item.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            dedup.append(item)
            if len(dedup) >= target:
                break
        return dedup

    def fetch_page_text(self, url: str, max_chars: int = 3000) -> dict[str, Any]:
        resp = self.http.get(url, follow_redirects=True)
        resp.raise_for_status()
        content_type = str(resp.headers.get("content-type") or "")
        body = resp.text

        kind = "page"
        entry_count = 0
        text = ""
        if _looks_like_feed(str(resp.url), content_type):
            try:
                parsed, entry_count = _extract_feed_text(body)
                if parsed:
                    text = parsed
                    kind = "feed"
            except Exception:
                text = ""

        if (
            kind == "feed"
            and entry_count == 0
            and "rss.arxiv.org/rss/" in str(resp.url)
        ):
            try:
                category = str(resp.url).rstrip("/").rsplit("/", 1)[-1]
                api_resp = self.http.get(
                    "https://export.arxiv.org/api/query",
                    params={
                        "search_query": f"cat:{category}",
                        "sortBy": "submittedDate",
                        "sortOrder": "descending",
                        "max_results": "8",
                    },
                    follow_redirects=True,
                )
                api_resp.raise_for_status()
                parsed, api_count = _extract_feed_text(api_resp.text)
                if parsed and api_count > 0:
                    text = parsed
                    kind = "feed_arxiv_api_fallback"
                    entry_count = api_count
            except Exception:
                pass

        if not text:
            text = _strip_tags(body)

        return {
            "url": str(resp.url),
            "status": resp.status_code,
            "kind": kind,
            "entry_count": entry_count,
            "content": text[:max_chars],
        }
