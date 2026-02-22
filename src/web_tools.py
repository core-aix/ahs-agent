from __future__ import annotations

import html
import re
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


@dataclass
class WebTools:
    allow_insecure_tls: bool = False

    def __post_init__(self) -> None:
        self.http = httpx.Client(timeout=20.0, verify=not self.allow_insecure_tls)

    def close(self) -> None:
        self.http.close()

    def _search_duckduckgo(self, query: str, limit: int) -> list[dict[str, str]]:
        url = f"https://duckduckgo.com/html/?q={quote(query)}"
        resp = self.http.get(url, follow_redirects=True)
        resp.raise_for_status()
        body = resp.text

        matches = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
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
        url = f"https://www.bing.com/search?q={quote(query)}"
        resp = self.http.get(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        body = resp.text

        matches = re.findall(
            r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>.*?<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )

        results: list[dict[str, str]] = []
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
        text = _strip_tags(resp.text)
        return {
            "url": str(resp.url),
            "status": resp.status_code,
            "content": text[:max_chars],
        }
