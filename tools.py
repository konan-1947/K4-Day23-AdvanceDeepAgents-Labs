"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

# Rate limiting and concurrency locks
_arxiv_lock = threading.Lock()
_last_arxiv_time = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def _redact_key(text: str) -> str:
    """Che giấu EXA_API_KEY và các khóa bí mật trong thông báo lỗi."""
    key = (os.getenv("EXA_API_KEY") or "").strip()
    if key and key in text:
        text = text.replace(key, "[REDACTED]")
    return text


def _retry_after(response, default: float | None) -> float | None:
    """Return a numeric Retry-After value, falling back for missing or malformed headers."""
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.HTTPStatusError, httpx.TransportError) as exc:
            # Chỉ retry các mã HTTP tạm thời: 429, 500, 502, 503, 504
            if isinstance(exc, httpx.HTTPStatusError):
                if exc.response.status_code not in {429, 500, 502, 503, 504}:
                    raise

            # Lần thử cuối thất bại: ném lại lỗi, không ngủ thêm
            if attempt == attempts - 1:
                raise

            delay = None
            if isinstance(exc, RetryableError) and exc.retry_after is not None:
                try:
                    delay = float(exc.retry_after)
                except (ValueError, TypeError):
                    delay = None
            elif isinstance(exc, httpx.HTTPStatusError):
                delay = _retry_after(exc.response, default=None)

            if delay is None:
                # Backoff lũy thừa + jitter ngẫu nhiên
                delay = base * (2 ** attempt) + random.uniform(0.1, 1.0)

            # Chặn trên bằng cap
            delay = min(delay, cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _last_arxiv_time
    try:
        # Làm sạch truy vấn: chỉ giữ ký tự chữ/số và gạch nối
        terms = re.findall(r"[A-Za-z0-9\-]+", query or "")
        if not terms:
            return "NO RESULTS"

        # Tôn trọng giới hạn của arXiv: ít nhất 3 giây giữa hai lần gọi
        with _arxiv_lock:
            elapsed = time.monotonic() - _last_arxiv_time
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            _last_arxiv_time = time.monotonic()

        clamped_max = max(1, min(int(max_results), 30))
        search_query = " AND ".join(f"all:{t}" for t in terms)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def _fetch():
            resp = httpx.get(ARXIV_URL, params=params, timeout=30.0, follow_redirects=True)
            if resp.status_code == 429:
                raise RetryableError(
                    "arXiv HTTP 429 rate limit",
                    retry_after=_retry_after(resp, 10.0),
                )
            resp.raise_for_status()
            return resp

        # Cho arXiv cap 60s để vượt qua rate limit tập thể
        resp = with_retry(_fetch, attempts=5, base=2.0, cap=60.0)

        # Parse Atom XML
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = xml.etree.ElementTree.fromstring(resp.text)
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            if id_elem is None or not id_elem.text:
                continue
            raw_id = id_elem.text.strip().split("/abs/")[-1]
            clean_id = re.sub(r"v\d+$", "", raw_id)
            url = f"https://arxiv.org/abs/{clean_id}"

            pub_elem = entry.find("atom:published", ns)
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title_elem = entry.find("atom:title", ns)
            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"

            summary_elem = entry.find("atom:summary", ns)
            summary = " ".join(summary_elem.text.split())[:600] if summary_elem is not None and summary_elem.text else ""

            records.append({
                "id": clean_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(int(limit), 100))
        params = {"limit": clamped_limit}
        if date and date.strip():
            params["date"] = date.strip()

        def _fetch():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=30.0, follow_redirects=True)
            if resp.status_code == 429:
                raise RetryableError("HF HTTP 429 rate limit", retry_after=_retry_after(resp, 5.0))
            resp.raise_for_status()
            return resp.json()

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
            paper_id = paper.get("id") or item.get("id")
            if not paper_id:
                continue

            paper_id = str(paper_id).strip()
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = " ".join(str(paper.get("title") or item.get("title") or "").split()) or "Untitled"
            summary = " ".join(str(paper.get("summary") or item.get("summary") or "").split())[:600]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = paper.get("githubRepo") or item.get("githubRepo")
            stars = int(paper.get("githubStars") or item.get("githubStars") or 0)

            # Lọc theo keyword nếu có
            if keyword and keyword.strip():
                kw = keyword.strip().lower()
                if kw not in (title + " " + summary).lower():
                    continue

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        # Sắp xếp theo upvotes giảm dần
        records.sort(key=lambda r: r["upvotes"], reverse=True)

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        q = (query or "").strip()
        if not q:
            return "NO RESULTS"
        clamped_limit = max(1, min(int(limit), 50))
        params = {"q": q, "limit": clamped_limit}

        def _fetch():
            resp = httpx.get(HF_SEARCH_URL, params=params, timeout=30.0, follow_redirects=True)
            if resp.status_code == 429:
                raise RetryableError("HF HTTP 429 rate limit", retry_after=_retry_after(resp, 5.0))
            resp.raise_for_status()
            return resp.json()

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        for item in items:
            if not isinstance(item, dict):
                continue
            paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
            paper_id = paper.get("id") or item.get("id")
            if not paper_id:
                continue

            paper_id = str(paper_id).strip()
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            title = " ".join(str(paper.get("title") or item.get("title") or "").split()) or "Untitled"

            # Ưu tiên ai_summary nếu có
            raw_summary = (
                paper.get("ai_summary")
                or item.get("ai_summary")
                or paper.get("summary")
                or item.get("summary")
                or ""
            )
            summary = " ".join(str(raw_summary).split())[:600]
            upvotes = int(paper.get("upvotes") or item.get("upvotes") or 0)
            github = paper.get("githubRepo") or item.get("githubRepo")
            stars = int(paper.get("githubStars") or item.get("githubStars") or 0)

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(name: str, arguments: dict) -> str:
    """Gọi công cụ MCP của Exa qua JSON-RPC POST."""
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    url = f"{EXA_URL}?exaApiKey={api_key}" if api_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments,
        },
    }

    def _do_request():
        resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        if resp.status_code == 429:
            raise RetryableError(
                "Exa HTTP 429 rate limit",
                retry_after=_retry_after(resp, 20.0),
            )
        resp.raise_for_status()

        # Parse SSE (data: ...) hoặc JSON thường
        text = resp.text
        data = None
        for line in text.splitlines():
            line_str = line.strip()
            if line_str.startswith("data:"):
                try:
                    data = json.loads(line_str[5:].strip())
                    break
                except json.JSONDecodeError:
                    pass
        if data is None:
            data = resp.json()

        # Xử lý lỗi JSON-RPC
        if "error" in data:
            err = data["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            if "rate limit" in err_msg.lower():
                raise RetryableError(f"Exa rate limit: {err_msg}", retry_after=20.0)
            raise RuntimeError(f"Exa error: {err_msg}")

        result = data.get("result", {})

        # Kiểm tra cờ rate limit trong result._meta (Exa free tier trả HTTP 200 kèm cờ)
        meta = result.get("_meta", {})
        meta_str = str(meta).lower()
        if "rate" in meta_str and "limit" in meta_str:
            raise RetryableError("Exa rate limit flag detected in _meta", retry_after=20.0)

        # Lấy nội dung text từ result.content
        contents = result.get("content", [])
        texts = []
        for item in contents:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        combined = "\n\n".join(texts).strip()

        # Kiểm tra nếu phản hồi là thông báo rate limit dạng văn bản ngắn
        if "rate limit" in combined.lower() and len(combined) < 500:
            raise RetryableError("Exa rate limit detected in content", retry_after=20.0)

        return combined

    # Exa có thể nghẽn IP chung; cho cap 60s
    return with_retry(_do_request, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        q = (query or "").strip()
        if not q:
            return "NO RESULTS"
        obj = objective.strip() if objective and objective.strip() else f"Search for {q}"
        num = max(1, min(int(num_results), 10))
        result = _call_exa_mcp("web_search_exa", {"query": q, "objective": obj, "numResults": num})
        if not result or not result.strip():
            return "NO RESULTS"
        return result
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        u = (url or "").strip()
        if not u:
            return "NO RESULTS"
        result = _call_exa_mcp("web_fetch_exa", {"urls": [u]})
        if not result or not result.strip():
            return "NO RESULTS"
        return result[:12000]
    except Exception as exc:
        return _redact_key(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
