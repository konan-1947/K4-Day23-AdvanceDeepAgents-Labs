"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    if not topic or not str(topic).strip():
        return "topic"
    text = str(topic).strip().lower()
    # Replace any sequence of non-alphanumeric characters with a single hyphen
    cleaned = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    # Clean up any potential directory traversal patterns
    cleaned = cleaned.replace("..", "").replace("/", "").replace("\\", "").strip("-")
    cleaned = cleaned[:60].rstrip("-")
    return cleaned or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please conduct an in-depth, rigorous research survey on the topic: '{topic.strip()}'.\n\n"
        "Follow the multi-agent research workflow strictly:\n"
        "1. Start by planning with `write_todos`, splitting the topic into at least 3 distinct thematic sub-questions.\n"
        "2. Delegate each sub-question in parallel to `researcher` subagents via `task`, covering multiple source families (`arxiv`, `hf-daily`, `hf-search`, `web`).\n"
        "3. Inspect the returned researcher notes in `/tmp/work/research/notes/`, aggregate all valid sources into `/tmp/work/research/sources.json` (numbered from 1, deduplicated URLs, at least 3 distinct source families).\n"
        "4. Synthesize the report into `/tmp/work/report/report.md` following `REPORT_TEMPLATE.md` with theme-by-theme comparisons and inline [n] citations. Do NOT write `## References` yourself.\n"
        "5. Execute `/tmp/work/research/finalize_citations.py` to auto-generate `## References` and align citations.\n"
        "6. Execute `/tmp/work/research/check_citations.py` and fix any issues until it prints `OK`.\n"
        "7. Spot-check 3-5 key claims with the `citation-checker` subagent."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_counts = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Tally tool calls from message attributes or additional_kwargs
        t_calls = getattr(msg, "tool_calls", None)
        if not t_calls and hasattr(msg, "additional_kwargs"):
            t_calls = msg.additional_kwargs.get("tool_calls")
        if t_calls:
            for call in t_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    tool_counts[name] += 1
                    if name == "task":
                        subagent_calls += 1

        # Tally tokens from usage_metadata
        usage = getattr(msg, "usage_metadata", None)
        if usage and isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)

    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_counts),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])

    report_bytes = files.get(REPORT_PATH)
    if report_bytes is None or not report_bytes.strip():
        raise RuntimeError(f"Report is missing or empty at {REPORT_PATH}")
    report_text = report_bytes.decode("utf-8")

    sources_bytes = files.get(SOURCES_PATH)
    if sources_bytes is None or not sources_bytes.strip():
        raise RuntimeError(f"sources.json is missing or empty at {SOURCES_PATH}")

    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"sources.json is invalid JSON: {exc}")

    if not isinstance(sources, list) or len(sources) == 0:
        raise RuntimeError("sources.json must be a non-empty list of objects")

    summary_info = summarize(messages, elapsed, model_name)
    distinct_families = sorted({
        s.get("source") for s in sources if isinstance(s, dict) and s.get("source")
    })

    meta = {
        "topic": topic,
        **summary_info,
        "n_sources": len(sources),
        "source_families": distinct_families,
    }

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    slug = slugify(topic)
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"
    report_path = reports_dir / f"{slug}.md"

    sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")

    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    cleaned_topic = (topic or "").strip()
    if not cleaned_topic:
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    model = make_model()
    model_name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or os.getenv("LAB_MODEL", "unknown")
    )
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })
        agent = build_lead_agent(backend, model)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": build_prompt(cleaned_topic)}]},
            config={"recursion_limit": 1000},
        )
        elapsed = time.monotonic() - start
        messages = result.get("messages", []) if isinstance(result, dict) else []

        try:
            report_path = save_outputs(backend, cleaned_topic, messages, elapsed, model_name)
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
