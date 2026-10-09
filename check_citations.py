"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REF_LINE = re.compile(r"^\[(\d+)\]\s+(.*)$")
_URL_REGEX = re.compile(r"https?://[^\s()\[\]<>]+")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []

    # 1. sources cannot be empty or invalid
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    # 2. validate source entries
    source_by_n = {}
    seen_urls = set()
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry must be an object: {entry!r}")
            continue

        n = entry.get("n")
        # bool is a subclass of int in Python, but True/False are not valid citation numbers.
        if type(n) is not int:
            problems.append(f"source n={n!r} must be an integer")
        elif n in source_by_n:
            problems.append(f"duplicate source number [{n}] in sources.json")
        else:
            source_by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source n={n!r} has invalid URL: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate URL in sources.json: {url}")
        else:
            seen_urls.add(url)

    # 3. check for "## References" heading
    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("report is missing '## References' heading")
        body = report_text
        ref_section = ""
    else:
        body = report_text[:matches[-1].start()]
        ref_section = report_text[matches[-1].end():]

    # 4. extract citations from body only (ignoring code blocks/spans and markdown links)
    cited_numbers = set()
    segments = _CODE.split(body)
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # inside code block or span
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited_numbers.add(num)

    for num in cited_numbers:
        if num not in source_by_n:
            problems.append(f"[{num}] cited in body but missing from sources.json")

    for num in source_by_n:
        if num not in cited_numbers:
            problems.append(f"source [{num}] never cited in report body")

    # 5 & 6. validate ## References section lines
    if matches:
        ref_lines = []
        for line in ref_section.strip().splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            m = _REF_LINE.match(line_str)
            if m:
                ref_lines.append((int(m.group(1)), line_str))
            else:
                problems.append(f"References line does not start with [n]: {line_str}")

        ref_numbers = set()
        for num, line_str in ref_lines:
            if num not in source_by_n:
                problems.append(f"reference [{num}] is not a source in sources.json")
            if num in ref_numbers:
                problems.append(f"duplicate reference line for [{num}]")
            else:
                ref_numbers.add(num)

            urls = _URL_REGEX.findall(line_str)
            if len(urls) == 0:
                problems.append(f"reference line [{num}] contains no URL: {line_str}")
            elif len(urls) > 1:
                problems.append(f"reference line [{num}] contains multiple URLs: {line_str}")
            else:
                ref_url = urls[0].rstrip(".,;)")
                expected_url = source_by_n.get(num, {}).get("url")
                if expected_url and ref_url != expected_url:
                    problems.append(f"reference [{num}] URL mismatch: expected '{expected_url}', got '{ref_url}'")

        for num in source_by_n:
            if num not in ref_numbers:
                problems.append(f"missing reference line for source [{num}]")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
