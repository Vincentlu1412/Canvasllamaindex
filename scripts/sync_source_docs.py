"""Fetch official UC Berkeley pages and cache their text content locally.

This does NOT auto-overwrite data/source_docs/*.md. It writes cleaned page
text to data/raw_scrapes/<slug>.txt so a human (or an LLM-assisted editing
pass) can diff it against the current curated doc and fold in anything that
actually changed -- the same manual-review workflow used to build the
current docs, just repeatable instead of one-off.

We deliberately don't auto-write straight into data/source_docs/: raw page
text is full of nav menus, cookie banners, and other noise that would hurt
retrieval quality if embedded as-is, and this project's whole premise is
not shipping unverified "facts" into the knowledge base.

Usage:
    python scripts/sync_source_docs.py                  # fetch everything
    python scripts/sync_source_docs.py --only cal1card,dining_services
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_scrapes"
REQUEST_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (compatible; CanvasBerkeleyAssistantBot/0.1; "
    "educational RAG project, non-commercial data collection)"
)


@dataclass(frozen=True)
class Source:
    slug: str  # matches data/source_docs/<slug>.md when a curated doc exists
    title: str
    url: str


# Sources matching the current curated docs -- re-fetching these lets you
# diff the live page against data/source_docs/<slug>.md to spot drift.
SOURCES: list[Source] = [
    Source("cal1card", "Cal 1 Card", "https://cal1card.berkeley.edu/"),
    Source("campus_map", "Campus Map", "https://www.berkeley.edu/map/"),
    Source("dining_services", "Berkeley Dining", "https://dining.berkeley.edu/"),
    Source("health_services", "University Health Services (Tang Center)", "https://uhs.berkeley.edu/"),
    Source("library_services", "UC Berkeley Library", "https://www.lib.berkeley.edu/"),
    Source("recreation_services", "Recreational Sports (RSF)", "https://recsports.berkeley.edu/"),
    Source("student_services", "Cal Student Central", "https://calstudentcentral.berkeley.edu/"),
    Source("tech_services", "Student Technology Services", "https://sts.berkeley.edu/"),
    Source("transportation", "Parking & Transportation", "https://pt.berkeley.edu/"),
    Source("accessibility_safety_dsp", "Disabled Students' Program", "https://dsp.berkeley.edu/"),
    Source("accessibility_safety_ucpd", "UCPD / Campus Safety", "https://ucpd.berkeley.edu/"),
    # Candidate new topics for expanding coverage. URLs need verifying by a
    # human before any content gets folded into data/source_docs/ -- do not
    # just uncomment and trust an LLM's guess at the right subdomain.
    # Source("financial_aid", "Financial Aid & Scholarships", "https://financialaid.berkeley.edu/"),
    # Source("international_office", "Berkeley International Office", "https://internationaloffice.berkeley.edu/"),
    # Source("housing", "Cal Housing", "https://housing.berkeley.edu/"),
    # Source("registrar", "Registrar / Academic Calendar", "https://registrar.berkeley.edu/"),
]


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def fetch(source: Source) -> str | None:
    try:
        resp = requests.get(source.url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return extract_text(resp.text)
    except requests.RequestException as exc:
        print(f"[FAIL] {source.slug}: {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated slugs to fetch (default: all)")
    args = parser.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    fetched = failed = 0
    for source in SOURCES:
        if wanted and source.slug not in wanted:
            continue
        text = fetch(source)
        if text is None:
            failed += 1
            continue
        header = (
            f"# {source.title}\n"
            f"# source_url: {source.url}\n"
            f"# fetched_at: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n"
            f"{'-' * 60}\n\n"
        )
        out_path = RAW_DIR / f"{source.slug}.txt"
        out_path.write_text(header + text, encoding="utf-8")
        print(f"[OK] {source.slug} -> {out_path}")
        fetched += 1

    print(f"\nDone: {fetched} fetched, {failed} failed.")
    if fetched:
        print(
            "Review data/raw_scrapes/, then manually fold any real changes "
            "into the matching file under data/source_docs/ (keep the "
            "declarative, question-free style and update the "
            "'_Last reviewed:_' date)."
        )


if __name__ == "__main__":
    main()
