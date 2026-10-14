"""
arxiv-digest: daily digest of astro-ph.EP papers on planetary formation
Uses Gemini Flash (free) to filter and summarize, then sends via Gmail.
"""

import arxiv
from google import genai
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
import json
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
CATEGORY = "astro-ph.EP"
RESEARCH_INTEREST = "planetary formation, protoplanetary disks, planet formation mechanisms, dust and gas dynamics in disks, planetesimal formation, pebble accretion, giant planet formation, terrestrial planet formation"
MAX_PAPERS = 10          # how many recent papers to pull from arxiv
RELEVANCE_THRESHOLD = 6  # out of 10 — tune this up/down to be more/less selective
GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
GMAIL_ADDRESS   = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASS  = os.environ["GMAIL_APP_PASSWORD"]
TO_EMAIL        = os.environ["TO_EMAIL"]
# ─────────────────────────────────────────────────────────────────────────────


def fetch_recent_papers(category: str, max_results: int) -> list[arxiv.Result]:
    """Fetch the most recent papers from an arxiv category."""
    print(f"📡 Fetching up to {max_results} papers from {category}...")
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = list(arxiv.Client().results(search))
    print(f"   → Found {len(results)} papers")
    return results


def score_and_summarize(paper: arxiv.Result, client) -> dict | None:
    """
    Ask Gemini to:
      1. Score relevance for planetary formation research (1-10)
      2. If relevant (≥ threshold), produce a short summary + comment
    Returns a dict with paper info, or None if not relevant.
    Raises on API/parsing errors so the caller can count failures.
    """
    prompt = f"""You are an astrophysics researcher specialising in planetary formation.

Research interests: {RESEARCH_INTEREST}

Evaluate this arxiv paper:
Title: {paper.title}
Authors: {', '.join(a.name for a in paper.authors[:5])}{' et al.' if len(paper.authors) > 5 else ''}
Abstract: {paper.summary}

Reply with ONLY a JSON object (no markdown, no explanation) in this exact format:
{{
  "score": <integer 1-10, how relevant is this for planetary formation research>,
  "reason": "<one sentence: why relevant or not>",
  "summary": "<if score >= {RELEVANCE_THRESHOLD}: 2-3 sentence plain-language summary of the paper's contribution; else empty string>",
  "comment": "<if score >= {RELEVANCE_THRESHOLD}: 1-2 sentence comment on why this could matter for planetary formation research; else empty string>"
}}"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    text = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    score = int(data.get("score", 0))
    if score >= RELEVANCE_THRESHOLD:
        return {
            "title":   paper.title,
            "authors": ', '.join(a.name for a in paper.authors[:4]) + (' et al.' if len(paper.authors) > 4 else ''),
            "url":     paper.entry_id,
            "score":   score,
            "reason":  data.get("reason", ""),
            "summary": data.get("summary", ""),
            "comment": data.get("comment", ""),
        }
    return None


def build_html_email(papers: list[dict], date_str: str) -> str:
    """Build a clean HTML email body."""
    count = len(papers)
    header_color = "#1a1a2e"
    accent = "#4a90d9"

    items_html = ""
    for i, p in enumerate(papers, 1):
        stars = "⭐" * (p["score"] // 2)
        items_html += f"""
        <div style="margin-bottom:28px; padding:20px; background:#f8f9ff;
                    border-left:4px solid {accent}; border-radius:6px;">
          <div style="font-size:13px; color:#888; margin-bottom:4px;">
            #{i} &nbsp;·&nbsp; Rilevanza: {p['score']}/10 {stars}
          </div>
          <h2 style="margin:0 0 6px 0; font-size:17px; color:{header_color};">
            <a href="{p['url']}" style="color:{header_color}; text-decoration:none;">
              {p['title']}
            </a>
          </h2>
          <div style="font-size:13px; color:#666; margin-bottom:12px;">
            {p['authors']}
          </div>
          <p style="margin:0 0 8px 0; font-size:14px; color:#333; line-height:1.6;">
            <strong>📄 Riassunto:</strong> {p['summary']}
          </p>
          <p style="margin:0; font-size:14px; color:#555; line-height:1.6;
                    background:#eef3fb; padding:10px 12px; border-radius:4px;">
            <strong>💡 Perché è interessante:</strong> {p['comment']}
          </p>
          <div style="margin-top:10px;">
            <a href="{p['url']}" style="font-size:13px; color:{accent};">
              → Leggi su arxiv
            </a>
          </div>
        </div>"""

    if not items_html:
        items_html = """
        <p style="color:#888; font-style:italic; padding:20px 0;">
          Nessun paper rilevante trovato oggi. Riprova domani!
        </p>"""

    return f"""
    <html><body style="font-family:Georgia,serif; max-width:680px; margin:0 auto;
                       padding:20px; color:#222;">
      <div style="background:{header_color}; color:white; padding:24px 28px;
                  border-radius:8px; margin-bottom:28px;">
        <h1 style="margin:0 0 6px 0; font-size:22px;">🪐 arxiv Digest — Formazione Planetaria</h1>
        <div style="font-size:14px; opacity:0.8;">{date_str} &nbsp;·&nbsp;
          {count} paper{"s" if count != 1 else ""} selezionato{"i" if count != 1 else ""}
          da <code>astro-ph.EP</code>
        </div>
      </div>
      {items_html}
      <hr style="border:none; border-top:1px solid #eee; margin:32px 0;">
      <p style="font-size:12px; color:#aaa; text-align:center;">
        Generato automaticamente · Gemini Flash · GitHub Actions
      </p>
    </body></html>"""


def send_email(html_body: str, date_str: str, paper_count: int):
    """Send the digest via Gmail SMTP."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🪐 arxiv Digest — {paper_count} paper{'s' if paper_count != 1 else ''} su formazione planetaria ({date_str})"
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    print(f"📧 Sending email to {TO_EMAIL}...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASS)
        server.sendmail(GMAIL_ADDRESS, TO_EMAIL, msg.as_string())
    print("   ✅ Email sent!")


def main():
    date_str = datetime.now(timezone.utc).strftime("%d %B %Y")
    print(f"\n🌌 arxiv Digest — {date_str}\n{'─'*50}")

    # Init Gemini
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Fetch papers
    papers = fetch_recent_papers(CATEGORY, MAX_PAPERS)

    # Filter & summarize
    relevant = []
    errors = 0
    for i, paper in enumerate(papers, 1):
        print(f"   [{i}/{len(papers)}] {paper.title[:70]}...")
        try:
            result = score_and_summarize(paper, client)
        except Exception as e:
            errors += 1
            print(f"      ⚠️  Error: {e}")
            time.sleep(1.5)
            continue
        if result:
            relevant.append(result)
            print(f"      ✅ Score {result['score']}/10 — kept")
        else:
            print(f"      ⏭️  Not relevant enough")
        time.sleep(1.5)  # be polite to the API (free tier rate limits)

    if not papers:
        raise SystemExit("❌ No papers returned by arxiv")
    if errors == len(papers):
        raise SystemExit("❌ Gemini failed on every paper — not sending an empty digest")

    # Sort by score descending
    relevant.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n📊 {len(relevant)}/{len(papers)} papers selected as relevant\n")

    # Build & send email
    html = build_html_email(relevant, date_str)
    send_email(html, date_str, len(relevant))


if __name__ == "__main__":
    main()
