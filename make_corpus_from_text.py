
import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any, Optional

try:
    import yaml  # Optional, only for YAML manifest
except Exception:
    yaml = None

DEFAULT_MIN_LEN = 800
DEFAULT_MAX_LEN = 1200
DEFAULT_OVERLAP = 120

SKILL_KEYWORDS = {
    "python": "Python",
    "sql": "SQL",
    "r ": "R",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "looker": "Looker",
    "sigma": "Sigma",
    "dbt": "DBT",
    "bigquery": "BigQuery",
    "aws": "AWS",
    "gcp": "GCP",
    "tensor": "TensorFlow",
    "scikit": "Scikit-learn",
    "prophet": "Prophet",
    "isolation forest": "Isolation Forest",
    "lstm": "LSTM",
    "merlion": "Merlion",
    "anomaly": "Anomaly Detection",
    "rag": "RAG",
    "embedding": "Embeddings",
    "time series": "Time Series",
    "nps": "NPS",
    "dashboard": "Dashboards",
    "ga4": "Google Analytics",
    "google analytics": "Google Analytics",
    "ab test": "A/B Testing",
    "a/b": "A/B Testing",
}

def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"([.!?])", text)
    sentences = []
    for i in range(0, len(parts), 2):
        sent = parts[i].strip()
        if i+1 < len(parts):
            sent += parts[i+1]
        if sent:
            sentences.append(sent.strip())
    return sentences

def chunk_text(text: str, min_len=DEFAULT_MIN_LEN, max_len=DEFAULT_MAX_LEN, overlap=DEFAULT_OVERLAP) -> List[str]:
    sents = split_into_sentences(text)
    chunks = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_len:
            cur = f"{cur} {s}"
        else:
            if len(cur) < min_len and s:
                cur = f"{cur} {s}"
            chunks.append(cur.strip())
            if overlap > 0 and len(chunks[-1]) > overlap:
                cur = chunks[-1][-overlap:] + " " + s
            else:
                cur = s
            cur = re.sub(r"^\S*\s", "", cur, count=1)
    if cur:
        chunks.append(cur.strip())
    merged = []
    buf = ""
    for ch in chunks:
        if not buf:
            buf = ch
            continue
        if len(buf) < min_len:
            if len(buf) + 1 + len(ch) <= max_len * 1.2:
                buf = f"{buf} {ch}"
            else:
                merged.append(buf)
                buf = ch
        else:
            merged.append(buf)
            buf = ch
    if buf:
        merged.append(buf)
    return [m.strip() for m in merged if m.strip()]

def infer_skills(text: str) -> List[str]:
    t = text.lower()
    skills = set()
    for kw, label in SKILL_KEYWORDS.items():
        if kw in t:
            skills.add(label)
    return sorted(skills)

def build_records(
    text: str,
    source: str,
    section: str,
    url: Optional[str],
    date_range: str = "",
    id_prefix: Optional[str] = None,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Dict[str, Any]]:
    chunks = chunk_text(text, min_len=min_len, max_len=max_len, overlap=overlap)
    records = []
    base_prefix = id_prefix or f"{source}#{re.sub(r'[^a-z0-9]+', '_', section.lower()).strip('_')}"
    for i, ch in enumerate(chunks, start=1):
        rec = {
            "id": f"{base_prefix}_c{i}",
            "source": source,
            "section": section,
            "date_range": date_range,
            "skills": infer_skills(ch),
            "text": ch,
        }
        if url:
            rec["url"] = url
        records.append(rec)
    return records

def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if path.lower().endswith((".yaml", ".yml")):
        if yaml is None:
            raise RuntimeError("PyYAML not installed. `pip install pyyaml` or use JSON manifest.")
        manifest = yaml.safe_load(raw)
    else:
        manifest = json.loads(raw)
    if not isinstance(manifest, list):
        raise ValueError("Manifest must be a list of items.")
    return manifest

def main():
    p = argparse.ArgumentParser(description="Build JSONL chunks from raw text for Resume Q&A corpus.")
    p.add_argument("--text_file", action="append", help="Path to a text/markdown file. Can be repeated.")
    p.add_argument("--source", action="append", help="Source label for the preceding --text_file (website/linkedin/other). Repeat aligned with --text_file.")
    p.add_argument("--section", action="append", help="Section label for the preceding --text_file (e.g., 'About Me'). Repeat aligned with --text_file.")
    p.add_argument("--url", action="append", help="Optional URL for the preceding --text_file. Repeat aligned with --text_file.")
    p.add_argument("--date_range", action="append", help="Optional date_range for the preceding --text_file. Repeat aligned with --text_file.")
    p.add_argument("--id_prefix", action="append", help="Optional id prefix for the preceding --text_file.")
    p.add_argument("--min_len", type=int, default=DEFAULT_MIN_LEN)
    p.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("--manifest", help="YAML or JSON manifest listing items: {text_file, source, section, url?, date_range?, id_prefix?}")
    p.add_argument("--out", default="corpus_extra.jsonl", help="Output JSONL path")
    args = p.parse_args()

    all_items = []

    if args.manifest:
        items = load_manifest(args.manifest)
        for it in items:
            for key in ["text_file", "source", "section"]:
                if key not in it:
                    raise ValueError(f"Manifest item missing required key: {key}")
            all_items.append({
                "text_file": it["text_file"],
                "source": it["source"],
                "section": it["section"],
                "url": it.get("url"),
                "date_range": it.get("date_range", ""),
                "id_prefix": it.get("id_prefix"),
            })

    if args.text_file:
        tfiles = args.text_file
        n = len(tfiles)

        def get_aligned(lst, i):
            if not lst: return None
            return lst[i] if i < len(lst) else lst[-1]

        for i in range(n):
            all_items.append({
                "text_file": tfiles[i],
                "source": get_aligned(args.source, i) or "website",
                "section": get_aligned(args.section, i) or "Untitled",
                "url": get_aligned(args.url, i),
                "date_range": get_aligned(args.date_range, i) or "",
                "id_prefix": get_aligned(args.id_prefix, i),
            })

    if not all_items:
        print("No input provided. Use --manifest or --text_file ... --source ... --section ...", file=sys.stderr)
        sys.exit(1)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    total = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for item in all_items:
            with open(item["text_file"], "r", encoding="utf-8") as f_in:
                raw = f_in.read()
            recs = build_records(
                text=raw,
                source=item["source"],
                section=item["section"],
                url=item["url"],
                date_range=item["date_range"],
                id_prefix=item["id_prefix"],
                min_len=args.min_len,
                max_len=args.max_len,
                overlap=args.overlap,
            )
            for rec in recs:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} records to {out_path}")

if __name__ == "__main__":
    main()
