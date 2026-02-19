from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from coverage_pipeline.db import get_conn

RAW_INSERT_SQL = """
INSERT INTO raw_coverage_data (
    batch_id,
    source_name,
    payer_name,
    state_name,
    acronym,
    expansion,
    explanation,
    coverage_status,
    row_hash,
    raw_payload,
    ingested_at
)
VALUES (
    %(batch_id)s,
    %(source_name)s,
    %(payer_name)s,
    %(state_name)s,
    %(acronym)s,
    %(expansion)s,
    %(explanation)s,
    %(coverage_status)s,
    %(row_hash)s,
    %(raw_payload)s,
    %(ingested_at)s
)
ON CONFLICT (batch_id, row_hash) DO NOTHING
"""


def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode CSV file: {path}")


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() == "nan" or text == "":
        return None
    return text


def ingest_csv(input_path: str, batch_id: str | None = None, source_name: str | None = None) -> dict[str, Any]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    df = _read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]

    batch = batch_id or str(uuid.uuid4())
    source = source_name or os.path.basename(input_path)
    now = datetime.now(timezone.utc)

    inserted = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                payload = {k: (None if pd.isna(v) else str(v)) for k, v in row.to_dict().items()}
                payload_json = json.dumps(payload, sort_keys=True)
                row_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

                params = {
                    "batch_id": batch,
                    "source_name": source,
                    "payer_name": _norm(payload.get("PAYER NAME") or payload.get("payer_name")),
                    "state_name": _norm(payload.get("STATE NAME") or payload.get("state_name")),
                    "acronym": _norm(payload.get("Acronym") or payload.get("ACRONYM") or payload.get("acronym")),
                    "expansion": _norm(payload.get("EXPANSION") or payload.get("expansion")),
                    "explanation": _norm(payload.get("Explanation") or payload.get("EXPLANATION") or payload.get("explanation")),
                    "coverage_status": _norm(payload.get("Coverage Status") or payload.get("COVERAGE STATUS") or payload.get("coverage_status")),
                    "row_hash": row_hash,
                    "raw_payload": payload_json,
                    "ingested_at": now,
                }
                cur.execute(RAW_INSERT_SQL, params)
                inserted += cur.rowcount
        conn.commit()

    return {"batch_id": batch, "source_name": source, "rows_seen": len(df), "rows_inserted": inserted}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw coverage CSV into Postgres raw_coverage_data")
    parser.add_argument("--input", required=True, help="Path to source CSV")
    parser.add_argument("--batch-id", default=None, help="Optional caller-provided batch_id")
    parser.add_argument("--source-name", default=None, help="Optional source name")
    args = parser.parse_args()

    result = ingest_csv(args.input, batch_id=args.batch_id, source_name=args.source_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
