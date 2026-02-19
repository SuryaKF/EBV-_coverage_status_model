from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from coverage_pipeline.db import get_conn
from coverage_pipeline.normalization import VALID_CANONICAL_CLASSES, canonicalize_coverage_status

CURATED_INSERT_SQL = """
INSERT INTO curated_coverage_data (
    stg_id,
    batch_id,
    payer_name,
    state_name,
    acronym,
    expansion,
    explanation,
    coverage_status,
    row_hash,
    processed_at
)
VALUES (
    %(stg_id)s,
    %(batch_id)s,
    %(payer_name)s,
    %(state_name)s,
    %(acronym)s,
    %(expansion)s,
    %(explanation)s,
    %(coverage_status)s,
    %(row_hash)s,
    %(processed_at)s
)
ON CONFLICT (batch_id, row_hash) DO NOTHING
"""

REJECT_INSERT_SQL = """
INSERT INTO rejected_coverage_data (
    batch_id,
    source_name,
    raw_id,
    row_hash,
    raw_payload,
    reject_reason,
    rejected_at
)
VALUES (
    %(batch_id)s,
    %(source_name)s,
    %(raw_id)s,
    %(row_hash)s,
    %(raw_payload)s,
    %(reject_reason)s,
    %(rejected_at)s
)
ON CONFLICT (batch_id, row_hash, reject_reason) DO NOTHING
"""


def transform_batch(batch_id: str) -> dict[str, Any]:
    processed_at = datetime.now(timezone.utc)
    inserted = 0
    rejected = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.*, r.id AS raw_id, r.source_name, r.raw_payload
                FROM stg_coverage_data s
                JOIN raw_coverage_data r
                    ON r.id = s.raw_id
                LEFT JOIN curated_coverage_data c
                    ON c.batch_id = s.batch_id AND c.row_hash = s.row_hash
                WHERE s.batch_id = %s AND c.id IS NULL
                """,
                (batch_id,),
            )
            rows = cur.fetchall()

            for row in rows:
                canonical = canonicalize_coverage_status(row["coverage_status"], row["acronym"])
                if canonical not in VALID_CANONICAL_CLASSES:
                    cur.execute(
                        REJECT_INSERT_SQL,
                        {
                            "batch_id": row["batch_id"],
                            "source_name": row["source_name"],
                            "raw_id": row["raw_id"],
                            "row_hash": row["row_hash"],
                            "raw_payload": json.dumps(row.get("raw_payload") or {}),
                            "reject_reason": "cannot_map_to_canonical_class",
                            "rejected_at": processed_at,
                        },
                    )
                    rejected += cur.rowcount
                    continue

                cur.execute(
                    CURATED_INSERT_SQL,
                    {
                        "stg_id": row["id"],
                        "batch_id": row["batch_id"],
                        "payer_name": row["payer_name"],
                        "state_name": row["state_name"],
                        "acronym": row["acronym"],
                        "expansion": row["expansion"],
                        "explanation": row["explanation"],
                        "coverage_status": canonical,
                        "row_hash": row["row_hash"],
                        "processed_at": processed_at,
                    },
                )
                inserted += cur.rowcount

        conn.commit()

    return {"batch_id": batch_id, "rows_curated": inserted, "rows_rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Transform staged rows into canonical curated rows")
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    result = transform_batch(args.batch_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
