from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from coverage_pipeline.db import get_conn

ALLOWED_BASE_STATUSES = {
    "covered",
    "not covered",
    "coverage with conditions",
    "covered with condition",
    "coverage with condition",
    "yes",
    "no",
    "y",
    "n",
    "non-covered",
}

REQUIRED_FIELDS = ["payer_name", "state_name", "acronym", "expansion", "explanation", "coverage_status"]


STG_INSERT_SQL = """
INSERT INTO stg_coverage_data (
    raw_id,
    batch_id,
    source_name,
    payer_name,
    state_name,
    acronym,
    expansion,
    explanation,
    coverage_status,
    row_hash,
    validated_at
)
VALUES (
    %(raw_id)s,
    %(batch_id)s,
    %(source_name)s,
    %(payer_name)s,
    %(state_name)s,
    %(acronym)s,
    %(expansion)s,
    %(explanation)s,
    %(coverage_status)s,
    %(row_hash)s,
    %(validated_at)s
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


def _validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if row.get(field) is None or str(row[field]).strip() == "":
            errors.append(f"missing_{field}")

    raw_status = (row.get("coverage_status") or "").strip().lower()
    if raw_status and raw_status not in ALLOWED_BASE_STATUSES and "covered" not in raw_status and "condition" not in raw_status:
        errors.append("invalid_coverage_status")

    return errors


def validate_batch(batch_id: str) -> dict[str, Any]:
    validated_at = datetime.now(timezone.utc)
    staged = 0
    rejected = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.*
                FROM raw_coverage_data r
                LEFT JOIN stg_coverage_data s
                    ON s.batch_id = r.batch_id AND s.row_hash = r.row_hash
                WHERE r.batch_id = %s AND s.id IS NULL
                """,
                (batch_id,),
            )
            rows = cur.fetchall()

            for row in rows:
                errors = _validate_row(row)
                if errors:
                    params = {
                        "batch_id": row["batch_id"],
                        "source_name": row["source_name"],
                        "raw_id": row["id"],
                        "row_hash": row["row_hash"],
                        "raw_payload": json.dumps(row.get("raw_payload") or {}),
                        "reject_reason": ",".join(errors),
                        "rejected_at": validated_at,
                    }
                    cur.execute(REJECT_INSERT_SQL, params)
                    rejected += cur.rowcount
                else:
                    params = {
                        "raw_id": row["id"],
                        "batch_id": row["batch_id"],
                        "source_name": row["source_name"],
                        "payer_name": row["payer_name"],
                        "state_name": row["state_name"],
                        "acronym": row["acronym"],
                        "expansion": row["expansion"],
                        "explanation": row["explanation"],
                        "coverage_status": row["coverage_status"],
                        "row_hash": row["row_hash"],
                        "validated_at": validated_at,
                    }
                    cur.execute(STG_INSERT_SQL, params)
                    staged += cur.rowcount
        conn.commit()

    return {"batch_id": batch_id, "rows_staged": staged, "rows_rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw batch and split staged vs rejected rows")
    parser.add_argument("--batch-id", required=True)
    args = parser.parse_args()
    result = validate_batch(args.batch_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
