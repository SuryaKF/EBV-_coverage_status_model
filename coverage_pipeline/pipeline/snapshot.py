from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone

from coverage_pipeline.config import settings
from coverage_pipeline.db import get_conn


def create_snapshot(batch_id: str | None = None) -> dict[str, str | int]:
    now = datetime.now(timezone.utc)
    snapshot_id = str(uuid.uuid4())
    suffix = now.strftime("%Y%m%d_%H%M")
    table_name = f"train_snapshot_{suffix}"

    where_clause = ""
    params: list[str] = []
    if batch_id:
        where_clause = "WHERE batch_id = %s"
        params.append(batch_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS
                SELECT
                    payer_name,
                    state_name,
                    acronym,
                    expansion,
                    explanation,
                    coverage_status,
                    batch_id,
                    row_hash,
                    processed_at
                FROM curated_coverage_data
                {where_clause}
                """,
                params,
            )
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}")
            row_count = int(cur.fetchone()["cnt"])

            cur.execute(
                """
                INSERT INTO dataset_registry (
                    snapshot_id,
                    snapshot_table,
                    row_count,
                    feature_version,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (snapshot_id, table_name, row_count, settings.feature_version, now),
            )
        conn.commit()

    return {
        "snapshot_id": snapshot_id,
        "snapshot_table": table_name,
        "row_count": row_count,
        "feature_version": settings.feature_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create immutable training snapshot from curated data")
    parser.add_argument("--batch-id", default=None)
    args = parser.parse_args()
    result = create_snapshot(batch_id=args.batch_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
