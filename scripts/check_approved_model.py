from __future__ import annotations

import sys

from coverage_pipeline.db import get_conn


def main() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM model_registry WHERE status = 'approved'")
            count = int(cur.fetchone()["cnt"])

    if count != 1:
        raise SystemExit(f"Expected exactly one approved model, found: {count}")

    print("approved_model_count=1")


if __name__ == "__main__":
    main()
