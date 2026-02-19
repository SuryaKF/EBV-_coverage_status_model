from __future__ import annotations

from pathlib import Path

from coverage_pipeline.db import get_conn


def main() -> None:
    sql_path = Path("sql/001_init_pipeline.sql")
    sql = sql_path.read_text(encoding="utf-8")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

    print(f"schema_applied={sql_path}")


if __name__ == "__main__":
    main()
