from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from coverage_pipeline.config import require_database_url


@contextmanager
def get_conn(autocommit: bool = False) -> Iterator[psycopg.Connection[Any]]:
    conn = psycopg.connect(require_database_url(), row_factory=dict_row)
    conn.autocommit = autocommit
    try:
        yield conn
    finally:
        conn.close()
