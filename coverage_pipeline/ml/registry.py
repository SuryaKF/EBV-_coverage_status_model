from __future__ import annotations

from datetime import datetime, timezone

from coverage_pipeline.db import get_conn

VALID_STATUSES = {"staging", "approved", "archived"}


def register_model(
    model_version: str,
    artifact_uri: str,
    snapshot_id: str,
    feature_version: str,
    metrics_json: str,
    status: str = "staging",
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_registry (
                    model_version,
                    artifact_uri,
                    snapshot_id,
                    feature_version,
                    metrics_json,
                    status,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                (
                    model_version,
                    artifact_uri,
                    snapshot_id,
                    feature_version,
                    metrics_json,
                    status,
                    datetime.now(timezone.utc),
                ),
            )
        conn.commit()


def approve_model(model_version: str, approved_by: str) -> None:
    approved_at = datetime.now(timezone.utc)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE model_registry SET status = 'archived' WHERE status = 'approved'")
            cur.execute(
                """
                UPDATE model_registry
                SET status = 'approved', approved_by = %s, approved_at = %s
                WHERE model_version = %s
                """,
                (approved_by, approved_at, model_version),
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"Model version not found: {model_version}")
        conn.commit()


def get_approved_model() -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT model_version, artifact_uri, feature_version, snapshot_id, metrics_json
                FROM model_registry
                WHERE status = 'approved'
                ORDER BY approved_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("No approved model found in model_registry")
            return row


def get_snapshot(snapshot_id: str | None) -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if snapshot_id:
                cur.execute(
                    "SELECT snapshot_id, snapshot_table, feature_version FROM dataset_registry WHERE snapshot_id = %s",
                    (snapshot_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT snapshot_id, snapshot_table, feature_version
                    FROM dataset_registry
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("No dataset snapshot found")
            return row
