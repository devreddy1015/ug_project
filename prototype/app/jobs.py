import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .db import get_conn


STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELED = "canceled"

TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELED}
VALID_STATUSES = {
    STATUS_QUEUED,
    STATUS_PROCESSING,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_CANCELED,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_row(record: Dict[str, Any]) -> Dict[str, Any]:
    if record.get("result_json"):
        try:
            record["result"] = json.loads(record["result_json"])
        except json.JSONDecodeError:
            record["result"] = None
    else:
        record["result"] = None
    return record


def create_job(
    modality: str,
    queue_name: str,
    input_path: Optional[str] = None,
    input_text: Optional[str] = None,
    input_filename: Optional[str] = None,
    input_content_type: Optional[str] = None,
    input_size_bytes: Optional[int] = None,
    idempotency_key: Optional[str] = None,
) -> Tuple[str, bool]:
    if idempotency_key:
        existing = get_job_by_idempotency_key(idempotency_key)
        if existing is not None:
            return existing["id"], False

    job_id = str(uuid.uuid4())
    now = _now()
    with get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO jobs ("
                "id, modality, status, queue_name, created_at, updated_at, "
                "input_path, input_text, input_filename, input_content_type, input_size_bytes, idempotency_key"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    modality,
                    STATUS_QUEUED,
                    queue_name,
                    now,
                    now,
                    input_path,
                    input_text,
                    input_filename,
                    input_content_type,
                    input_size_bytes,
                    idempotency_key,
                ),
            )
        except sqlite3.IntegrityError:
            if idempotency_key:
                existing = get_job_by_idempotency_key(idempotency_key)
                if existing is not None:
                    return existing["id"], False
            raise
    return job_id, True


def get_job_by_idempotency_key(idempotency_key: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        return _decode_row(dict(row))


def update_job_status(
    job_id: str,
    status: str,
    *,
    task_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Unsupported status: {status}")

    now = _now()
    with get_conn() as conn:
        if status == STATUS_PROCESSING:
            conn.execute(
                "UPDATE jobs "
                "SET status = ?, updated_at = ?, started_at = COALESCE(started_at, ?), "
                "attempts = attempts + 1, task_id = COALESCE(?, task_id), error_message = NULL "
                "WHERE id = ?",
                (status, now, now, task_id, job_id),
            )
            return

        completed_at = now if status in TERMINAL_STATUSES else None
        conn.execute(
            "UPDATE jobs "
            "SET status = ?, updated_at = ?, completed_at = ?, task_id = COALESCE(?, task_id), "
            "error_message = CASE WHEN ? IS NULL THEN error_message ELSE ? END "
            "WHERE id = ?",
            (status, now, completed_at, task_id, error_message, error_message, job_id),
        )


def save_job_result(job_id: str, result: Dict[str, Any]) -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs "
            "SET status = ?, updated_at = ?, completed_at = ?, result_json = ?, error_message = NULL "
            "WHERE id = ?",
            (STATUS_COMPLETED, now, now, json.dumps(result), job_id),
        )


def mark_job_failed(job_id: str, error_message: str) -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ?, completed_at = ?, error_message = ? WHERE id = ?",
            (STATUS_FAILED, now, now, error_message, job_id),
        )


def cancel_job(job_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False
        if row["status"] in TERMINAL_STATUSES:
            return False
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ?, completed_at = ? WHERE id = ?",
            (STATUS_CANCELED, _now(), _now(), job_id),
        )
        return True


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return _decode_row(dict(row))


def list_jobs(
    *,
    limit: int,
    offset: int,
    status: Optional[str] = None,
    modality: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clauses = []
    params: List[Any] = []

    if status:
        clauses.append("status = ?")
        params.append(status)
    if modality:
        clauses.append("modality = ?")
        params.append(modality)

    where = ""
    if clauses:
        where = " WHERE " + " AND ".join(clauses)

    query = (
        "SELECT * FROM jobs"
        + where
        + " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])

    with get_conn() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
        return [_decode_row(dict(row)) for row in rows]


def count_jobs(*, status: Optional[str] = None, modality: Optional[str] = None) -> int:
    clauses = []
    params: List[Any] = []

    if status:
        clauses.append("status = ?")
        params.append(status)
    if modality:
        clauses.append("modality = ?")
        params.append(modality)

    where = ""
    if clauses:
        where = " WHERE " + " AND ".join(clauses)

    with get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM jobs" + where, tuple(params)).fetchone()
        return int(row["count"]) if row is not None else 0


def get_job_metrics() -> Dict[str, Any]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    with get_conn() as conn:
        by_status_rows = conn.execute(
            "SELECT status, COUNT(*) AS total FROM jobs GROUP BY status"
        ).fetchall()
        recent_failure_rows = conn.execute(
            "SELECT updated_at FROM jobs WHERE status = ?",
            (STATUS_FAILED,),
        ).fetchall()

    by_status = {row["status"]: int(row["total"]) for row in by_status_rows}
    recent_failures_24h = 0
    for row in recent_failure_rows:
        value = str(row["updated_at"] or "")
        try:
            timestamp = datetime.fromisoformat(value)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if timestamp >= cutoff:
                recent_failures_24h += 1
        except ValueError:
            continue

    return {
        "jobs_by_status": by_status,
        "recent_failures_24h": recent_failures_24h,
    }
