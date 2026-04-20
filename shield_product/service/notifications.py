from typing import Any, Dict, Optional, Tuple

import requests

from .config import SETTINGS


def build_webhook_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "modality": job.get("modality"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "completed_at": job.get("completed_at"),
        "result": job.get("result") or {},
        "error_message": job.get("error_message"),
    }


def post_job_webhook(
    webhook_url: str,
    payload: Dict[str, Any],
    *,
    timeout_seconds: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    timeout = int(timeout_seconds or SETTINGS.webhook_timeout_seconds)
    attempts = int(max_attempts or SETTINGS.webhook_max_attempts)

    headers = {
        "Content-Type": "application/json",
        "User-Agent": SETTINGS.webhook_user_agent,
    }

    last_error: Optional[str] = None
    for _ in range(max(attempts, 1)):
        try:
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=timeout)
            if 200 <= int(response.status_code) < 300:
                return True, None
            last_error = f"Webhook HTTP {response.status_code}"
        except Exception as error:
            last_error = str(error)

    return False, last_error
