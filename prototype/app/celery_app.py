from celery import Celery
from kombu import Queue

from .config import SETTINGS


celery_app = Celery(
    "shield_proto",
    broker=SETTINGS.redis_url,
    backend=SETTINGS.redis_url,
)

celery_app.conf.update(
    task_queues=(
        Queue("text"),
        Queue("image"),
        Queue("video"),
        Queue("audio"),
        Queue("default"),
    ),
    task_routes={
        "app.tasks.score_text": {"queue": "text"},
        "app.tasks.score_image": {"queue": "image"},
        "app.tasks.score_video": {"queue": "video"},
        "app.tasks.score_audio": {"queue": "audio"},
    },
    task_default_queue="default",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=SETTINGS.result_ttl_seconds,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=SETTINGS.task_soft_time_limit_sec,
    task_time_limit=SETTINGS.task_time_limit_sec,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    broker_connection_retry_on_startup=True,
    timezone="UTC",
    enable_utc=True,
)

celery_app.autodiscover_tasks(["app"])
