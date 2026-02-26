"""Celery application configuration with Redis broker and beat schedule."""

from celery import Celery
from celery.schedules import crontab

from finsight.config.settings import settings

app = Celery(
    "finsight",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["finsight.workers.tasks"],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    task_soft_time_limit=300,
    task_time_limit=600,
)

app.conf.beat_schedule = {
    "fetch-rss-feeds": {
        "task": "finsight.workers.tasks.fetch_rss_feeds",
        "schedule": 60.0,
    },
    "scrape-web-news": {
        "task": "finsight.workers.tasks.scrape_web_news",
        "schedule": 300.0,
    },
    "fetch-social-feeds": {
        "task": "finsight.workers.tasks.fetch_social_feeds",
        "schedule": 300.0,
    },
    "refresh-market-summary": {
        "task": "finsight.workers.tasks.refresh_market_summary",
        "schedule": float(settings.summary_refresh_interval),
    },
    "cleanup-expired-chunks": {
        "task": "finsight.workers.tasks.cleanup_expired_chunks",
        "schedule": crontab(minute=0, hour="*/6"),
    },
    "check-price-alerts": {
        "task": "finsight.workers.tasks.check_price_alerts",
        "schedule": 60.0,
    },
}
