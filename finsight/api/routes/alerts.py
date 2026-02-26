"""GET /alerts endpoint for recent system alerts."""

from fastapi import APIRouter

from finsight.config.logging import get_logger
from finsight.inference.alerter import MarketAlerter

logger = get_logger(__name__)

router = APIRouter()


@router.get("/alerts")
async def get_alerts(limit: int = 20):
    alerter = MarketAlerter()
    return {"alerts": alerter.get_recent_alerts(limit=limit)}
