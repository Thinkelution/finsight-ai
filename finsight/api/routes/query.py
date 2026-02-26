"""POST /query endpoint for financial question answering."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from finsight.api.schemas import QueryRequest, QueryResponse
from finsight.config.logging import get_logger
from finsight.inference.query_engine import FinancialQueryEngine

logger = get_logger(__name__)

router = APIRouter()

_engine: FinancialQueryEngine | None = None


def _get_engine() -> FinancialQueryEngine:
    global _engine
    if _engine is None:
        _engine = FinancialQueryEngine()
    return _engine


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    engine = _get_engine()

    if request.stream:
        return StreamingResponse(
            engine.query_stream(
                user_question=request.question,
                asset_class=request.asset_class,
                hours_back=request.hours_back,
            ),
            media_type="text/plain",
        )

    result = engine.query(
        user_question=request.question,
        asset_class=request.asset_class,
        hours_back=request.hours_back,
    )

    return QueryResponse(**result)
