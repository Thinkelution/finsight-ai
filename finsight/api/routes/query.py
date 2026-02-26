"""POST /query endpoint for financial question answering."""

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from finsight.api.rate_limiter import rate_limiter
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
async def query(request_body: QueryRequest, request: Request):
    rate_limiter.check(request)

    engine = _get_engine()
    session_id = request_body.session_id or str(uuid.uuid4())

    if request_body.stream:
        return StreamingResponse(
            engine.query_stream(
                user_question=request_body.question,
                asset_class=request_body.asset_class,
                hours_back=request_body.hours_back,
            ),
            media_type="text/plain",
        )

    result = engine.query(
        user_question=request_body.question,
        asset_class=request_body.asset_class,
        hours_back=request_body.hours_back,
    )

    from finsight.inference.chat_history import ChatHistory
    history = ChatHistory(session_id)
    history.add_user_message(request_body.question)
    history.add_assistant_message(result["answer"])

    return QueryResponse(
        **result,
        session_id=session_id,
    )
