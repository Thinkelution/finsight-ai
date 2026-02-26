"""RAG query engine: embed question -> retrieve -> build context -> LLM."""

import ollama as ollama_client

from finsight.config.logging import get_logger
from finsight.config.settings import settings
from finsight.inference.context_builder import ContextBuilder
from finsight.inference.fallback import query_with_fallback
from finsight.inference.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from finsight.processing.embedder import embed_text
from finsight.storage.retriever import TimeWeightedRetriever

logger = get_logger(__name__)


class FinancialQueryEngine:
    def __init__(self):
        self.retriever = TimeWeightedRetriever()
        self.context_builder = ContextBuilder()

    def query(
        self,
        user_question: str,
        asset_class: str | None = None,
        hours_back: int = 24,
    ) -> dict:
        logger.info("query_start", question=user_question[:100], asset_class=asset_class)

        question_embedding = embed_text(user_question)

        chunks = self.retriever.retrieve(
            query_embedding=question_embedding,
            k=settings.retrieval_top_k,
            asset_class=asset_class,
            hours_back=hours_back,
        )

        context = self.context_builder.build_context(chunks)

        user_prompt = build_user_prompt(
            question=user_question,
            news_chunks=context["news_chunks"],
            live_prices=context["live_prices"],
            market_summary=context["market_summary"],
        )

        llm_result = query_with_fallback(user_prompt)

        result = {
            "answer": llm_result["answer"],
            "sources": context["source_urls"],
            "live_prices_at": context["live_prices"].get("timestamp", "N/A"),
            "chunks_used": len(chunks),
            "provider": llm_result.get("provider", "ollama"),
        }

        logger.info(
            "query_complete",
            chunks_used=len(chunks),
            answer_len=len(llm_result["answer"]),
            provider=llm_result.get("provider"),
        )
        return result

    def query_stream(
        self,
        user_question: str,
        asset_class: str | None = None,
        hours_back: int = 24,
    ):
        """Streaming version of query that yields answer tokens."""
        question_embedding = embed_text(user_question)

        chunks = self.retriever.retrieve(
            query_embedding=question_embedding,
            k=settings.retrieval_top_k,
            asset_class=asset_class,
            hours_back=hours_back,
        )

        context = self.context_builder.build_context(chunks)

        user_prompt = build_user_prompt(
            question=user_question,
            news_chunks=context["news_chunks"],
            live_prices=context["live_prices"],
            market_summary=context["market_summary"],
        )

        try:
            stream = ollama_client.chat(
                model=settings.ollama_llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.1, "num_ctx": 8192},
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                yield token
        except Exception as e:
            logger.error("llm_stream_failed", error=str(e))
            yield f"\n[Error: {str(e)}]"
