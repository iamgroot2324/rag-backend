"""Custom RAG pipeline (deliberately not using RetrievalQAChain)."""

from app.core.config import Settings
from app.schemas.chat import SourceChunk
from app.services.llm import LLMClient
from app.services.vector_store import RetrievedChunk, VectorStore


# Turns a context-dependent follow-up ("what about its price?")
# into a standalone query.
_CONDENSE_PROMPT = (
    "Rewrite the user's latest message as a standalone search query, "
    "using the conversation for context (resolve pronouns and references). "
    "Return only the query."
)


# Grounds the model in the retrieved context to reduce hallucination.
_ANSWER_PROMPT = (
    "You are a helpful assistant answering questions using the provided document context.\n"
    "Use the context as the primary source of truth.\n"
    "If the answer is explicitly stated or can be directly determined from the context, "
    "answer the user's question directly and confidently.\n"
    "Do not say you don't know when the requested information is present in the context.\n"
    "If the user asks for a person's name and the person's name appears in the context, "
    "give the name directly.\n"
    "If the user asks what the document is about, summarize the available context.\n"
    "If the requested information is genuinely not present in the context, say "
    "\"I don't know.\"\n\n"
    "Context:\n{context}"
)

class RagService:
    """Pipeline: condense question -> embed -> retrieve -> generate answer."""

    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._store = store
        self._top_k = settings.retrieval_top_k

    def _condense(
        self,
        history: list[dict[str, str]],
        message: str,
    ) -> str:
        """Make the question self-contained so multi-turn queries retrieve well."""

        if not history:
            # First turn: nothing to resolve.
            return message

        messages = [
            {"role": "system", "content": _CONDENSE_PROMPT},
            *history[-6:],
            {"role": "user", "content": message},
        ]

        return self._llm.chat(messages).strip() or message

    def answer(
        self,
        history: list[dict[str, str]],
        message: str,
    ) -> tuple[str, list[SourceChunk]]:
        """Answer a question from the ingested documents; returns (answer, sources)."""

        # 1. Rewrite the question
        # 2. Embed it
        # 3. Find the closest chunks
        query = self._condense(history, message)

        vector = self._llm.embed([query])[0]

        chunks: list[RetrievedChunk] = self._store.search(
            vector,
            self._top_k,
        )

        

        # 4. Build a numbered context block from the retrieved chunks.
        context = (
            "\n\n".join(
                f"[{i + 1}] {c.text}"
                for i, c in enumerate(chunks)
            )
            or "(no documents found)"
        )

        # 5. Generate the answer with the full chat history
        # so replies stay conversational.
        messages = [
            {
                "role": "system",
                "content": _ANSWER_PROMPT.format(context=context),
            },
            *history,
            {
                "role": "user",
                "content": message,
            },
        ]

        reply = self._llm.chat(messages)

        sources = [
            SourceChunk(
                document_id=c.document_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                score=c.score,
            )
            for c in chunks
        ]

        return reply, sources