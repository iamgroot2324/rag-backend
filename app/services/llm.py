"""LLM and embedding client.

Embeddings are generated locally using Sentence Transformers. Chat completions go to
any OpenAI-compatible server (Ollama locally by default, or OpenAI itself) and are
configured through settings, so switching provider needs no code change.
"""

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.core.config import Settings


class LLMClient:
    """Provides local embeddings and chat completions."""

    def __init__(self, settings: Settings) -> None:
        # Chat client: points at Ollama by default (see llm_base_url in settings).
        self._chat_client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=120,  # local models can be slow, especially on the first call
        )
        self._chat_model = settings.llm_chat_model

        # Local embedding model.
        self._embedder = SentenceTransformer(settings.local_embedding_model)

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Generate embeddings locally in batches."""
        if not texts:
            return []

        vectors: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            embeddings = self._embedder.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            vectors.extend(embedding.tolist() for embedding in embeddings)

        return vectors

    def chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        """Run a chat completion; with json_mode the model must return a JSON object."""
        kwargs: dict[str, object] = {
            "model": self._chat_model,
            "messages": messages,
            "temperature": 0.2,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._chat_client.chat.completions.create(**kwargs)  # type: ignore[arg-type]

        return resp.choices[0].message.content or ""