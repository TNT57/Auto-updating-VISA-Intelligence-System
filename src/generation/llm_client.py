"""
LLM client with Groq integration (free, instant inference).

Supports easy switching between Groq and OpenAI via configuration.
Includes retry logic and streaming support.
"""

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.generation.prompt_templates import (
    FOLLOW_UP_TEMPLATE,
    GROUNDING_PROMPT,
    RAG_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from src.utils.config import settings


class LLMClient:
    """LLM client with abstraction for multiple providers."""

    def __init__(self, provider: str | None = None):
        self.provider = provider or self._detect_provider()
        self._client = None
        self._setup_client()

    def _detect_provider(self) -> str:
        """Auto-detect which LLM provider to use based on available API keys."""
        if settings.groq_api_key:
            return "groq"
        elif settings.openai_api_key:
            return "openai"
        else:
            logger.warning(
                "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in .env"
            )
            return "groq"  # Default, will fail gracefully

    def _setup_client(self) -> None:
        """Initialize the LLM client based on provider."""
        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
            self._model = "llama-3.3-70b-versatile"
            logger.info("LLM client initialized: Groq ({})", self._model)

        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key)
            self._model = "gpt-4o-mini"
            logger.info("LLM client initialized: OpenAI ({})", self._model)

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt override
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower = more deterministic)

        Returns:
            Generated text string
        """
        system = system_prompt or SYSTEM_PROMPT

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = response.choices[0].message.content
            logger.debug(
                "LLM response generated ({} tokens)",
                response.usage.completion_tokens if response.usage else "?",
            )
            return result

        except Exception as e:
            logger.error("LLM generation failed: {}", e)
            raise

    def answer_question(
        self,
        question: str,
        context: str,
        chat_history: str | None = None,
    ) -> str:
        """
        Answer a question using RAG (Retrieval-Augmented Generation).

        Args:
            question: User's question
            context: Retrieved context from vector store
            chat_history: Optional conversation history text

        Returns:
            Generated answer with source citations
        """
        if chat_history:
            prompt = FOLLOW_UP_TEMPLATE.format(
                chat_history=chat_history,
                context=context,
                question=question,
            )
        else:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )
        return self.generate(prompt=prompt)

    def stream_answer(
        self,
        question: str,
        context: str,
        chat_history: str | None = None,
    ):
        """
        Stream a RAG response token by token (for real-time UI).

        Supports conversation memory via chat_history parameter.

        Yields chunks of text as they are generated.
        """
        if chat_history:
            prompt = FOLLOW_UP_TEMPLATE.format(
                chat_history=chat_history,
                context=context,
                question=question,
            )
        else:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error("LLM streaming failed: {}", e)
            yield f"Error generating response: {e}"

    def verify_grounding(
        self,
        answer: str,
        context: str,
    ) -> tuple[str, str]:
        """
        Verify whether the generated answer is grounded in the context.

        Uses a lightweight LLM call to check if the answer's claims
        are supported by the retrieved documents.

        Returns:
            Tuple of (verdict, explanation) where verdict is one of:
            "GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED"
        """
        # The checker must see everything the answer was written from. This
        # was truncated to 3,000 characters, and a 5-chunk context runs to
        # ~5,500 — so a correct answer citing the later chunks was reported
        # PARTIALLY_GROUNDED because the evidence had been cut off. A false
        # "unsupported" on the one safety mechanism is worse than the tokens.
        # 40,000 characters is roughly 10k tokens, comfortably inside the
        # model's window while still bounding a pathological context.
        MAX_GROUNDING_CHARS = 40_000
        if len(context) > MAX_GROUNDING_CHARS:
            logger.warning(
                "Grounding context truncated {} -> {} chars; verdict may be "
                "conservative", len(context), MAX_GROUNDING_CHARS,
            )
        prompt = GROUNDING_PROMPT.format(
            context=context[:MAX_GROUNDING_CHARS],
            answer=answer,
        )
        try:
            raw = self.generate(
                prompt=prompt,
                system_prompt="You are a concise fact-checker. Respond with only the verdict and one explanation sentence.",
                max_tokens=150,
                temperature=0.0,
            )
            # Parse verdict
            raw_upper = raw.upper().strip()
            if raw_upper.startswith("GROUNDED"):
                verdict = "GROUNDED"
            elif raw_upper.startswith("PARTIALLY_GROUNDED"):
                verdict = "PARTIALLY_GROUNDED"
            elif raw_upper.startswith("UNGROUNDED"):
                verdict = "UNGROUNDED"
            else:
                verdict = "UNKNOWN"
            logger.info("Grounding check: {} — {}", verdict, raw.strip()[:80])
            return verdict, raw.strip()
        except Exception as e:
            logger.warning("Grounding verification failed: {}", e)
            return "UNKNOWN", str(e)
