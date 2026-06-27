import json

from openai import AsyncOpenAI

from api.config import get_settings


def _default_client() -> AsyncOpenAI:
    """Construct the real OpenAI client from application settings.

    Called lazily from ``analyze_journal_entry`` so tests can inject a
    ``MockAsyncOpenAI`` without ever triggering this code path.
    """
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: AsyncOpenAI | None = None,
) -> dict:

    if client is None:
        client = _default_client()

    response = await client.chat.completions.create(
        model=get_settings().openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a journal analyst. Analyze the journal entry and respond "
                    "with a JSON object containing exactly these keys: "
                    "sentiment (one of: positive,negative,neutral), "
                    "summary (a brief summary string), "
                    "topics (a list of strings)."
                ),
            },
            {"role": "user", "content": entry_text},
        ],
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("LLm returned no entry")
    parsed = json.loads(raw)

    return {
        "entry_id": entry_id,
        "sentiment": parsed["sentiment"],
        "summary": parsed["summary"],
        "topics": parsed["topics"],
    }
    """Analyze a journal entry using an OpenAI-compatible LLM.

    Args:
        entry_id: ID of the entry being analyzed (pass through to the result).
        entry_text: Combined work + struggle + intention text.
        client: OpenAI client. If None, a default one is constructed from
            application settings. Tests pass in a MockAsyncOpenAI here; production code
            in the router calls this with no ``client`` argument.

    Returns:
        A dict matching AnalysisResponse:
            {
                "entry_id":  str,
                "sentiment": str,   # "positive" | "negative" | "neutral"
                "summary":   str,
                "topics":    list[str],
            }

    TODO (Task 4):
      1. If ``client is None``, call ``_default_client()`` to construct one.
      2. Build a messages list that includes ``entry_text`` somewhere
         (the unit tests check that the entry text reaches the LLM).
      3. Call ``client.chat.completions.create(...)`` with a model name
         (use ``get_settings().openai_model`` — defaults to "gpt-4o-mini").
      4. Parse the assistant's JSON response with ``json.loads()``.
      5. Return a dict with ``entry_id``, ``sentiment``, ``summary``, ``topics``.
    """
    raise NotImplementedError(
        "Task 4: implement analyze_journal_entry using the openai SDK. "
        "See tests/test_llm_service.py for the test contract."
    )
