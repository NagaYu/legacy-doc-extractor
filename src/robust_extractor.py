"""
Production-hardened async, structured extraction pipeline.

Enhancements provided by this module:
  1. 100% structuring guarantee ...... Pydantic v2 custom @field_validator checks ISO-date / numeric
     validity; on failure it returns the error reason to the LLM and re-generates via a
     "self-correction loop" (up to 2 retries).
  2. Cost / latency reduction ...... Compresses the prompt to a minimum and uses only a single compact
     JSON few-shot example. asyncio async batching + tenacity exponential-backoff retries dodge rate limits.
  3. Confidence quantification ...... Each extracted field carries a quote (supporting source text) and a
     confidence (0.0-1.0). Fields below 0.7 emit a [WARNING] log and are flagged for human review.

If an API key (ANTHROPIC_API_KEY) is present it uses a real LLM (AsyncAnthropic); otherwise it uses a
deterministic mock LLM. The mock goes through the same validation / self-correction / confidence /
token-accounting path, so the whole pipeline can be demonstrated end to end even without a key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# Reuse the existing rule-based extraction (v1) as the mock LLM's "raw material"
from extractor import extract_with_fallback

logger = logging.getLogger("robust_extractor")

MODEL_ID = "claude-opus-4-8"
MAX_SELF_CORRECTIONS = 2        # max self-corrections (= retry limit). Total attempts = 1 + 2 = 3
MAX_CONCURRENCY = 5             # max concurrent requests (first line of defense against rate limits)
CONFIDENCE_THRESHOLD = 0.7      # fields below this are flagged for human review


# ===========================================================================
# Exceptions
# ===========================================================================
class TransientAPIError(Exception):
    """A retryable exception simulating a transient API failure (e.g. 429)."""


class SelfCorrectionExhausted(Exception):
    """Could not obtain valid JSON even after exhausting the self-correction limit."""


# ===========================================================================
# 1. Pydantic schema (with confidence + custom validators)
# ===========================================================================
T = TypeVar("T")


class DocumentType(str, Enum):
    REAL_ESTATE = "real_estate"
    INSURANCE = "insurance"
    UNKNOWN = "unknown"


class Tracked(BaseModel, Generic[T]):
    """Extraction result for one field. Tracks the value, supporting quote, and confidence together."""

    value: Optional[T] = Field(None, description="Extracted / normalized value")
    quote: str = Field("", description="Minimal supporting quote from the source text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")


class MonetaryItem(BaseModel):
    label: str
    amount_yen: int = Field(..., description="Amount normalized to an integer in yen")
    quote: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("amount_yen", mode="before")
    @classmethod
    def _amount_must_be_int_yen(cls, v):
        # If the LLM returns a string like "120,000円" or "12万円", reject it to trigger self-correction
        if isinstance(v, str):
            raise ValueError(
                f"amount_yen must be an integer in yen; '{v}' is an un-normalized string"
            )
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"amount_yen must be an integer (received: {v})")
        iv = int(v)
        if iv < 0:
            raise ValueError("amount_yen must be >= 0")
        return iv


class ContractData(BaseModel):
    """Final schema with confidence."""

    document_type: DocumentType
    contractor_name: Tracked[str]
    counterparty_name: Tracked[str]
    contract_date: Tracked[date]
    monetary_amounts: list[MonetaryItem] = Field(default_factory=list)
    key_clauses: list[str] = Field(default_factory=list)
    summary: str

    @field_validator("contract_date")
    @classmethod
    def _date_must_be_iso_and_plausible(cls, v: "Tracked[date]") -> "Tracked[date]":
        # pydantic already converted Tracked[date].value from an ISO string to a date.
        # Here we additionally check it falls in a plausible range (catches un-converted Japanese eras, etc.).
        if v.value is not None and not (1900 <= v.value.year <= 2100):
            raise ValueError(
                f"contract_date.value={v.value} is outside the plausible range (1900-2100). "
                "Check the Japanese-era to Gregorian conversion and ISO format (YYYY-MM-DD)"
            )
        return v

    def low_confidence_fields(self, threshold: float = CONFIDENCE_THRESHOLD) -> list[tuple[str, float]]:
        """List fields whose confidence is below the threshold as (field_name, confidence)."""
        flagged: list[tuple[str, float]] = []
        for name in ("contractor_name", "counterparty_name", "contract_date"):
            tracked: Tracked = getattr(self, name)
            if tracked.confidence < threshold:
                flagged.append((name, tracked.confidence))
        for i, m in enumerate(self.monetary_amounts):
            if m.confidence < threshold:
                flagged.append((f"monetary_amounts[{i}]({m.label})", m.confidence))
        return flagged


# ===========================================================================
# 2. Compressed prompt + compact few-shot (optimized for token efficiency)
# ===========================================================================
SYSTEM_PROMPT = (
    "Extract JSON only from Japanese insurance/real-estate contracts. "
    "Give each value value/quote/confidence(0-1). quote = minimal supporting source quote. "
    "Normalize: amount = integer yen (12万円->120000, incl. kanji numerals), "
    "date = ISO (令和N年->(2018+N), YYYY-MM-DD). "
    "monetary_amounts[] = {label,amount_yen(int yen),quote,confidence}. "
    "document_type=real_estate|insurance|unknown. Unknown = value:null / low confidence. "
    "Keep extracted names/labels in their original Japanese. No prose or code fences; return JSON only."
)

# The few-shot is a single minimal JSON example (one user→assistant round trip) to save tokens
_FEWSHOT_INPUT = "<doc>\n借主は田中とする。賃料は月額5万円。契約は令和5年4月1日。\n</doc>"
_FEWSHOT_OUTPUT = json.dumps(
    {
        "document_type": "real_estate",
        "contractor_name": {"value": "田中", "quote": "借主は田中", "confidence": 0.95},
        "counterparty_name": {"value": None, "quote": "", "confidence": 0.3},
        "contract_date": {"value": "2023-04-01", "quote": "令和5年4月1日", "confidence": 0.97},
        "monetary_amounts": [
            {"label": "賃料", "amount_yen": 50000, "quote": "月額5万円", "confidence": 0.96}
        ],
        "key_clauses": [],
        "summary": "賃貸借契約",
    },
    ensure_ascii=False,
    separators=(",", ":"),  # trim whitespace to reduce tokens
)

FEWSHOT_MESSAGES = [
    {"role": "user", "content": _FEWSHOT_INPUT},
    {"role": "assistant", "content": _FEWSHOT_OUTPUT},
]

CORRECTION_MARKER = "[SELF-CORRECTION REQUEST]"


def _estimate_tokens(text: str) -> int:
    """Rough token count for mixed-Japanese text (≈ 2.3 chars/token)."""
    return max(1, math.ceil(len(text) / 2.3))


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


# ===========================================================================
# 3. LLM clients (real API / mock)
# ===========================================================================
class RealAsyncLLM:
    """Real LLM client using AsyncAnthropic."""

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

    async def complete(self, messages: list[dict]) -> tuple[str, Usage]:
        resp = await self._client.messages.create(
            model=MODEL_ID,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return _strip_json(text), usage


class MockAsyncLLM:
    """
    Deterministic mock LLM. Satisfies the same (messages -> (json string, usage)) contract as the real LLM.

    For demonstration purposes it deliberately:
      - raises a transient error (TransientAPIError) on the very first call to trigger tenacity backoff
      - for documents containing "覚書", returns an un-normalized date (Japanese-era notation) on the
        first attempt to trigger the self-correction loop
    """

    def __init__(self) -> None:
        self._transient_fired = False

    async def complete(self, messages: list[dict]) -> tuple[str, Usage]:
        await asyncio.sleep(0.05)  # simulate network latency

        # Simulate a rate limit (429) exactly once → tenacity retries after backoff
        if not self._transient_fired:
            self._transient_fired = True
            raise TransientAPIError("Simulated HTTP 429 rate_limit_error")

        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        is_correction = CORRECTION_MARKER in last_user
        doc_text = _extract_doc(messages)

        json_str = self._build(doc_text, is_correction)

        # Estimate token usage (treat system + all messages as input, the generated JSON as output)
        in_text = SYSTEM_PROMPT + "".join(str(m["content"]) for m in messages)
        usage = Usage(_estimate_tokens(in_text), _estimate_tokens(json_str))
        return json_str, usage

    def _build(self, text: str, is_correction: bool) -> str:
        base = extract_with_fallback(text)  # use the v1 rule-based result as raw material

        # Lower the confidence for amounts containing kanji numerals (expresses OCR/parse uncertainty)
        def amount_conf(orig: str) -> float:
            kanji = "〇零一壱二弐三参四肆五伍六七八九十拾百千万萬億"
            return 0.66 if any(c in orig for c in kanji) else 0.94

        monetary = [
            {
                "label": m.label,
                "amount_yen": m.amount_yen,
                "quote": m.original_text,
                "confidence": amount_conf(m.original_text),
            }
            for m in base.monetary_amounts
        ]

        # Insurer names vary in format and are error-prone → lower confidence to demonstrate [WARNING]
        counter_conf = 0.64 if base.document_type == DocumentType.INSURANCE else 0.95

        # For "覚書" documents, return an un-normalized date (era notation) on the first attempt only,
        # to induce validation failure → self-correction
        inject_bad_date = (not is_correction) and ("覚書" in text)
        if inject_bad_date:
            date_value: Optional[str] = "令和7年1月10日"  # not ISO → rejected by validation
            date_conf = 0.5
        else:
            date_value = base.contract_date.isoformat() if base.contract_date else None
            date_conf = 0.9 if base.contract_date else 0.3

        payload = {
            "document_type": base.document_type.value,
            "contractor_name": {
                "value": base.contractor_name,
                "quote": base.contractor_name,
                "confidence": 0.96,
            },
            "counterparty_name": {
                "value": base.counterparty_name,
                "quote": base.counterparty_name or "",
                "confidence": counter_conf if base.counterparty_name else 0.3,
            },
            "contract_date": {
                "value": date_value,
                "quote": _date_quote(text),
                "confidence": date_conf,
            },
            "monetary_amounts": monetary,
            "key_clauses": base.key_clauses,
            "summary": base.summary,
        }
        return json.dumps(payload, ensure_ascii=False)


def _strip_json(text: str) -> str:
    """Strip a ```json ...``` code fence and return the JSON body."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_doc(messages: list[dict]) -> str:
    """Extract the source text from <doc>...</doc> within messages."""
    for m in reversed(messages):
        content = m.get("content", "")
        if isinstance(content, str):
            mt = re.search(r"<doc>\n?(.*?)\n?</doc>", content, re.DOTALL)
            if mt:
                return mt.group(1)
    return ""


def _date_quote(text: str) -> str:
    m = re.search(r"(令和|平成)?\s*[0-9０-９元]+\s*年\s*[0-9０-９]+\s*月\s*[0-9０-９]+\s*日", text)
    return m.group(0).strip() if m else ""


def _make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Detected ANTHROPIC_API_KEY -> using real LLM (AsyncAnthropic)")
        return RealAsyncLLM(), "LLM (AsyncAnthropic Structured Extraction)"
    logger.info("ANTHROPIC_API_KEY not set -> using deterministic mock LLM")
    return MockAsyncLLM(), "MockLLM (deterministic, no key required)"


# ===========================================================================
# 4. Retry (exponential backoff) + self-correction loop
# ===========================================================================
def _retryable_exceptions():
    excs: tuple = (TransientAPIError,)
    try:
        import anthropic

        excs += (
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        )
    except Exception:  # noqa: BLE001
        pass
    return excs


@retry(
    retry=retry_if_exception_type(_retryable_exceptions()),
    wait=wait_random_exponential(multiplier=0.5, max=8),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _invoke_with_backoff(client, messages: list[dict]) -> tuple[str, Usage]:
    """Make one LLM call, retrying with exponential backoff against rate limits, etc."""
    return await client.complete(messages)


def _format_validation_feedback(err: ValidationError, raw: str) -> str:
    lines = [f"{CORRECTION_MARKER} The previous JSON failed validation. Fix the following and return JSON only:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        lines.append(f"- {loc}: {e['msg']}")
    return "\n".join(lines)


@dataclass
class ExtractionResult:
    name: str
    data: Optional[ContractData]
    usage: Usage
    attempts: int
    corrections: int
    flagged: list[tuple[str, float]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.data is not None


async def extract_one(client, name: str, text: str) -> ExtractionResult:
    """Extract one document. On validation failure, feed the reason back to the LLM and self-correct up to 2 times."""
    messages = list(FEWSHOT_MESSAGES) + [{"role": "user", "content": f"<doc>\n{text}\n</doc>"}]
    usage = Usage()
    last_err: Optional[ValidationError] = None

    for attempt in range(1 + MAX_SELF_CORRECTIONS):
        raw, call_usage = await _invoke_with_backoff(client, messages)
        usage.add(call_usage)
        try:
            data = ContractData.model_validate_json(raw)
        except ValidationError as exc:
            last_err = exc
            if attempt < MAX_SELF_CORRECTIONS:
                logger.info(
                    "[%s] validation failed (attempt %d/%d) -> requesting self-correction: %s",
                    name, attempt + 1, 1 + MAX_SELF_CORRECTIONS,
                    "; ".join(e["msg"] for e in exc.errors()),
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": _format_validation_feedback(exc, raw)})
                continue
            break  # limit reached

        # Validation succeeded → confidence flags
        flagged = data.low_confidence_fields()
        for field_name, conf in flagged:
            logger.warning(
                "[%s] low confidence, needs human review: %s (confidence=%.2f < %.2f)",
                name, field_name, conf, CONFIDENCE_THRESHOLD,
            )
        return ExtractionResult(
            name=name, data=data, usage=usage,
            attempts=attempt + 1, corrections=attempt, flagged=flagged,
        )

    return ExtractionResult(
        name=name, data=None, usage=usage,
        attempts=1 + MAX_SELF_CORRECTIONS, corrections=MAX_SELF_CORRECTIONS,
        error=f"Exceeded self-correction limit ({MAX_SELF_CORRECTIONS}): "
              + ("; ".join(e["msg"] for e in last_err.errors()) if last_err else "unknown"),
    )


async def extract_batch(docs: list[tuple[str, str]]) -> tuple[list[ExtractionResult], str]:
    """
    Extract multiple documents as an async batch.

    docs: [(name, text), ...]
    Returns: (list of results, name of the client used)
    """
    client, client_name = _make_client()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(name: str, text: str) -> ExtractionResult:
        async with sem:
            return await extract_one(client, name, text)

    results = await asyncio.gather(*(worker(n, t) for n, t in docs))
    return list(results), client_name
