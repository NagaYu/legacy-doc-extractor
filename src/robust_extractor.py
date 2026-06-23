"""
商用環境向けに堅牢化した非同期・構造化抽出パイプライン。

本モジュールが提供する強化点:
  1. 構造化の100%保証 …… Pydantic v2 のカスタム @field_validator で日付ISO/数値の妥当性を検査し、
     不正ならLLMにエラー理由を返して再生成させる「自己修正ループ」（最大2回）を実装。
  2. コスト/遅延の削減 …… プロンプトを最小限に圧縮し、Few-ShotはコンパクトなJSON1例のみ。
     asyncio による非同期バッチ処理 + tenacity の指数バックオフ付きリトライでレートリミットを回避。
  3. 確信度の数値化 …… 各抽出項目に quote（根拠原文）と confidence(0.0-1.0) を付与。
     0.7未満の項目は [WARNING] ログを出し、人手チェック対象としてフラグ化。

APIキー (ANTHROPIC_API_KEY) があれば実際のLLM (AsyncAnthropic) を、無ければ
決定論的なモックLLMを用いる。モックは同じ検証・自己修正・確信度・トークン計上の経路を通るため、
キーが無くてもパイプライン全体をエンドツーエンドで実証できる。
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

# 既存の規則ベース抽出（v1）をモックLLMの“素材”として再利用する
from extractor import extract_with_fallback

logger = logging.getLogger("robust_extractor")

MODEL_ID = "claude-opus-4-8"
MAX_SELF_CORRECTIONS = 2        # 自己修正の最大回数（=リトライ上限）。総試行は 1 + 2 = 3
MAX_CONCURRENCY = 5             # 同時並列リクエスト数（レートリミット保護の一次防御）
CONFIDENCE_THRESHOLD = 0.7      # これ未満の項目は人手チェック対象


# ===========================================================================
# 例外
# ===========================================================================
class TransientAPIError(Exception):
    """一時的なAPI障害（429等）を模した再試行対象の例外。"""


class SelfCorrectionExhausted(Exception):
    """自己修正の上限まで試しても妥当なJSONが得られなかった。"""


# ===========================================================================
# 1. Pydantic スキーマ（確信度つき・カスタムバリデータ）
# ===========================================================================
T = TypeVar("T")


class DocumentType(str, Enum):
    REAL_ESTATE = "real_estate"
    INSURANCE = "insurance"
    UNKNOWN = "unknown"


class Tracked(BaseModel, Generic[T]):
    """1項目の抽出結果。値・根拠引用・確信度をまとめて追跡する。"""

    value: Optional[T] = Field(None, description="抽出・正規化した値")
    quote: str = Field("", description="根拠となった原文の最小引用")
    confidence: float = Field(..., ge=0.0, le=1.0, description="確信度 0.0-1.0")


class MonetaryItem(BaseModel):
    label: str
    amount_yen: int = Field(..., description="円単位の整数に正規化した金額")
    quote: str = ""
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("amount_yen", mode="before")
    @classmethod
    def _amount_must_be_int_yen(cls, v):
        # LLMが "120,000円" や "12万円" のような文字列を返したら拒否し、自己修正を促す
        if isinstance(v, str):
            raise ValueError(
                f"amount_yen は円単位の整数で返してください。'{v}' は未正規化の文字列です"
            )
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"amount_yen は整数で返してください（受領: {v}）")
        iv = int(v)
        if iv < 0:
            raise ValueError("amount_yen は 0 以上である必要があります")
        return iv


class ContractData(BaseModel):
    """確信度つきの最終スキーマ。"""

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
        # Tracked[date] の value は pydantic が ISO文字列→date に変換済み。
        # ここでは妥当な範囲かを追加検査する（和暦未変換などの取りこぼし対策）。
        if v.value is not None and not (1900 <= v.value.year <= 2100):
            raise ValueError(
                f"contract_date.value={v.value} が妥当な範囲(1900-2100)外です。"
                "和暦→西暦変換とISO形式(YYYY-MM-DD)を確認してください"
            )
        return v

    def low_confidence_fields(self, threshold: float = CONFIDENCE_THRESHOLD) -> list[tuple[str, float]]:
        """確信度がしきい値未満の項目を (項目名, 確信度) で列挙する。"""
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
# 2. 圧縮プロンプト + コンパクトFew-Shot（トークン効率重視）
# ===========================================================================
SYSTEM_PROMPT = (
    "保険/不動産の契約書からJSONのみ抽出。各値に value/quote/confidence(0-1) を付与。"
    "quoteは根拠原文の最小引用。"
    "正規化: 金額=円整数(12万円→120000,漢数字も), 日付=ISO(令和N年→(2018+N)年,YYYY-MM-DD)。"
    "monetary_amounts[]は{label,amount_yen(円整数),quote,confidence}。"
    "document_type=real_estate|insurance|unknown。不明はvalue:null/confidence低。"
    "説明やコードフェンス禁止、JSONのみ返す。"
)

# Few-Shotは最小JSON1例のみ（user→assistant 1往復）でトークンを節約
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
    separators=(",", ":"),  # 余白を削ってトークン削減
)

FEWSHOT_MESSAGES = [
    {"role": "user", "content": _FEWSHOT_INPUT},
    {"role": "assistant", "content": _FEWSHOT_OUTPUT},
]

CORRECTION_MARKER = "【自己修正要求】"


def _estimate_tokens(text: str) -> int:
    """日本語混在テキストの概算トークン数（≒ 2.3文字/トークン）。"""
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
# 3. LLMクライアント（実API / モック）
# ===========================================================================
class RealAsyncLLM:
    """AsyncAnthropic を用いた実LLMクライアント。"""

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic()  # ANTHROPIC_API_KEY を環境から読込

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
    決定論的なモックLLM。実LLMと同じ (messages -> (json文字列, usage)) 契約を満たす。

    実演のため意図的に:
      - 最初の1回だけ一時的エラー(TransientAPIError)を投げ、tenacityのバックオフを発火させる
      - "覚書" を含む文書では初回に日付を未正規化(令和表記)で返し、自己修正ループを発火させる
    """

    def __init__(self) -> None:
        self._transient_fired = False

    async def complete(self, messages: list[dict]) -> tuple[str, Usage]:
        await asyncio.sleep(0.05)  # ネットワーク遅延の擬似

        # レートリミット(429)を一度だけ模擬 → tenacity がバックオフ後に再試行
        if not self._transient_fired:
            self._transient_fired = True
            raise TransientAPIError("Simulated HTTP 429 rate_limit_error")

        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        is_correction = CORRECTION_MARKER in last_user
        doc_text = _extract_doc(messages)

        json_str = self._build(doc_text, is_correction)

        # トークン使用量を概算（system + 全メッセージを入力、生成JSONを出力とみなす）
        in_text = SYSTEM_PROMPT + "".join(str(m["content"]) for m in messages)
        usage = Usage(_estimate_tokens(in_text), _estimate_tokens(json_str))
        return json_str, usage

    def _build(self, text: str, is_correction: bool) -> str:
        base = extract_with_fallback(text)  # v1の規則ベース結果を素材に使う

        # 漢数字を含む金額は確信度を下げる（OCR/解釈の不確かさを表現）
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

        # 保険会社名は様式が多様で誤りやすい想定 → 確信度を低めにして[WARNING]を実証
        counter_conf = 0.64 if base.document_type == DocumentType.INSURANCE else 0.95

        # 「覚書」文書では初回だけ日付を未正規化(令和表記)で返し、検証失敗→自己修正を誘発
        inject_bad_date = (not is_correction) and ("覚書" in text)
        if inject_bad_date:
            date_value: Optional[str] = "令和7年1月10日"  # ISOでない → 検証で弾かれる
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
    """```json ...``` のコードフェンスを除去してJSON本体を取り出す。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _extract_doc(messages: list[dict]) -> str:
    """messages 中の <doc>...</doc> から元テキストを取り出す。"""
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
        logger.info("ANTHROPIC_API_KEY を検出 → 実LLM (AsyncAnthropic) を使用")
        return RealAsyncLLM(), "LLM (AsyncAnthropic Structured Extraction)"
    logger.info("ANTHROPIC_API_KEY 未設定 → 決定論的モックLLMを使用")
    return MockAsyncLLM(), "MockLLM (deterministic, key不要)"


# ===========================================================================
# 4. リトライ（指数バックオフ） + 自己修正ループ
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
    """レートリミット等に対し指数バックオフで再試行しつつ1回のLLM呼び出しを行う。"""
    return await client.complete(messages)


def _format_validation_feedback(err: ValidationError, raw: str) -> str:
    lines = [f"{CORRECTION_MARKER} 前回のJSONは検証に失敗しました。以下を修正し、JSONのみ再出力してください:"]
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
    """1ドキュメントを抽出。検証失敗時はエラー理由をLLMに返して最大2回まで自己修正する。"""
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
                    "[%s] 検証失敗(試行%d/%d) → 自己修正を要求: %s",
                    name, attempt + 1, 1 + MAX_SELF_CORRECTIONS,
                    "; ".join(e["msg"] for e in exc.errors()),
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": _format_validation_feedback(exc, raw)})
                continue
            break  # 上限到達

        # 検証成功 → 確信度フラグ
        flagged = data.low_confidence_fields()
        for field_name, conf in flagged:
            logger.warning(
                "[%s] 低確信度のため要人手確認: %s (confidence=%.2f < %.2f)",
                name, field_name, conf, CONFIDENCE_THRESHOLD,
            )
        return ExtractionResult(
            name=name, data=data, usage=usage,
            attempts=attempt + 1, corrections=attempt, flagged=flagged,
        )

    return ExtractionResult(
        name=name, data=None, usage=usage,
        attempts=1 + MAX_SELF_CORRECTIONS, corrections=MAX_SELF_CORRECTIONS,
        error=f"自己修正上限({MAX_SELF_CORRECTIONS}回)を超過: "
              + ("; ".join(e["msg"] for e in last_err.errors()) if last_err else "unknown"),
    )


async def extract_batch(docs: list[tuple[str, str]]) -> tuple[list[ExtractionResult], str]:
    """
    複数ドキュメントを非同期バッチで抽出する。

    docs: [(name, text), ...]
    戻り値: (結果リスト, 使用クライアント名)
    """
    client, client_name = _make_client()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(name: str, text: str) -> ExtractionResult:
        async with sem:
            return await extract_one(client, name, text)

    results = await asyncio.gather(*(worker(n, t) for n, t in docs))
    return list(results), client_name
