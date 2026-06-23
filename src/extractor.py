"""
非構造化な契約書/約款テキストから、Pydanticモデルに準拠した構造化データを抽出する。

抽出方式は2系統:
  1. LLM抽出 (extract_with_llm) … Anthropic の Structured Outputs を使用。
     環境変数 ANTHROPIC_API_KEY が設定されている場合に利用される。高精度。
  2. 規則ベース抽出 (extract_with_fallback) … 正規表現による決定論的フォールバック。
     APIキーが無い／API呼び出しに失敗した場合でもデモがエンドツーエンドで動くようにする。

呼び出し側は extract_contract() を使えば、利用可能な方式が自動で選択される。
"""

from __future__ import annotations

import os
import re
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ===========================================================================
# 1. Pydantic スキーマ定義
# ===========================================================================
class DocumentType(str, Enum):
    """ドキュメント種別。"""

    REAL_ESTATE = "real_estate"   # 不動産（賃貸借契約など）
    INSURANCE = "insurance"       # 保険（約款・契約内容）
    UNKNOWN = "unknown"


class MonetaryItem(BaseModel):
    """正規化された金額項目。表記揺れを吸収して円単位の整数に統一する。"""

    label: str = Field(..., description="金額の名目（例: 月額家賃, 敷金, 保険金額, 保険料）")
    amount_yen: int = Field(..., description="円単位に正規化した金額（整数）。例: 12万円→120000")
    original_text: str = Field(
        ..., description="元テキストでの表記（例: '金弐拾肆萬円', '35万円', '120,000円'）"
    )


class ContractData(BaseModel):
    """契約書/約款から抽出する構造化データの最終スキーマ。"""

    document_type: DocumentType = Field(
        ..., description="ドキュメント種別。不動産=real_estate, 保険=insurance"
    )
    contractor_name: str = Field(
        ..., description="契約者（借主・保険契約者）の氏名または名称"
    )
    counterparty_name: Optional[str] = Field(
        None, description="相手方（貸主・保険会社）の名称。不明な場合は null"
    )
    contract_date: Optional[date] = Field(
        None,
        description="契約日/契約始期。和暦・西暦の混在を吸収しISO形式(YYYY-MM-DD)に正規化する。不明なら null",
    )
    monetary_amounts: list[MonetaryItem] = Field(
        default_factory=list,
        description="契約に関わる金額一覧。円・万円・漢数字の表記揺れを円単位に正規化する",
    )
    key_clauses: list[str] = Field(
        default_factory=list,
        description="重要な特約・免責事項を1項目1文で簡潔に要約したもの",
    )
    summary: str = Field(..., description="契約全体の1〜2文の要約")


# ===========================================================================
# 2. LLM抽出（Anthropic Structured Outputs）
# ===========================================================================
MODEL_ID = "claude-opus-4-8"

SYSTEM_PROMPT = """\
あなたは保険・不動産業界のレガシーな契約書・約款を読み解く専門のデータ抽出エンジンです。
表記が不統一で冗長な日本語テキストから、指定スキーマに厳密に従って情報を抽出してください。

レガジー文書ゆえの「表記揺れ」を必ず正規化してください:

【金額の正規化】amount_yen は必ず「円単位の整数」にすること。
  - "120,000円" → 120000
  - "12万円" / "金壱拾弐萬円" → 120000
  - "35万円" → 350000
  - "金壱千万円" / "10,000,000円" → 10000000
  - 漢数字（壱弐参…萬…）やカンマ・全角数字も解釈して算用数字に直すこと。
  - original_text には元の表記をそのまま残すこと。

【日付の正規化】contract_date は ISO形式(YYYY-MM-DD)にすること。
  - 和暦は西暦へ変換する。令和N年 = (2018+N)年。例: 令和6年4月1日 → 2024-04-01
  - 平成N年 = (1988+N)年。
  - 「契約日」「契約始期」に相当する日付を採用すること。

【特約・免責】key_clauses には重要な特約事項・免責事項を、それぞれ簡潔な1文に要約して列挙すること。

不明な項目は無理に埋めず、任意項目は null / 空配列にしてください。
"""


def extract_with_llm(text: str) -> ContractData:
    """Anthropic の Structured Outputs を用いて構造化抽出する。"""
    import anthropic  # 遅延 import（フォールバック時に依存を強制しない）

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY を環境変数から読み込む

    response = client.messages.parse(
        model=MODEL_ID,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"以下の契約書テキストから情報を抽出してください。\n\n---\n{text}\n---",
            }
        ],
        output_format=ContractData,
    )
    parsed = response.parsed_output
    if parsed is None:
        raise ValueError("LLMがスキーマに準拠した出力を返しませんでした。")
    return parsed


# ===========================================================================
# 3. 規則ベース抽出（決定論的フォールバック）
# ===========================================================================
_KANJI_DIGIT = {
    "〇": 0, "零": 0, "一": 1, "壱": 1, "二": 2, "弐": 2, "三": 3, "参": 3,
    "四": 4, "肆": 4, "五": 5, "伍": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_KANJI_UNIT_SMALL = {"十": 10, "拾": 10, "百": 100, "千": 1000}
_KANJI_UNIT_LARGE = {"万": 10_000, "萬": 10_000, "億": 100_000_000}


def _zen_to_han(s: str) -> str:
    """全角英数字・カンマを半角へ変換する。"""
    return s.translate(str.maketrans(
        "０１２３４５６７８９，．",
        "0123456789,.",
    ))


def parse_kanji_number(s: str) -> Optional[int]:
    """漢数字（壱千万円 等）を整数に変換する。失敗時は None。"""
    s = s.replace("金", "").replace("也", "").replace("円", "")
    s = s.replace("圓", "").strip()
    if not s:
        return None

    total = 0          # 確定済みの合計
    section = 0        # 「万」「億」までの区間値
    current = 0        # 直近の桁の値
    matched = False

    for ch in s:
        if ch in _KANJI_DIGIT:
            current = _KANJI_DIGIT[ch]
            matched = True
        elif ch in _KANJI_UNIT_SMALL:
            section += (current or 1) * _KANJI_UNIT_SMALL[ch]
            current = 0
            matched = True
        elif ch in _KANJI_UNIT_LARGE:
            section += current
            total += section * _KANJI_UNIT_LARGE[ch]
            section = 0
            current = 0
            matched = True
        else:
            # 想定外の文字が混ざる場合は漢数字ではないと判断
            return None

    result = total + section + current
    return result if matched else None


def normalize_amount(raw: str) -> Optional[int]:
    """金額表記（120,000円 / 12万円 / 金壱千万円 など）を円単位の整数へ正規化する。"""
    raw = _zen_to_han(raw)

    # パターンA: 算用数字 + 「万」(+「円」) 例: 35万円, 210万円
    m = re.search(r"([0-9,]+)\s*万\s*円?", raw)
    if m:
        return int(m.group(1).replace(",", "")) * 10_000

    # パターンB: 算用数字 + 「円」 例: 120,000円, 10,000,000円
    m = re.search(r"([0-9,]+)\s*円", raw)
    if m:
        return int(m.group(1).replace(",", ""))

    # パターンC: 漢数字主体 例: 金弐拾肆萬円, 金壱千万円
    val = parse_kanji_number(raw)
    return val


# 「ラベル ... 金額表記」を拾うための行ベースの手がかり
# より具体的なラベルを先に並べる（同一行に複数ラベルが出る場合の優先順位）
_AMOUNT_LABELS = ["月額家賃", "敷金", "礼金", "共益費", "保証金", "保険金額", "保険料", "賃料", "家賃"]


def _extract_amounts_fallback(text: str) -> list[MonetaryItem]:
    items: list[MonetaryItem] = []
    seen: set[tuple[str, int]] = set()

    # 金額らしき表記を文中から広く拾う
    amount_pattern = re.compile(
        r"(?:金)?\s*(?:[0-9０-９,，]+\s*万?\s*円|[〇零一壱二弐三参四肆五伍六七八九十拾百千万萬億]+\s*円|[〇零一壱二弐三参四肆五伍六七八九十拾百千万萬億]+萬円)"
    )

    for line in text.splitlines():
        for m in amount_pattern.finditer(line):
            raw = m.group(0).strip()
            amount = normalize_amount(raw)
            if amount is None or amount <= 0:
                continue
            # 行内のラベルを推定（最初に現れるラベル語）
            label = next((lab for lab in _AMOUNT_LABELS if lab in line), "金額")
            key = (label, amount)
            if key in seen:
                continue
            seen.add(key)
            items.append(MonetaryItem(label=label, amount_yen=amount, original_text=raw))
    return items


def _extract_date_fallback(text: str) -> Optional[date]:
    text = _zen_to_han(text)

    # 和暦: 令和N年M月D日
    m = re.search(r"令和\s*([0-9元]+)\s*年\s*([0-9]+)\s*月\s*([0-9]+)\s*日", text)
    if m:
        y = 1 if m.group(1) == "元" else int(m.group(1))
        return date(2018 + y, int(m.group(2)), int(m.group(3)))

    # 和暦: 平成N年M月D日
    m = re.search(r"平成\s*([0-9元]+)\s*年\s*([0-9]+)\s*月\s*([0-9]+)\s*日", text)
    if m:
        y = 1 if m.group(1) == "元" else int(m.group(1))
        return date(1988 + y, int(m.group(2)), int(m.group(3)))

    # 西暦: YYYY年M月D日
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    return None


def _detect_doc_type(text: str) -> DocumentType:
    if any(k in text for k in ["保険", "約款", "保険金", "保険料", "告知義務"]):
        return DocumentType.INSURANCE
    if any(k in text for k in ["賃貸", "家賃", "敷金", "礼金", "借主", "貸主", "賃料"]):
        return DocumentType.REAL_ESTATE
    return DocumentType.UNKNOWN


def _extract_name_fallback(text: str, patterns: list[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip("　 　")
    return None


def _extract_clauses_fallback(text: str) -> list[str]:
    """特約・免責に関連する条項を簡易抽出する。

    構造化された短い行はそのまま、改行の無い長い一段落は文（。区切り）に分割して
    キーワードを含む文だけを拾う。
    """
    keywords = ["クリーニング", "ペット", "転貸", "又貸し", "原状回復", "中途解約",
                "免責", "告知義務", "解約", "責任開始", "解除"]
    clauses: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or "≪" in line or "要旨" in line:  # 見出し行は除外
            continue
        # 長い一段落は文単位に分割（覚書のようなベタ書き対策）
        segments = [s for s in line.split("。") if s] if len(line) > 80 else [line]
        long_para = len(line) > 80
        for seg in segments:
            # 先頭の番号・記号（（１）/ ・ / 【】 等）を除去
            s = re.sub(r"^[\s　・（）()【】0-9０-９]+", "", seg).strip("　 ")
            if not s or len(s) <= 6:
                continue
            if any(k in s for k in keywords):
                if long_para and not s.endswith("。"):
                    s += "。"
                if s not in clauses:
                    clauses.append(s)
    return clauses[:6]


def extract_with_fallback(text: str) -> ContractData:
    """API無しでも動く、正規表現ベースの決定論的抽出。"""
    doc_type = _detect_doc_type(text)

    contractor = _extract_name_fallback(text, [
        r"」という。）\s*は\s*[　 ]*([^\n。、（]+?)\s*とする",   # 「契約者」という。）は 山田 太郎 とする
        r"【ご契約者さま】\s*[　 ]*([^\n　]+?)\s*様",            # 【ご契約者さま】保戸田 花子 様
        r"借主は\s*([^\n（。]+?)\s*(?:（|であり|とする)",        # 借主は 田中商事株式会社（…
    ])
    counterparty = _extract_name_fallback(text, [
        r"貸主(?:は)?\s*[　 ]*([^\n。、（]+?)\s*とする",          # 貸主は 株式会社さくら不動産管理 とする
        r"【保険会社】\s*[　 ]*([^\n　]+)",                       # 【保険会社】あさひ生命保険相互会社
        r"貸主は\s*([^\n。]+?)\s*である",                        # 貸主は 野村ビルディング合同会社 である
    ])

    return ContractData(
        document_type=doc_type,
        contractor_name=contractor or "（抽出失敗）",
        counterparty_name=counterparty,
        contract_date=_extract_date_fallback(text),
        monetary_amounts=_extract_amounts_fallback(text),
        key_clauses=_extract_clauses_fallback(text),
        summary=f"{doc_type.value} 区分の契約書から規則ベースで抽出した構造化データ。",
    )


# ===========================================================================
# 4. ディスパッチャ
# ===========================================================================
def extract_contract(text: str) -> tuple[ContractData, str]:
    """
    利用可能な抽出方式を自動選択して構造化データを返す。

    戻り値: (ContractData, 使用した方式名)
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return extract_with_llm(text), "LLM (Anthropic Structured Outputs)"
        except Exception as exc:  # noqa: BLE001 - フォールバックに切替える
            print(f"  [warn] LLM抽出に失敗したためフォールバックします: {exc}")
    return extract_with_fallback(text), "規則ベース (正規表現フォールバック)"
