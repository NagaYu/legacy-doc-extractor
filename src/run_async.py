"""
Hardened async batch runner script (v2).

  1. Load sample_contract_*.txt from data/
  2. Extract structured data concurrently with asyncio (self-correction loop + exponential-backoff retries)
  3. Emit [WARNING] for fields with confidence < 0.7 and flag them for human review
  4. Save to output/extracted_data_v2.json
  5. Print a summary of processing time / token consumption / parse success rate
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from robust_extractor import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    Usage,
    extract_batch,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
SEP = "═" * 78


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s",
    )


async def main() -> int:
    _setup_logging()

    samples = sorted(DATA_DIR.glob("sample_contract_*.txt"))
    if not samples:
        print(f"[error] テストデータがありません: {DATA_DIR} （先に data/generate_data.py を実行）")
        return 1

    docs = [(p.name, p.read_text(encoding="utf-8")) for p in samples]

    print(SEP)
    print(f"🚀 非同期バッチ抽出を開始（{len(docs)} 件 / 最大並列数あり）")
    print(SEP)

    t0 = time.perf_counter()
    results, client_name = await extract_batch(docs)
    elapsed = time.perf_counter() - t0

    # Format and save the results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    total_usage = Usage()
    success = 0
    total_corrections = 0
    total_flags = 0

    for r in results:
        total_usage.add(r.usage)
        total_corrections += r.corrections
        if r.ok:
            success += 1
            total_flags += len(r.flagged)
            payload.append({
                "source_file": r.name,
                "attempts": r.attempts,
                "self_corrections": r.corrections,
                "tokens": {"input": r.usage.input_tokens, "output": r.usage.output_tokens},
                "needs_human_review": [
                    {"field": f, "confidence": round(c, 2)} for f, c in r.flagged
                ],
                "data": r.data.model_dump(mode="json"),
            })
        else:
            payload.append({
                "source_file": r.name,
                "attempts": r.attempts,
                "error": r.error,
            })

    out_path = OUTPUT_DIR / "extracted_data_v2.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print per-document highlights
    for r in results:
        print()
        print(f"📄 {r.name}")
        if r.ok:
            d = r.data
            print(f"   状態        : ✅ 検証通過（試行 {r.attempts} 回 / 自己修正 {r.corrections} 回）")
            print(f"   契約者      : {d.contractor_name.value!r}  (conf={d.contractor_name.confidence:.2f})")
            print(f"   相手方      : {d.counterparty_name.value!r}  (conf={d.counterparty_name.confidence:.2f})")
            dv = d.contract_date.value.isoformat() if d.contract_date.value else None
            print(f"   契約日(ISO) : {dv}  (conf={d.contract_date.confidence:.2f})")
            for m in d.monetary_amounts:
                print(f"     ・{m.label}: {m.amount_yen:,}円  (conf={m.confidence:.2f}, 原文='{m.quote}')")
            if r.flagged:
                fl = ", ".join(f"{f}({c:.2f})" for f, c in r.flagged)
                print(f"   ⚠ 要人手確認: {fl}")
        else:
            print(f"   状態        : ❌ 失敗 — {r.error}")

    # Summary
    rate = (success / len(results) * 100) if results else 0.0
    print()
    print(SEP)
    print("📊 バッチ処理サマリー")
    print(SEP)
    print(f"  使用クライアント   : {client_name}")
    print(f"  処理件数           : {len(results)}")
    print(f"  パース成功率       : {success}/{len(results)}  ({rate:.0f}%)")
    print(f"  処理時間           : {elapsed:.2f} 秒  (1件あたり {elapsed/len(results):.2f} 秒)")
    print(f"  トークン消費量     : 合計 {total_usage.total:,}  "
          f"(入力 {total_usage.input_tokens:,} / 出力 {total_usage.output_tokens:,})")
    print(f"  自己修正の総回数   : {total_corrections}")
    print(f"  要人手確認フラグ数 : {total_flags}  (確信度 < {CONFIDENCE_THRESHOLD})")
    print(f"  保存先             : {out_path}")
    print()

    return 0 if success == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
