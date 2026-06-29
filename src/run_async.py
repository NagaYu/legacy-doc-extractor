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
        print(f"[error] No test data found: {DATA_DIR} (run data/generate_data.py first)")
        return 1

    docs = [(p.name, p.read_text(encoding="utf-8")) for p in samples]

    print(SEP)
    print(f"🚀 Starting async batch extraction ({len(docs)} docs / bounded concurrency)")
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
            print(f"   status       : ✅ validated (attempts {r.attempts} / self-corrections {r.corrections})")
            print(f"   contractor   : {d.contractor_name.value!r}  (conf={d.contractor_name.confidence:.2f})")
            print(f"   counterparty : {d.counterparty_name.value!r}  (conf={d.counterparty_name.confidence:.2f})")
            dv = d.contract_date.value.isoformat() if d.contract_date.value else None
            print(f"   date (ISO)   : {dv}  (conf={d.contract_date.confidence:.2f})")
            for m in d.monetary_amounts:
                print(f"     - {m.label}: {m.amount_yen:,} yen  (conf={m.confidence:.2f}, source='{m.quote}')")
            if r.flagged:
                fl = ", ".join(f"{f}({c:.2f})" for f, c in r.flagged)
                print(f"   ⚠ needs human review: {fl}")
        else:
            print(f"   status       : ❌ failed — {r.error}")

    # Summary
    rate = (success / len(results) * 100) if results else 0.0
    print()
    print(SEP)
    print("📊 Batch processing summary")
    print(SEP)
    print(f"  Client used        : {client_name}")
    print(f"  Documents          : {len(results)}")
    print(f"  Parse success rate : {success}/{len(results)}  ({rate:.0f}%)")
    print(f"  Processing time    : {elapsed:.2f} s  ({elapsed/len(results):.2f} s/doc)")
    print(f"  Token consumption  : total {total_usage.total:,}  "
          f"(input {total_usage.input_tokens:,} / output {total_usage.output_tokens:,})")
    print(f"  Self-corrections   : {total_corrections}")
    print(f"  Human-review flags : {total_flags}  (confidence < {CONFIDENCE_THRESHOLD})")
    print(f"  Saved to           : {out_path}")
    print()

    return 0 if success == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
