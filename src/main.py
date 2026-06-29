"""
Demo runner script.

  1. Load sample_contract_*.txt under data/
  2. Extract structured data with extractor.extract_contract()
  3. Save whatever passes Pydantic validation to output/extracted_data.json
  4. Print the "source text" and "structured JSON" side by side in the console
"""

from __future__ import annotations

import json
import pathlib
import sys

# Add src/ to the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pydantic import ValidationError  # noqa: E402

from extractor import ContractData, extract_contract  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

SEP = "═" * 78


def _print_source(text: str) -> None:
    print("📄 Source text (unstructured)")
    print("-" * 78)
    for line in text.rstrip().splitlines():
        print(f"  {line}")
    print()


def _print_structured(data: ContractData) -> None:
    print("✅ Extracted & structured data (validated JSON)")
    print("-" * 78)
    # Pydantic v2: render ISO dates etc. into JSON-compatible form, then pretty-print
    rendered = json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=2)
    for line in rendered.splitlines():
        print(f"  {line}")
    print()


def main() -> int:
    samples = sorted(DATA_DIR.glob("sample_contract_*.txt"))
    if not samples:
        print(f"[error] No test data found: {DATA_DIR}")
        print("        Run `python data/generate_data.py` first.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    success = 0
    failed = 0

    for path in samples:
        text = path.read_text(encoding="utf-8")

        print(SEP)
        print(f"🗂  {path.name}")
        print(SEP)
        _print_source(text)

        try:
            data, method = extract_contract(text)
            # Re-validate here (explicitly guarantee type / required-field checks)
            validated = ContractData.model_validate(data.model_dump())
        except ValidationError as exc:
            failed += 1
            print(f"❌ Validation failed: {exc}")
            print()
            continue
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"❌ Error during extraction: {exc}")
            print()
            continue

        success += 1
        print(f"🔧 Extraction method: {method}")
        print()
        _print_structured(validated)

        results.append({
            "source_file": path.name,
            "extraction_method": method,
            "data": validated.model_dump(mode="json"),
        })

    # Save results
    out_path = OUTPUT_DIR / "extracted_data.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(SEP)
    print("📊 Summary")
    print(SEP)
    print(f"  Files processed   : {len(samples)}")
    print(f"  Succeeded (valid) : {success}")
    print(f"  Failed            : {failed}")
    print(f"  Saved to          : {out_path}")
    print()

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
