"""
デモ実行スクリプト。

  1. data/ 配下の sample_contract_*.txt を読み込む
  2. extractor.extract_contract() で構造化データを抽出
  3. Pydantic によるバリデーションを通過したものを output/extracted_data.json に保存
  4. 「元テキスト」と「構造化JSON」をコンソールに並べて表示
"""

from __future__ import annotations

import json
import pathlib
import sys

# src/ を import パスに追加
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from pydantic import ValidationError  # noqa: E402

from extractor import ContractData, extract_contract  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

SEP = "═" * 78


def _print_source(text: str) -> None:
    print("📄 元テキスト（非構造化）")
    print("-" * 78)
    for line in text.rstrip().splitlines():
        print(f"  {line}")
    print()


def _print_structured(data: ContractData) -> None:
    print("✅ 抽出・構造化されたデータ（バリデーション済み JSON）")
    print("-" * 78)
    # Pydantic v2: ISO日付などを JSON 互換へ整形して整形出力
    rendered = json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=2)
    for line in rendered.splitlines():
        print(f"  {line}")
    print()


def main() -> int:
    samples = sorted(DATA_DIR.glob("sample_contract_*.txt"))
    if not samples:
        print(f"[error] テストデータが見つかりません: {DATA_DIR}")
        print("        先に `python data/generate_data.py` を実行してください。")
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
            # ここで再バリデーション（型・必須項目チェックを明示的に保証）
            validated = ContractData.model_validate(data.model_dump())
        except ValidationError as exc:
            failed += 1
            print(f"❌ バリデーション失敗: {exc}")
            print()
            continue
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"❌ 抽出処理でエラー: {exc}")
            print()
            continue

        success += 1
        print(f"🔧 抽出方式: {method}")
        print()
        _print_structured(validated)

        results.append({
            "source_file": path.name,
            "extraction_method": method,
            "data": validated.model_dump(mode="json"),
        })

    # 結果の保存
    out_path = OUTPUT_DIR / "extracted_data.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(SEP)
    print("📊 サマリー")
    print(SEP)
    print(f"  対象ファイル数 : {len(samples)}")
    print(f"  成功（検証通過）: {success}")
    print(f"  失敗           : {failed}")
    print(f"  保存先         : {out_path}")
    print()

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
