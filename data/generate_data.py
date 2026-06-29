"""
Script that generates fictional unstructured documents for legacy industries (insurance / real estate).

It deliberately bakes in the usual "legacy" quirks:
  - inconsistent amount notation (120,000円 / 12万円 / 金壱拾弐萬円, etc.)
  - mixed Japanese-era and Gregorian dates (令和6年 / 2024年)
  - unnecessary preambles, boilerplate, and inconsistent line breaks
  - mixed full-width / half-width characters

Running it writes sample_contract_1〜3.txt under data/.
"""

from __future__ import annotations

import pathlib

# The directory this script lives in (= data/)
DATA_DIR = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1) Real estate lease agreement (lots of notation variance)
# ---------------------------------------------------------------------------
CONTRACT_1 = """\
　　　　　　　　建物賃貸借契約書（写）

本契約書は、貸主と借主との間で締結されるものであり、下記条項に従うものとする。
なお、本書面は控えとして借主へ交付するものである。

第１条（契約者）
　本物件の借主（以下「契約者」という。）は　山田 太郎　とする。
　貸主は　株式会社さくら不動産管理　とする。

第２条（契約期間）
　契約の始期は令和6年4月1日とし、期間は２年間とする。
　（参考：西暦では2024年4月1日からとなる。）

第３条（賃料等）
　月額家賃は金120,000円（消費税込み）とする。
　敷金は家賃の2ヶ月分（即ち金弐拾肆萬円）を預託するものとする。
　礼金については1ヶ月分、すなわち12万円を申し受ける。
　共益費は月額金8,000円也。

第４条（特約事項）
　・退去時にはクリーニング費用として実費を借主負担にて徴収するものとする。
　・ペットの飼育は原則として認めない。ただし貸主の書面による承諾を得た場合はこの限りではない。
　・契約者は、第三者への又貸し（転貸）を行ってはならない。

以上、本契約の成立を証するため本書を作成する。
"""


# ---------------------------------------------------------------------------
# 2) Life insurance contract notice (policy excerpt, with verbose boilerplate)
# ---------------------------------------------------------------------------
CONTRACT_2 = """\
ご契約内容のお知らせ　兼　約款抜粋

平素は格別のお引き立てを賜り、厚く御礼申し上げます。
このたびはご契約いただき誠にありがとうございます。下記の通りご契約内容をご確認ください。

【ご契約者さま】保戸田 花子　様
【保険会社】　　あさひ生命保険相互会社

■保険の種類：定期保険（無配当）
■契約日　　：2023年11月15日（和暦：令和5年11月15日）
■保険金額　：金壱千万円（10,000,000円）
■月払保険料：金8,500円
■保険期間　：10年

≪主な特約・免責事項について（要旨）≫
（１）責任開始期前に生じた疾病・傷害については、保険金をお支払いできない場合があります。
（２）告知義務違反があった場合、当社は契約を解除することがあります。
（３）保険契約者は、保険期間中いつでも将来に向かって契約を解約することができます。

本書面に関するお問い合わせは、担当代理店または当社カスタマーセンターまでご連絡ください。
"""


# ---------------------------------------------------------------------------
# 3) Commercial tenant lease (万円 notation, Japanese-era only, few line breaks)
# ---------------------------------------------------------------------------
CONTRACT_3 = """\
事業用建物賃貸借に関する覚書

本覚書は当事者間の合意事項を記録したものである。借主は田中商事株式会社（代表者　田中一郎）であり、貸主は野村ビルディング合同会社である。賃貸借の期間については令和7年1月10日を始期とし、以後3年間とする。賃料は月額35万円（税別）とし、別途消費税を加算して支払うものとする。保証金として賃料の6ヶ月分（金210万円）を契約締結時に預け入れる。なお原状回復費用は退去時に別途精算するものとし、通常損耗を超える毀損については借主の負担とする。中途解約の場合は6ヶ月前までに書面で通知することを要する。
"""


SAMPLES = {
    "sample_contract_1.txt": CONTRACT_1,
    "sample_contract_2.txt": CONTRACT_2,
    "sample_contract_3.txt": CONTRACT_3,
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, body in SAMPLES.items():
        path = DATA_DIR / filename
        path.write_text(body, encoding="utf-8")
        print(f"[generated] {path}  ({len(body)} chars)")


if __name__ == "__main__":
    main()
