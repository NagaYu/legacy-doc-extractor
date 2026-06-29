# Unstructured Data Extraction Tool for Legacy Industries (Insurance / Real Estate) — Prototype

A prototype automation tool that extracts the fields you need from messy "contract / policy text"
— full of inconsistent notation and verbose boilerplate — and converts them into structured JSON
that conforms to a Pydantic schema.

> ⚠️ **This repository is a prototype (proof of concept), not for production use.**
> - The bundled `data/sample_contract_*.txt` files are **fictional samples**. They contain no real contracts or personal data.
> - **Do not feed in real contracts or personally identifiable information (PII).** When using LLM
>   extraction, the text is sent to an external API. Handling of real data is the user's responsibility
>   and must comply with your organization's policies and applicable laws.
> - **Always review the extracted results by hand.** The tool outputs confidence scores and a
>   `needs_human_review` flag, but it does not guarantee accuracy.
> - When `ANTHROPIC_API_KEY` is unset, it runs via a mock / regex fallback. That path is **for
>   demonstration only** and is not production quality. Verify the actual extraction accuracy with a real LLM by setting the key.

## Features
- **Strict schema definition / validation with Pydantic** (`ContractData` in `src/extractor.py`)
- **LLM structured extraction** (Anthropic Structured Outputs / model `claude-opus-4-8`)
- **Normalization for legacy documents**
  - Amounts: `120,000円` / `12万円` / `金弐拾肆萬円` (kanji numerals) → unified to an integer in yen
  - Dates: absorbs Japanese-era `令和6年4月1日` ⇄ Gregorian `2024年` and unifies to ISO format `2024-04-01`
- **API-key-free fallback**: when the `ANTHROPIC_API_KEY` environment variable is absent, it
  automatically switches to deterministic regex-based extraction so the demo runs end to end.

## Directory layout
```
.
├── requirements.txt
├── data/
│   ├── generate_data.py        # Generates fictional legacy contract texts
│   └── sample_contract_*.txt   # Generated test data (2 real estate + 1 insurance)
├── src/
│   ├── extractor.py            # Pydantic schema + extraction pipeline
│   └── main.py                 # load → extract → validate → save → side-by-side display
└── output/
    └── extracted_data.json     # Extraction results
```

## Setup & run
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# 1) Generate test data
./.venv/bin/python data/generate_data.py

# 2) (Optional) Set an API key only if you want high-accuracy LLM extraction
export ANTHROPIC_API_KEY="sk-ant-..."

# 3) Run the extraction demo (shows the source text alongside the structured JSON)
./.venv/bin/python src/main.py
```

If `ANTHROPIC_API_KEY` is set, LLM extraction is used; if not, rule-based extraction is used
(the console prints which "extraction method" was used). Results are saved to `output/extracted_data.json`.

## v2: Production-hardened version (async batch)

`src/robust_extractor.py` + `src/run_async.py` is a hardened version built to withstand API cost
limits and error rates.

```bash
./.venv/bin/python src/run_async.py   # async batch extraction + summary
```

Enhancements:
- **100% structuring guarantee** — Pydantic v2 `@field_validator` (ISO date / integer-amount checks).
  On validation failure, the error reason is fed back to the LLM for **self-correction (up to 2 retries)**.
- **Cost / latency reduction** — compressed prompt + a single minimal-JSON few-shot example,
  `asyncio` async batching, and `tenacity` exponential-backoff retries (auto-recovers from rate-limit 429s).
- **Confidence quantification** — each field carries a `quote` (supporting source text) and a
  `confidence` (0.0–1.0). Anything below 0.7 emits a `[WARNING]` log and is flagged for human review.
- Prints a summary of **processing time / token consumption / parse success rate** at the end.
- Results are saved to `output/extracted_data_v2.json`.

When no API key is present, a deterministic mock LLM runs the same validation / self-correction /
confidence / token-accounting path (so you can demonstrate the self-correction loop and backoff retries firing).

> ℹ️ **Under the mock run, the summary values (processing time, token consumption, confidence, etc.) are approximate/fixed demo values.**
> Measure the real cost, latency, and extraction accuracy with a real LLM by setting `ANTHROPIC_API_KEY`
> (on the real-LLM path, token counts come from the API response's `usage`).
