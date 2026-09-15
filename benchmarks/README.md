# Chunking benchmark

Fixed-window chunking (the original `ingest.py` behaviour) versus semantic
breakpoint chunking (the new default), measured on three public manuals with
five questions each, ordered from simple to complex.

| Manual | Pages | Fixed chunks | Semantic chunks | Semantic ingest time |
|---|---:|---:|---:|---:|
| *An Introduction to R* (CRAN R-intro.pdf) | 103 | 336 | 315 | ~1 min |
| *The Python Tutorial* 3.13 (tutorial.pdf) | 167 | 474 | 411 | ~1 min |
| *The Julia Language* 1.13.0 manual | 1956 | 4126 | 3693 | ~4 min |

Both regimes use the same extractor (PyMuPDF), the same embedding model
(ChromaDB's default `all-MiniLM-L6-v2`), the same top-5 retrieval and the
chatbot's own prompt. Answers came from `qwen3.5:4b`; a second model,
`qwen3-next:80b-a3b-instruct`, scored each answer 1 to 5 against a reference
answer without knowing which regime produced it. Both ran locally in Ollama.

## Results (run 2026-09-14)

| Metric | Fixed window | Semantic |
|---|---:|---:|
| Mean judge score (1 to 5) | 4.00 | 4.53 |
| Questions where the top-1 chunk contains every key term | 7 / 15 | 8 / 15 |
| Questions where any top-5 chunk contains every key term | 12 / 15 | 12 / 15 |
| Top-1 chunk starts and ends on a clean boundary | 0 / 15 | 11 / 15 |
| Mean top-1 cosine distance (lower is closer) | 0.772 | 0.737 |

Per question (fixed / semantic judge score):

| | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| R | 3 / **5** | 5 / 5 | 4 / **5** | 5 / 5 | **5** / 4 |
| Python | 4 / **5** | 4 / **5** | 4 / **5** | 5 / 5 | 5 / 5 |
| Julia | 1 / **5** | 5 / 5 | **4** / 3 | 2 / **4** | **4** / 2 |

Semantic chunking scored higher on 8 questions, tied on 4 and lower on 3.

## What the numbers say

Retrieval by key-term hits is almost the same in both regimes: the same
passages get found. What changes is what the model is handed. Every fixed
chunk in the top-1 slot starts mid-sentence (0 of 15 have a clean start), and
several cut through the sentence that carries the answer. The clearest case is
the Julia question about `!` at the end of a function name: the semantic top
chunk contains the manual's one sentence stating the convention, and the model
answers correctly; the fixed pipeline never surfaces that sentence intact and
the model replies that the documentation does not mention it (score 1 vs 5).

The three losses are worth reading too. On the R regression question the
semantic chunks over-committed to the nonlinear-regression section; on the two
Julia losses (broadcasting, macros) the 4B answer model hallucinated syntax
from a correct but shorter context. Semantic chunks are on average shorter and
carry no overlap, so a small model occasionally gets less to work with.

## Reproduce

```bash
# One ChromaDB collection per (language, regime)
python ingest.py ./pdfs/r      --collection r_fixed      --chunking fixed
python ingest.py ./pdfs/r      --collection r_semantic
python ingest.py ./pdfs/python --collection python_fixed --chunking fixed
python ingest.py ./pdfs/python --collection python_semantic
python ingest.py ./pdfs/julia  --collection julia_fixed  --chunking fixed
python ingest.py ./pdfs/julia  --collection julia_semantic

# Retrieval metrics, answers and judge scores -> results.json
python benchmarks/compare_chunking.py --ollama-url http://127.0.0.1:11434 \
    --answer-model qwen3.5:4b --judge-model qwen3-next:80b-a3b-instruct

# Or retrieval metrics only, no Ollama needed
python benchmarks/compare_chunking.py --skip-llm

# Self-contained HTML report with chunks, answers and a per-question chart
python benchmarks/build_report.py --credit "Suggested by @someone on Bluesky"
```

The manuals used here are public: R-intro.pdf from CRAN, tutorial.pdf from the
Python 3.13 PDF archive on docs.python.org, and julia-1.13.0.pdf from the
JuliaLang/docs.julialang.org repository's `assets` branch.

## Caveats

- Fifteen questions is a small sample; one judge call can move the mean by 0.07.
- The judge is a language model. Its reasons are recorded in `results.json` so
  every score can be checked by hand.
- Questions and key terms were written before looking at the retrieved chunks,
  but by the same person who wrote the chunker.
- The answer model is small (4B) on purpose: it shows how much the context
  matters. A larger model can paper over a bad chunk.
