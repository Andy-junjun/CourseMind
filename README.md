# CourseMind

CourseMind is a course-material learning assistant skeleton for a 3-day sprint. The repository is intentionally runnable in `mock` mode first, while keeping stable interfaces for real PDF parsing, retrieval, reranking, LLM generation, quiz generation, and Bandit review recommendation.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Default mode is `mock`, so the app can run without an API key, FAISS, or trained models.

## Modes

```text
mock   : deterministic fallback behavior, suitable for integration and demos
hybrid : real modules may be used when available, fallback otherwise
real   : expected to use real API/model/index implementations
```

Set mode with:

```bash
set COURSEMIND_MODE=mock
```

## Team Boundaries

```text
A group: integration, Streamlit UI, README, runnable demo
B group: PDF parsing, chunking, embedding, FAISS/BM25, GraphRAG-lite, retrieval
C group: LLM client, generator, MiniRanker, answer guard
D group: Bandit recommender, quiz feedback, test cases, PPT, demo materials
```

All modules must follow `src/schemas.py` and `docs/api_contract.md`. Do not rename shared fields or change public function signatures without synchronizing the whole team.

## Required Deliverables

Every member must contribute at least one traceable repository artifact: a PR, source module, test case file, screenshot, experiment result, PPT contribution note, or documentation update.

Never commit API keys, local model weights, private course files, or generated indexes.

