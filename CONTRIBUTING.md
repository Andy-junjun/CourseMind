# Contributing

## Hard Rules

1. `src/schemas.py` is the shared contract. Do not rename fields casually.
2. Public function signatures in `docs/api_contract.md` are frozen for the first sprint.
3. Keep `mock` mode runnable after every change.
4. Put non-code work into the repository so it can be checked and cited.
5. Do not commit API keys, private PDFs, model weights, or generated indexes.

## Group Ownership

```text
A group owns app.py, integration, environment docs, and demo stability.
B group owns document loading, chunking, retrieval, BM25, vectors, and graph expansion.
C group owns LLM calls, generation, answer guard, MiniRanker, and optional model training.
D group owns quiz feedback, Bandit recommendation, test cases, experiment records, and PPT materials.
```

Code can cross group boundaries only through the public interfaces.

