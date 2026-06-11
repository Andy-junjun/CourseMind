# Task Board

Use this file as the sprint source of truth. Do not rely only on chat messages.

Each member must own at least one traceable deliverable: code, docs, test cases, screenshots, experiment records, or PPT contribution notes.

## A1 Project Bootstrap And Environment

Owner:
Collaborators:

Input:
- Repository skeleton
- `.env.example`

Output:
- `README.md`
- `requirements.txt`
- verified startup command

Files:
- `README.md`
- `requirements.txt`
- `.env.example`

Acceptance:
- A new member can clone the repository and run `streamlit run app.py`.
- README explains `mock`, `hybrid`, and `real` modes.
- README clearly says API keys must not be committed.

## A2 Streamlit Main Layout

Owner:
Collaborators:

Input:
- Public functions from `src/`

Output:
- Complete visible app pages

Files:
- `app.py`

Acceptance:
- App has Q&A, summary, quiz, review, retrieval visualization, and status areas.
- The app starts in `mock` mode without external services.
- The sidebar displays mode, page count, and chunk count.

## A3 End-To-End Demo Flow

Owner:
Collaborators:

Input:
- `demo/demo_questions.md`
- public module functions

Output:
- One stable demo path

Files:
- `app.py`
- `demo/demo_questions.md`

Acceptance:
- Demo can show upload/build knowledge base, Q&A, citations, quiz, wrong feedback, and review recommendation.
- Demo can run with sample text if no PDF is uploaded.
- Demo path is documented in `demo/demo_questions.md`.

## A4 Integration Guard And Bug Fixing

Owner:
Collaborators:

Input:
- All group modules

Output:
- Stable integration branch

Files:
- `app.py`
- `src/config.py`
- `tests/`

Acceptance:
- `python -m pytest` passes.
- `python -m compileall src app.py` passes.
- No public function signature is changed without updating `docs/api_contract.md`.

## B1 PDF Parsing

Owner:
Collaborators:

Input:
- `data/raw/*.pdf`
- `data/raw/*.txt`

Output:
- `list[DocumentPage]`

Files:
- `src/document_loader.py`

Acceptance:
- Can parse at least one PDF when PyMuPDF is installed.
- Can load `.txt` or `.md` fallback files.
- Each page has `file_name`, `page`, and `text`.

## B2 Chunking And Concept Extraction

Owner:
Collaborators:

Input:
- `list[DocumentPage]`

Output:
- `list[Chunk]`

Files:
- `src/chunker.py`
- `tests/test_chunker.py`

Acceptance:
- Every chunk includes `chunk_id`, `file_name`, `page`, `text`, `chapter`, and `concepts`.
- `chunk_id` is stable across repeated runs.
- Chunk size and overlap are configurable.

## B3 Embedding And Vector Store

Owner:
Collaborators:

Input:
- Query text
- Chunk text

Output:
- Embedding vectors
- vector similarity score

Files:
- `src/embedder.py`
- `src/vector_store.py`

Acceptance:
- Mock embedding works without model downloads.
- Real embedding can later be added without changing `retrieve(...)`.
- Similarity scores are numeric and visible through retrieval results.

## B4 BM25 / Keyword Retrieval

Owner:
Collaborators:

Input:
- Query
- `list[Chunk]`

Output:
- BM25 or keyword score

Files:
- `src/bm25_store.py`
- `src/retriever.py`

Acceptance:
- Keyword overlap fallback works in `mock` mode.
- Each `RetrievedChunk` includes `bm25_score`.
- Top-K retrieval returns deterministic results for the same query.

## B5 GraphRAG-lite Expansion

Owner:
Collaborators:

Input:
- Seed `list[RetrievedChunk]`
- all chunks

Output:
- expanded `list[RetrievedChunk]`

Files:
- `src/graph_store.py`

Acceptance:
- Same-page chunks can receive graph score.
- Shared-concept chunks can receive graph score.
- Expanded candidates still use the shared `RetrievedChunk` schema.

## C1 MiniRanker Scoring

Owner:
Collaborators:

Input:
- Query
- `list[RetrievedChunk]`

Output:
- `list[RankedChunk]`

Files:
- `src/miniranker.py`

Acceptance:
- Each result has `ranker_score`.
- UI can show ranking score.
- The module works without `miniranker.pt` in `mock` mode.

## C2 LLM Client And Mode Switch

Owner:
Collaborators:

Input:
- Prompt
- environment variables

Output:
- generated text

Files:
- `src/llm_client.py`
- `src/config.py`
- `.env.example`

Acceptance:
- `mock` mode returns deterministic text.
- `real` mode has a clear implementation location.
- Missing API key does not crash `mock` mode.

## C3 Answer Generation And Citations

Owner:
Collaborators:

Input:
- User query
- `list[RankedChunk]`

Output:
- answer dict with citations

Files:
- `src/generator.py`

Acceptance:
- Answer includes citation entries.
- Each citation includes `chunk_id`, `file_name`, `page`, and score.
- Citation data can be rendered by `app.py`.

## C4 Refusal Guard And Summary

Owner:
Collaborators:

Input:
- Query
- ranked evidence
- chunks

Output:
- refusal decision
- document summary

Files:
- `src/answer_guard.py`
- `src/generator.py`

Acceptance:
- Low-evidence queries can be refused.
- Summary works in `mock` mode.
- Refusal behavior can be demonstrated with an out-of-scope question.

## D1 Quiz Generation

Owner:
Collaborators:

Input:
- `list[Chunk]`

Output:
- `list[QuizItem]`

Files:
- `src/study_tools.py`

Acceptance:
- Generates at least 3 quiz items when enough chunks exist.
- Every quiz item includes question, options, answer, explanation, concept, and source chunk id.
- Quiz can be rendered in `app.py`.

## D2 Bandit Review Recommendation

Owner:
Collaborators:

Input:
- quiz feedback
- concept

Output:
- recommended review concept
- persistent state file

Files:
- `src/bandit_recommender.py`
- `tests/test_bandit.py`

Acceptance:
- `update_feedback(concept, correct)` records attempts.
- Wrong answers increase review priority.
- Recommendation output includes concept, score, scores, and reason.

## D3 Test Cases And Experiment Records

Owner:
Collaborators:

Input:
- course materials
- demo questions

Output:
- test cases
- experiment plan and results

Files:
- `demo/test_cases.csv`
- `docs/experiment_plan.md`

Acceptance:
- At least 20 questions are prepared before final presentation.
- Questions include factual, summary, quiz-generation, and out-of-scope types.
- Experiment metrics include Recall@5, answer accuracy, citation accuracy, refusal accuracy, and response time.

## D4 PPT, Screenshots, And Contribution Records

Owner:
Collaborators:

Input:
- app screenshots
- experiment records
- member work records

Output:
- PPT materials
- contribution notes
- demo screenshots

Files:
- `demo/screenshots/`
- `ppt/member_contributions.md`

Acceptance:
- Every member has a contribution entry.
- Screenshots cover Q&A, citations, MiniRanker score, quiz, and Bandit recommendation.
- PPT only claims features that are implemented or clearly marked as fallback.

