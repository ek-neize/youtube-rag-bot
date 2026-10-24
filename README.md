# 🎬 YouTube RAG Bot

**Chat with any YouTube video.** Paste a link, get a summary, and ask questions. Every answer cites the exact moments in the video it came from, and you can click a timestamp to jump the player there.

Built with **LangChain**, **OpenAI**, **FAISS** and **Streamlit** as a hands-on project in Retrieval-Augmented Generation (RAG).

![CI](https://github.com/ek-neize/youtube-rag-bot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![App screenshot](docs/screenshot.png)

## ✨ Features

- **Q&A over any video with a transcript.** Ask in plain English and get answers grounded only in what was said.
- **Cited, clickable sources.** Each answer lists the transcript excerpts it used, with ▶ buttons that seek the embedded player to that moment.
- **One-click summaries.** You get an overview paragraph plus key takeaways. Long videos are handled with map-reduce summarization.
- **Smart transcript loading.** The app prefers manual English captions, falls back to auto-generated ones, then to machine-translated English.
- **Conversational memory.** Follow-ups like "tell me more about that" work: each one is rewritten into a standalone question using the chat history before searching, and the app shows what it searched for.
- **Streaming answers.** Replies appear word by word as the model writes them, like ChatGPT, instead of after a long pause.
- **Tunable retrieval.** Change the chat model, chunk size and *k* from the sidebar and see how answers change.
- **Tested offline.** 59 tests (unit + Streamlit UI) use fake LLMs and embeddings, so they run in about a second with no API key and no cost.

## 🧠 How it works

```mermaid
flowchart LR
    A[YouTube URL] --> B[youtube-transcript-api<br/>fetch captions]
    B --> C[Time-aware chunking<br/>start/end metadata]
    C --> D[OpenAI embeddings<br/>text-embedding-3-small]
    D --> E[(FAISS<br/>vector index)]
    Q[User question] --> R[Similarity search<br/>top-k chunks]
    E --> R
    R --> P[Prompt with time-labelled<br/>context]
    P --> L[OpenAI chat model<br/>gpt-4o-mini]
    L --> O[Answer + cited timestamps]
```

1. **Load:** fetch the video's captions as timed segments.
2. **Chunk:** group consecutive captions into ~1,000-character chunks, with a small overlap, keeping each chunk's start and end time.
3. **Embed and index:** convert chunks to vectors and store them in an in-memory FAISS index (built once per video and cached).
4. **Retrieve:** rewrite follow-ups into standalone questions using the chat history, then embed the question and pull the *k* most similar chunks.
5. **Generate:** the LLM answers using only those chunks, each labelled like `[2:15-2:48]`, and cites the times it relied on.

## 🛠️ Design decisions

| Decision | Why |
|---|---|
| **Time-aware chunking** instead of a plain character splitter | Chunking on caption boundaries keeps exact start/end times, which powers the cited, clickable sources. |
| **Core logic separated from the UI** (`core/` vs `ytbot.py`) | The RAG pipeline is plain Python that can be tested, reused in a CLI or API, or moved to another UI. |
| **Dependency injection for the LLM and embeddings** | `VideoQA` accepts any LangChain chat model and embeddings. Tests pass in fakes; swapping providers is a one-line change. |
| **LCEL chains** (`prompt \| llm \| parser`) | LangChain's current composition style; the older `LLMChain` is deprecated. |
| **Grounding prompt with an "I don't know" rule** | Reduces hallucination: the model says when the transcript doesn't contain the answer. |
| **Map-reduce summaries** | Long transcripts are summarized in sections, then merged, so they never overflow the context window. |
| **Per-video index caching** | Embeddings are computed once per video and session, not on every question, saving time and API cost. |

## 📁 Project structure

```
youtube-rag-bot/
├── ytbot.py              # Streamlit UI
├── core/
│   ├── config.py         # Settings from env vars / .env
│   ├── transcript.py     # URL parsing, transcript fetching, time-aware chunking
│   ├── prompts.py        # Summary, combine and Q&A prompts
│   └── rag.py            # VideoQA: FAISS index, retrieval, answer & summarize
├── tests/                # Offline unit + UI tests (fake LLM + embeddings)
├── .github/workflows/    # CI: lint + tests on every push
├── .vscode/              # Run/debug configs, test discovery, formatter
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```

## 🚀 Getting started

**Prerequisites:** Python 3.10+ and an [OpenAI API key](https://platform.openai.com/api-keys).

```bash
git clone https://github.com/ek-neize/youtube-rag-bot.git
cd youtube-rag-bot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env             # then set OPENAI_API_KEY in .env
streamlit run ytbot.py
```

Then open http://localhost:8501. In VS Code you can also use **Run and Debug → Streamlit: ytbot**.

## 🧪 Tests and linting

```bash
pytest -q          # 59 tests, run offline in about a second
ruff check .       # lint
```

GitHub Actions runs both on every push and pull request.

## 📏 Retrieval evaluation

A RAG app is only as good as its retrieval: if the right part of the transcript isn't retrieved, the model can't answer. So the project includes a small evaluation harness.

- **Test set:** [`evals/neural_networks.json`](evals/neural_networks.json) has 21 questions about [3Blue1Brown's neural network video](https://www.youtube.com/watch?v=aircAruvnKk), each labelled with the time range where the answer is spoken.
- **Hit rate@k:** share of questions where a correct chunk (one overlapping the answer's time range) is in the top *k*.
- **MRR@k:** mean reciprocal rank; 1.0 means the right chunk is always ranked first.
- **Video retrieved:** how much of the video the top *k* chunks span. Big chunks and big *k* inflate hit rate by retrieving more of the video, so the best setting is chosen among those retrieving at most 25%.

```bash
python scripts/evaluate_retrieval.py --show-misses --save evals/results.md
```

Results (`text-embedding-3-small`):

| Chunk size | Chunks | k | Hit rate | MRR | Video retrieved |
|---:|---:|---:|---:|---:|---:|
| 500 | 55 | 1 | 62% | 0.62 | 3% |
| 500 | 55 | 3 | 81% | 0.71 | 7% |
| 500 | 55 | 5 | 90% | 0.74 | 12% |
| 1000 | 22 | 1 | 90% | 0.90 | 5% |
| 1000 | 22 | 3 | 90% | 0.90 | 15% |
| 1000 | 22 | 5 | 100% | 0.93 | 25% |
| 1500 | 14 | 1 | 86% | 0.86 | 8% |
| 1500 | 14 | 3 | 95% | 0.90 | 23% |
| 1500 | 14 | 5 | 100% | 0.91 | 38% |
| 2000 | 10 | 1 | 76% | 0.76 | 11% |
| 2000 | 10 | 3 | 90% | 0.83 | 31% |
| 2000 | 10 | 5 | 95% | 0.84 | 51% |

## 🐳 Docker

```bash
docker build -t youtube-rag-bot .
docker run -p 8501:8501 --env-file .env youtube-rag-bot
```

## ⚙️ Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `CHAT_MODEL` | `gpt-4o-mini` | Model for answers and summaries |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embeddings |
| `CHUNK_SIZE` | `1000` | Target characters per chunk |
| `CHUNK_OVERLAP_SEGMENTS` | `2` | Caption lines shared between neighbouring chunks |
| `TOP_K` | `5` | Chunks retrieved per question |
| `MAX_HISTORY_MESSAGES` | `6` | Past chat messages used to understand follow-ups |
| `MAX_SINGLE_PASS_CHARS` | `100000` | Above this, summaries use map-reduce |

## 🩺 Troubleshooting

| Problem | Fix |
|---|---|
| "Transcripts are disabled for this video" | The uploader turned captions off; try another video. |
| "YouTube is blocking transcript requests" | Common on cloud servers; run locally or configure a proxy. |
| `AuthenticationError` | Check `OPENAI_API_KEY` in `.env` or in the sidebar. |
| `ModuleNotFoundError: core` | Run commands from the project root with `.venv` activated. |

## 🗺️ Roadmap

- [x] Conversational memory: rewrite follow-up questions using chat history
- [ ] Hybrid retrieval (BM25 + vectors) and re-ranking
- [ ] Persist FAISS indexes to disk so revisited videos skip embedding
- [x] Stream answers token by token
- [ ] Multi-video and playlist search
- [x] Retrieval evaluation set (hit rate / MRR) to tune chunk size and *k*

## 🙏 Acknowledgements

This project started from IBM's Skills Network guided project on building a YouTube summarizer with watsonx and Gradio. I rebuilt it with OpenAI and Streamlit and extended it with time-aware chunking, cited sources, tests and CI.

## 📄 License

[MIT](LICENSE) © 2026 Ekow Ashun
