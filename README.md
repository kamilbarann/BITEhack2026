# OrbitGuide

**AI-powered Space Law Assistant**

> Built at **BITEhack 2026** hackathon by team **Umiski** | Category: Artificial Intelligence

---

## About

OrbitGuide is a **RAG (Retrieval Augmented Generation)** application that answers questions about space law. It analyses regulatory documents from organisations such as NASA, ESA, UNOOSA, and ITU, then generates answers with source citations.

### Key Features
- **Semantic search** across a corpus of legal documents
- **LLM-generated answers** (Groq / Llama 3.1) with cited sources & page numbers
- **Confidence indicator** based on the number of unique source documents
- **Query expansion** (PL + EN) for better retrieval
- **Conversation memory** — the LLM retains the last 3 exchanges for follow-up questions
- **Domain filtering** — small-talk detector and out-of-domain guard

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│   PDF Documents │────▶│   Ingestion      │────▶│   ChromaDB    │
│   (NASA, ESA)   │     │   (embeddings)   │     │   (vectors)   │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
┌─────────────────┐     ┌──────────────────┐     ┌───────▼───────┐
│   User          │────▶│   Streamlit UI   │────▶│   Brain RAG   │
│                 │     │   (app.py)       │     │   (brain.py)  │
└─────────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                        ┌──────────────────┐     ┌───────▼───────┐
                        │   Response       │◀────│   LLM (Groq)  │
                        │   + sources      │     │   Llama 3.1   │
                        └──────────────────┘     └───────────────┘
```

---

## System Logic

### Confidence Score
The system rates answer confidence based on the number of unique source documents used:
- **High (100%):** 5 or more unique sources
- **Medium (60–80%):** 3–4 sources
- **Low (<60%):** 1–2 sources

### Conversation Context
The LLM remembers recent chat history (last 3 exchanges), allowing follow-up questions such as *"What are the costs?"* after asking about registration.

### Domain Filtering
- **Small-talk detector:** recognises greetings and casual messages, responding naturally without querying the knowledge base.
- **Out-of-domain guard:** declines questions unrelated to space law.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kamilbarann/BITEhack2026.git
cd BITEhack2026
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Required API keys:
- `GOOGLE_API_KEY` — for embeddings (gemini-embedding-001)
- `GROQ_API_KEY` — for LLM inference (llama-3.1-8b-instant)

---

## Usage

### 1. Index documents (one-time)

```bash
python -m src.ingestion
```

This creates a ChromaDB vector store from the PDF documents in `data/`.

### 2. Run the application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Evaluation

The system includes a built-in answer quality evaluation module:

```bash
python -m src.eval
```

> **Note:** Run as a module (`python -m ...`) from the project root.

### Metrics

| Metric | Description | Scale |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | 1–5 |
| **Relevancy** | Does the answer address the user's question? | 1–5 |
| **Out-of-domain handling** | Does the system decline off-topic questions? | 1–5 |

---

## Project Structure

```
OrbitGuide/
├── app.py              # Streamlit interface
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore
├── README.md
├── src/
│   ├── brain.py        # RAG logic + Chain of Thought
│   ├── ingestion.py    # PDF indexing pipeline
│   ├── eval.py         # Evaluation module
│   └── utils.py        # Configuration utilities
└── data/
    ├── *.pdf           # Source documents
    └── chroma_db/      # Vector database (generated)
```

---

## Data Sources

Documents analysed by the system:

| Document | Organisation |
|---|---|
| Outer Space Treaty (1967) | UN |
| Liability Convention | UNOOSA |
| Registration Convention | UNOOSA |
| Moon Agreement | UN |
| Space Debris Mitigation Guidelines | ESA |
| ITU Regulatory Procedures | ITU |
| Guidelines for Long-term Sustainability | COPUOS |

---

## Tech Stack

- **LangChain** — RAG orchestration
- **ChromaDB** — Vector database
- **Groq** — LLM inference (Llama 3.1)
- **Google AI** — Embeddings (gemini-embedding-001)
- **Streamlit** — User interface

---

## Team Umiski

Built at **BITEhack 2026** hackathon.

---

## License

MIT License
