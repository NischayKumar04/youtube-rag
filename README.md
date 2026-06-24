# 🎥 YouTube Video RAG

Ask questions about any YouTube video's content using Retrieval-Augmented Generation. This app fetches video transcripts, creates semantic embeddings, and uses an LLM to answer your questions grounded in the actual video content.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        YouTube Video RAG                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ YouTube  │───▶│  Transcript  │───▶│    Text      │               │
│  │ Video ID │    │  Fetcher     │    │  Splitter    │               │
│  └──────────┘    │ (youtube-    │    │ (Recursive   │               │
│                  │  transcript- │    │  Character   │               │
│                  │  api)        │    │  1000 chars) │               │
│                  └──────────────┘    └──────┬───────┘               │
│                                            │                        │
│                                            ▼                        │
│                                   ┌──────────────┐                  │
│                                   │   E5 Base    │                  │
│                                   │  Embeddings  │                  │
│                                   │  (Local,     │                  │
│                                   │   multilingual│                 │
│                                   │   e5-base)   │                  │
│                                   └──────┬───────┘                  │
│                                          │                          │
│                                          ▼                          │
│                                   ┌──────────────┐                  │
│                                   │    FAISS     │                  │
│                                   │ Vector Store │                  │
│                                   │ (In-Memory)  │                  │
│                                   └──────┬───────┘                  │
│                                          │                          │
│  ┌──────────┐                            │                          │
│  │  User    │    ┌──────────────┐        │                          │
│  │ Question │───▶│  Retriever   │◀───────┘                          │
│  └──────────┘    │  (Top-K      │                                   │
│       │          │   Similarity) │                                   │
│       │          └──────┬───────┘                                   │
│       │                 │                                           │
│       │                 ▼                                           │
│       │          ┌──────────────┐                                   │
│       └─────────▶│    Chat      │                                   │
│                  │   Prompt     │                                   │
│                  │  (System +   │                                   │
│                  │   Human Msg) │                                   │
│                  └──────┬───────┘                                   │
│                         │                                           │
│                         ▼                                           │
│                  ┌──────────────┐    ┌──────────────┐               │
│                  │   GLM-5.2   │───▶│   Answer     │               │
│                  │  (HF Chat   │    │  (Grounded   │               │
│                  │   API)      │    │   in context) │               │
│                  └──────────────┘    └──────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### LCEL Chain Flow

```
RunnableParallel ──┬── "context": Retriever → format_docs()
                   └── "question": RunnablePassthrough()
        │
        ▼
  ChatPromptTemplate (system + human messages)
        │
        ▼
  ChatHuggingFace (GLM-5.2 via HF Inference API)
        │
        ▼
  StrOutputParser → Final Answer
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) via HuggingFace Inference API |
| **Embeddings** | [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) (local) |
| **Vector Store** | FAISS (in-memory) |
| **Framework** | LangChain (LCEL) |
| **Transcript** | youtube-transcript-api |
| **Frontend** | Streamlit |

## Project Structure

```
RAG/
├── app.py                  # Streamlit web interface
├── main.py                 # CLI interface
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not tracked)
├── .gitignore
└── src/
    ├── transcript.py       # YouTube transcript fetcher
    ├── vectorstore.py      # E5 embeddings + FAISS vector store
    └── rag.py              # LangChain RAG chain builder
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/youtube-rag.git
cd youtube-rag
```

### 2. Create a virtual environment

```bash
python -m venv rag-venv

# Windows
rag-venv\Scripts\activate

# macOS/Linux
source rag-venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_api_token
```

Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens).

### 5. Run the app

**Streamlit (Web UI):**

```bash
streamlit run app.py
```

**CLI:**

```bash
python main.py
```

## Usage

1. Enter a YouTube Video ID (the part after `v=` in the URL)
2. Click **Load Video** — the app fetches the transcript, chunks it, and builds a vector store
3. Ask questions in the chat — answers are grounded in the actual video transcript

> **Note:** The first run downloads the `intfloat/multilingual-e5-base` embedding model (~1.1GB). This is cached locally for subsequent runs.

## How It Works

1. **Transcript Extraction** — Fetches the YouTube video transcript (Hindi/English) using `youtube-transcript-api`
2. **Chunking** — Splits the transcript into overlapping chunks (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter`
3. **Embedding** — Each chunk is embedded locally using the multilingual E5-base model with proper `passage:` prefix
4. **Indexing** — Chunks are stored in an in-memory FAISS vector index
5. **Retrieval** — User questions are embedded with `query:` prefix and matched against the index (top-K similarity)
6. **Generation** — Retrieved context + question are sent to GLM-5.2 via HuggingFace's chat API to generate a grounded answer


