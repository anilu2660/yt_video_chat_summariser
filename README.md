# 📺 YouTube Video Chatbot — RAG with LangChain & Streamlit

An interactive chatbot that lets you **ask questions about any YouTube video** using its transcript. Powered by **Retrieval-Augmented Generation (RAG)** with LangChain, OpenAI, and FAISS — wrapped in a clean Streamlit UI.

---

## 🚀 Features

- 🎬 **YouTube Transcript Ingestion** — Automatically fetches the English transcript of any YouTube video.
- ✂️ **Smart Chunking** — Splits the transcript into overlapping chunks for accurate retrieval.
- 🔍 **Semantic Search** — Embeds chunks with OpenAI's `text-embedding-3-small` and stores them in a FAISS vector store.
- 🤖 **Context-Aware Q&A** — Uses `gpt-4o-mini` to answer questions strictly from the retrieved transcript context.
- 💬 **Chat Interface** — Persistent, multi-turn conversation UI powered by Streamlit's chat components.
- 🔐 **Secure API Key Input** — OpenAI key entered at runtime via the sidebar (never hardcoded).

---

## 🏗️ Architecture

```
YouTube Video ID
       │
       ▼
YouTubeTranscriptApi  ──►  Raw Transcript Text
       │
       ▼
RecursiveCharacterTextSplitter  ──►  Text Chunks (1000 chars, 200 overlap)
       │
       ▼
OpenAIEmbeddings (text-embedding-3-small)
       │
       ▼
FAISS Vector Store  ──►  Retriever (top-4 similar chunks)
       │
       ▼
RunnableParallel ──► PromptTemplate ──► ChatOpenAI (gpt-4o-mini) ──► Answer
```

---

## 📋 Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/)
- A YouTube video that has **English captions/subtitles** available

---

## ⚙️ Installation

1. **Clone the repository / navigate to the project folder**

   ```bash
   cd RAG/yt_chatbot_rag
   ```

2. **Create and activate a virtual environment** *(recommended)*

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install streamlit langchain langchain-openai langchain-community \
               langchain-text-splitters faiss-cpu youtube-transcript-api
   ```

---

## ▶️ Running the App

```bash
streamlit run stremalit_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧑‍💻 How to Use

1. **Enter your OpenAI API key** in the sidebar.
2. **Paste a YouTube Video ID** in the input field (the part after `?v=` in a YouTube URL).
   - Example: for `https://www.youtube.com/watch?v=wrcQwMpAirQ`, the ID is `wrcQwMpAirQ`.
3. Click **"Process Video"** — the app will fetch the transcript and build the vector index.
4. Once loaded, **type your questions** in the chat input and get answers grounded in the video content.

> ⚠️ The chatbot only answers based on the video's transcript. If a question is outside the video's content, it will say so.

---

## 📁 Project Structure

```
yt_chatbot_rag/
├── stremalit_app.py        # Main Streamlit application
├── rag_using_langchain.py  # Original standalone RAG script (no UI)
└── README.md               # This file
```

---

## 🔧 Configuration

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Max characters per text chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| `embedding model` | `text-embedding-3-small` | OpenAI embedding model |
| `llm model` | `gpt-4o-mini` | OpenAI chat model |
| `retriever k` | `4` | Number of chunks retrieved per query |
| `temperature` | `0.2` | LLM creativity (lower = more factual) |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io/) | Web UI framework |
| [LangChain](https://www.langchain.com/) | RAG orchestration (chains, prompts, retrievers) |
| [OpenAI](https://openai.com/) | Embeddings (`text-embedding-3-small`) & LLM (`gpt-4o-mini`) |
| [FAISS](https://github.com/facebookresearch/faiss) | Local vector store for similarity search |
| [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) | Fetches YouTube video transcripts |

---

## ⚠️ Limitations

- Only works with YouTube videos that have **English transcripts** (auto-generated or manual).
- Answers are limited to the video's transcript content — it won't draw on outside knowledge.
- Very long videos may result in higher OpenAI API costs due to embedding all chunks.

---

## 📄 License

This project is for educational purposes. Feel free to extend and adapt it.
