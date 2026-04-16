<div align="center">

<br/>

```
██████╗ ███████╗ █████╗ ██╗         ███████╗███████╗████████╗ █████╗ ████████╗███████╗
██╔══██╗██╔════╝██╔══██╗██║         ██╔════╝██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██╔════╝
██████╔╝█████╗  ███████║██║         █████╗  ███████╗   ██║   ███████║   ██║   █████╗  
██╔══██╗██╔══╝  ██╔══██║██║         ██╔══╝  ╚════██║   ██║   ██╔══██║   ██║   ██╔══╝  
██║  ██║███████╗██║  ██║███████╗    ███████╗███████║   ██║   ██║  ██║   ██║   ███████╗
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝    ╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚══════╝
                             A S S I S T A N T
```

### 🏡 RAG-Powered Real Estate Intelligence — Ask Anything About the Market

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-E9704A?style=for-the-badge)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> *Feed it real estate URLs. Ask it anything. Get grounded, source-backed answers — powered by RAG.*

<br/>

</div>

---

## ✨ Overview

**Real Estate Assistant RAG** is an intelligent question-answering tool built on **Retrieval-Augmented Generation (RAG)**. It scrapes real estate news and market articles from any URL you provide, stores the content as vector embeddings in **ChromaDB**, and answers your questions using **Llama 3.3 70B** via Groq — all through a clean **Streamlit** interface.

No hallucinations. No guessing. Every answer is grounded in the documents you feed it.

---

## 🧠 How RAG Works Here

```
  📰 URLs             🔪 Chunking             🧲 Embedding             🗄️ Storage
─────────────    ──────────────────────    ──────────────────    ─────────────────────
  News articles  →  RecursiveCharacter  →  Alibaba GTE-base  →  ChromaDB VectorStore
  Market reports     TextSplitter            (HuggingFace)        (persisted locally)
  Blog posts         (1000 chars,
                      100 overlap)

        │
        │  At query time
        ▼
  ❓ User Question  →  Similarity Search  →  Top-k Chunks  →  Llama 3.3 70B  →  ✅ Answer
                        (ChromaDB)            (context)         (via Groq)       + Sources
```

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🌐 **URL Ingestion** | Scrape any real estate news article or market report |
| ✂️ **Smart Chunking** | Recursive text splitting with configurable size & overlap |
| 🧲 **Semantic Embeddings** | `Alibaba-NLP/gte-base-en-v1.5` — a powerful open-source embedding model |
| 🗄️ **Persistent Vector Store** | ChromaDB stores embeddings locally for fast retrieval |
| 🤖 **LLM-Powered Answers** | Llama 3.3 70B via Groq for blazing-fast inference |
| 📎 **Source Attribution** | Every answer comes with the source URLs it drew from |
| 🖥️ **Streamlit UI** | Clean, interactive web interface — no CLI required |
| ☁️ **Streamlit Cloud Ready** | Secrets management works both locally and on Streamlit Cloud |

---

## 🗂️ Project Structure

```
real-estate-assistant-rag/
│
├── 📄 rag.py                    # Core RAG logic (ingest, embed, retrieve, answer)
├── 📄 main.py                   # Streamlit application UI
├── 📄 requirements.txt          # Python dependencies
│
├── 📁 resources/
│   └── 📁 vectorstore/          # Persisted ChromaDB embeddings
│
├── 📁 .devcontainer/            # Dev container configuration
├── 📁 .vscode/                  # VSCode workspace settings
└── 📄 .gitignore
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.10+**
- A free [Groq API key](https://console.groq.com/)

### 1. Clone the Repository

```bash
git clone https://github.com/anshul4uhh/real-estate-assistant-rag.git
cd real-estate-assistant-rag
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> 💡 Get your free API key at [console.groq.com](https://console.groq.com/)

### 5. Run the App

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501` 🎉

---

## 🖥️ Usage

**Step 1 — Enter URLs**

Paste one or more real estate news URLs into the sidebar (e.g., CNBC, Rightmove, Zillow articles).

**Step 2 — Process**

Click **"Process URLs"**. The app will:
- Scrape the content from each URL
- Split the text into chunks
- Generate embeddings and store them in ChromaDB

**Step 3 — Ask Questions**

Type any question about the real estate market in the chat box:

```
"How did the Federal Reserve's rate changes affect mortgage rates?"
"What are analysts predicting for house prices in 2026?"
"Which cities have seen the biggest property value growth?"
"How is AI investment impacting real estate tech companies?"
```

**Step 4 — Get Answers with Sources**

The assistant responds with a grounded answer and lists the source URLs it referenced.

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **LLM** | Llama 3.3 70B Versatile (via Groq) |
| **Embeddings** | `Alibaba-NLP/gte-base-en-v1.5` (HuggingFace) |
| **Vector Store** | ChromaDB (local persistence) |
| **Document Loading** | LangChain `UnstructuredURLLoader` |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Orchestration** | LangChain Core + PromptTemplate |
| **Env Management** | python-dotenv + Streamlit Secrets |

---

## ☁️ Deploying to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app
3. Set the main file to `main.py`
4. Add your secret in **Settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

> The app automatically detects Streamlit secrets, so no changes to the code are needed!

---

## 🔧 Configuration

Key parameters you can tune in `rag.py`:

```python
CHUNK_SIZE = 1000          # Characters per text chunk
CHUNK_OVERLAP = 100        # Overlap between chunks for context continuity
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"   # HuggingFace embedding model
COLLECTION_NAME = "real_estate"                     # ChromaDB collection name
LLM_TEMPERATURE = 0.2      # Lower = more factual answers
MAX_TOKENS = 500            # Max LLM response length
TOP_K = 3                  # Number of chunks retrieved per query
```

---

## 💡 Example Queries

| Query | What it finds |
|---|---|
| `"What did the Fed do with interest rates?"` | Rate policy changes and mortgage impact |
| `"How is Rightmove performing financially?"` | Company earnings, AI investments |
| `"What's the outlook for home prices?"` | Market predictions from scraped articles |
| `"Why are mortgage rates still high?"` | Economic context from news sources |

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m 'Add your feature'`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [anshul4uhh](https://github.com/anshul4uhh)

⭐ If this helped you, give it a star!

</div>
