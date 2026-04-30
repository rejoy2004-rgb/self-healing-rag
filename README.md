# 🧠 Self-Healing RAG System

A Retrieval-Augmented Generation (RAG) system with a **self-correction loop** using LangGraph, ChromaDB, and Groq LLM.

---

## 🚀 Features

* 📄 Document ingestion into vector database (ChromaDB)
* 🔍 Semantic search using embeddings
* 🤖 LLM-based answer generation (Groq - Llama 3.3)
* ✅ Self-grading mechanism to detect hallucinations
* 🔁 Automatic retry with question rewriting
* ❌ Fallback: "I don't know" if answer is unsupported

---

## 🏗️ Architecture

User Question
→ Retrieve Documents (ChromaDB)
→ Generate Answer (LLM)
→ Grade Answer
→ PASS → Return
→ FAIL → Rewrite → Retry

---

## 📂 Project Structure

```
self_healing_rag/
│
├── docs/
│   ├── python_basics.txt
│   ├── machine_learning.txt
│
├── ingest.py
├── rag_agent.py
├── .env
├── pyproject.toml
```

---

## ⚙️ Setup

### 1. Install dependencies

```bash
uv add langgraph langchain langchain-groq langchain-community chromadb python-dotenv
```

### 2. Add API key

Create `.env` file:

```
GROQ_API_KEY=your_key_here
```

---

## ▶️ Run

### Ingest documents

```bash
uv run ingest.py
```

### Run agent

```bash
uv run rag_agent.py
```

---

## 🧪 Example

**Input:**

```
What is machine learning?
```

**Output:**

```
Machine learning is a subset of artificial intelligence.
```

---

## 🧠 Tech Stack

* LangGraph
* LangChain
* ChromaDB
* Groq LLM (Llama 3.3)
* Python

---

## 🎯 Use Cases

* Internal knowledge base chatbot
* HR policy assistant
* Documentation Q&A system
* AI-powered search

---

## 📌 Future Improvements

* Streamlit UI
* PDF ingestion
* Better grading prompts
* Multi-document support

---
