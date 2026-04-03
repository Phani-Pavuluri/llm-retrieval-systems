# RAG System Shell: Modular LLM Retrieval System

## Overview
This project implements a **modular Retrieval-Augmented Generation (RAG) system** designed as a reusable shell for building LLM-powered applications across different domains.

The system combines:
- Semantic retrieval using embeddings
- Structured data filtering
- LLM-based reasoning and answer generation

While the initial implementation uses an Amazon product reviews dataset, the architecture is **domain-agnostic** and can be adapted to use cases such as:
- Marketing analytics
- Customer support copilots
- Knowledge base assistants
- Experimentation insights

---

## Motivation

Modern data science roles increasingly require building **end-to-end AI systems**, not just models.

This project focuses on:
- Understanding how LLM systems work under the hood
- Designing reliable retrieval pipelines
- Evaluating and improving answer quality
- Building reusable system architecture

---

## System Architecture

User Query → Embedding → Vector Search (FAISS) → Top-K Chunks → Prompt → LLM → Answer

---

## Key Components

### Data Layer
Loads and processes dataset and separates structured/unstructured data.

### Chunking
Splits text into overlapping chunks for retrieval.

### Embeddings
Supports:
- Sentence Transformers (local)
- OpenAI (optional)

### Vector Store
FAISS-based similarity search.

### Retriever
Fetches relevant chunks for a query.

### RAG Pipeline
Generates grounded answers using retrieved context.

---

## Project Structure

amazon-rag-shell/
├── data/
├── notebooks/
├── src/
├── scripts/
├── tests/
└── README.md

---

## Example Queries

- What complaints do users have about battery life?
- Which products mention durability issues?
- Summarize customer feedback trends

---

## Roadmap

Stage 1: Basic RAG  
Stage 2: Hybrid retrieval + filtering  
Stage 3: Routing + tools  
Stage 4: Evaluation  
Stage 5: Advanced orchestration (LangGraph)

---

## Getting Started

pip install -r requirements.txt

Add dataset to data/raw/

Run:
python scripts/build_index.py  
python scripts/run_query.py

---

## Author

Phani Pavuluri
