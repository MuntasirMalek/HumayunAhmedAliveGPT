# HumayunAhmedAliveGPT

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model%20Repository-yellow)](https://huggingface.co/YOUR_HF_USERNAME/HumayunAhmedAliveGPT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language: Bengali](https://img.shields.io/badge/Language-Bengali-green.svg)](#)

**HumayunAhmedAliveGPT** is an open-source Bengali persona language model and Retrieval-Augmented Generation (RAG) system developed to replicate the conversational style, dry humor, storytelling cadence, and philosophical outlook of legendary Bangladeshi author and filmmaker Humayun Ahmed.

---

## Project Overview

Standard language models often respond in a generic, robotic tone and struggle with factual accuracy regarding regional literature. This project solves both challenges through a hybrid architecture:

1. **Supervised Fine-Tuning (4-bit QLoRA):** The model is fine-tuned on a custom first-person Bengali instruction dataset (`humayun_train.jsonl`) curated from Humayun Ahmed's autobiographies, interviews, and literary works to capture his exact linguistic persona.
2. **Retrieval-Augmented Generation (RAG):** To eliminate hallucinations on biographical dates, character arcs, and bibliography, the model is paired with a FAISS vector database (`humayun_rag.jsonl`) using multilingual dense embeddings.

---

## System Architecture

```text
User Query (Bengali)
       │
       ▼
Multilingual Sentence Embedder (paraphrase-multilingual-MiniLM-L12-v2)
       │
       ▼
FAISS L2 Vector Index Search (humayun_rag.jsonl)
       │
       ▼
Context Injection + System Persona Prompt
       │
       ▼
Fine-Tuned HumayunAhmedAliveGPT Model (QLoRA Adapters)
       │
       ▼
Authentic First-Person Response in Bengali
