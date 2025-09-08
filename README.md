# OpenAI Agent PaperQA

An agentic RAG (Retrieval-Augmented Generation) system for public health research that reproduces PaperQA using OpenAI's Agent SDK. The system searches through scientific literature using a search-gather-response framework to provide evidence-based answers.

## Setup
Download or clone the repository. Install all the dependencies using uv. We will need to install uv ([Installing uv](https://docs.astral.sh/uv/getting-started/installation/)).

### 1. Install Dependencies
```bash
cd openai-agent-paperqa
uv sync
```

### 2. Run Notebook
```bash
jupyter notebook agent-paperqa.ipynb
```

## What it does

- Downloads and processes 2,914 CDC public health articles (2004-2023)
- Creates a searchable vector database with 169,394 document chunks
- Implements an AI agent that can answer complex research questions by searching, gathering evidence, and synthesizing responses from scientific literature