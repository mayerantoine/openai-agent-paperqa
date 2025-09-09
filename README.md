# OpenAI Agent PaperQA

An agentic RAG (Retrieval-Augmented Generation) system for public health research that reproduces PaperQA using OpenAI's Agent SDK. The system searches through scientific literature using a search-gather-response framework to provide evidence-based answers.

## Setup
Download or clone the repository. Install all the dependencies using uv. We will need to install uv ([Installing uv](https://docs.astral.sh/uv/getting-started/installation/)).

### 1. Install Dependencies
```bash
cd openai-agent-paperqa
uv sync
```
## Usage

You can run this system in two ways:

### Option 1: Run the Jupyter Notebook (Interactive)
If you want to explore the system interactively and see step-by-step execution:

```bash
uv run jupyter lab agent-paperqa.ipynb
```

### Option 2: Run the Main Script (Direct execution)
If you want to run the complete pipeline directly:

```bash
uv run main.py
```

This will automatically:
- Download CDC data if needed
- Process and index the articles
- Answer the example research question

## What it does

- Downloads and processes 2,914 CDC public health articles (2004-2023)
- Creates a searchable vector database with 169,394 document chunks
- Implements an AI agent that can answer complex research questions by searching, gathering evidence, and synthesizing responses from scientific literature