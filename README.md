# LangSmith

A comprehensive collection of LangChain and LangSmith examples demonstrating various LLM application patterns, from simple calls to complex agent workflows.

## Overview

This project showcases different approaches to building LLM applications using LangChain and LangSmith, including:

- **Simple LLM Calls**: Basic language model interactions
- **Sequential Chains**: Multi-step processing pipelines
- **RAG (Retrieval Augmented Generation)**: Document-based question answering with multiple implementations
- **Agents**: Autonomous agents with tool usage
- **LangGraph**: Complex workflows with state management

## Features

- 🔗 **LangChain Integration**: Full LangChain ecosystem support
- 📊 **LangSmith Tracing**: Built-in observability and debugging
- 📄 **PDF Processing**: RAG implementation with PDF document support
- 🤖 **Agent Framework**: ReAct agents with custom tools
- 🔄 **Workflow Management**: LangGraph for complex stateful workflows
- 🔍 **Vector Search**: FAISS-based semantic search

## Project Structure

```
.
├── 1_simple_llm_call.py      # Basic LLM interaction example
├── 2_sequential_chain.py     # Multi-step chain demonstration
├── 3_rag_v1.py               # Basic RAG implementation
├── 3_rag_v2.py               # RAG with improvements
├── 3_rag_v3.py               # RAG with additional features
├── 3_rag_v4.py               # RAG with caching and LangSmith tracing
├── 4_agent.py                # ReAct agent with tools
├── 5_langgraph.py            # LangGraph workflow example
├── islr.pdf                  # Sample PDF document for RAG
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Prerequisites

- Python 3.8 or higher
- OpenAI API key or OpenRouter API key
- (Optional) LangSmith account for tracing and observability

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd LangSmith
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here

# Optional: For LangSmith tracing
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=your_project_name
```

## Examples

### 1. Simple LLM Call (`1_simple_llm_call.py`)

Basic example of making a simple LLM call using LangChain:

```bash
python 1_simple_llm_call.py
```

**Features:**
- Uses OpenRouter API with Google Gemini model
- Simple prompt template
- String output parsing

### 2. Sequential Chain (`2_sequential_chain.py`)

Demonstrates chaining multiple LLM calls together:

```bash
python 2_sequential_chain.py
```

**Features:**
- Multi-step processing pipeline
- Different models for different steps
- LangSmith project tracking
- Metadata and tagging for observability

### 3. RAG Implementations (`3_rag_v*.py`)

Multiple versions of RAG (Retrieval Augmented Generation) implementations:

#### Version 1 (`3_rag_v1.py`)
Basic RAG with PDF document processing:
```bash
python 3_rag_v1.py
```

#### Version 4 (`3_rag_v4.py`)
Advanced RAG with:
- Index caching based on file fingerprints
- LangSmith tracing for all operations
- Optimized index loading
```bash
python 3_rag_v4.py
```

**Features:**
- PDF document loading and chunking
- FAISS vector store for semantic search
- Context-aware question answering
- Index persistence and caching

### 4. Agent (`4_agent.py`)

ReAct agent with custom tools:

```bash
python 4_agent.py
```

**Features:**
- DuckDuckGo search integration
- Custom weather API tool
- ReAct agent pattern
- Tool orchestration

**Note:** Update the weather API key in the script before running.

### 5. LangGraph (`5_langgraph.py`)

Complex workflow using LangGraph for essay evaluation:

```bash
python 5_langgraph.py
```

**Features:**
- Parallel evaluation across multiple dimensions
- State management with TypedDict
- Structured output parsing
- Multi-dimensional scoring system
- LangSmith tracing integration

## Key Technologies

- **LangChain**: Framework for building LLM applications
- **LangSmith**: Observability and debugging platform
- **LangGraph**: Stateful workflow management
- **FAISS**: Vector similarity search
- **OpenAI/OpenRouter**: LLM providers
- **PyPDF**: PDF document processing

## Dependencies

Key dependencies include:
- `langchain` & `langchain-community`: Core LangChain functionality
- `langchain-openai`: OpenAI integration
- `langgraph`: Workflow management
- `langsmith`: Tracing and observability
- `faiss-cpu`: Vector search
- `pypdf`: PDF processing
- `python-dotenv`: Environment variable management

See `requirements.txt` for the complete list.

## Usage Tips

1. **LangSmith Tracing**: Set up LangSmith credentials in your `.env` file to enable automatic tracing of all LangChain operations.

2. **Model Selection**: The examples use various models (Gemini, GPT-4o-mini). You can modify the model names in each script to use different providers.

3. **RAG Index Caching**: Version 4 of RAG automatically caches vector indices based on file fingerprints, making subsequent runs faster.

4. **Custom Tools**: The agent example shows how to create custom tools. You can extend this pattern for your own use cases.

5. **Workflow Customization**: The LangGraph example demonstrates how to build complex workflows. Modify the state schema and nodes to suit your needs.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
