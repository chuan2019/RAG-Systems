# RAG Systems

Local development environments for building Retrieval-Augmented Generation systems with Knowledge Graphs.

## Overview

- **KG-RAG**: Knowledge Graph RAG with MemGraph, Neo4j, and Ollama
- **RAG**: Traditional RAG implementations (coming soon)

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd RAG-Systems
uv sync  # or: pip install -e .

# Start KG-RAG cluster
cd kg-rag/docker
make quickstart
```

## KG-RAG Cluster

Complete local development stack with graph databases and local LLMs.

**Services:**
- **MemGraph** (bolt://localhost:7687, UI: http://localhost:3000)
- **Neo4j** (bolt://localhost:7688, UI: http://localhost:7474, auth: neo4j/testpass)
- **Ollama** (http://localhost:11434)

**Features:**
- Docker Compose orchestration
- Pre-configured APOC plugins
- GPU acceleration support
- Embedding models included
- Makefile commands for easy management

See [kg-rag/docker/README.md](kg-rag/docker/README.md) for detailed documentation.

## Common Commands

```bash
cd kg-rag/docker

make up          # Start all services
make down        # Stop all services
make status      # Check status
make logs        # View logs
make help        # Show all commands
```

## Project Structure

```
RAG-Systems/
├── kg-rag/
│   ├── docker/          # Docker setup
│   └── notebooks/       # Jupyter notebooks
├── rag/                 # Traditional RAG
└── docs/                # Documentation
```

## Resources

- [KG-RAG Documentation](kg-rag/docker/README.md)
- [MemGraph Docs](https://memgraph.com/docs)
- [Neo4j Docs](https://neo4j.com/docs/)
- [Ollama Docs](https://github.com/ollama/ollama)
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
