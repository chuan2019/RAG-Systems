# RAG Systems

Local development environments for building Retrieval-Augmented Generation systems with Knowledge Graphs.

## Overview

Complete development stack featuring:
- **MemGraph** and **Neo4j** for Knowledge Graph RAG
- **Weaviate** for vector search
- **Ollama** for local LLM inference
- Jupyter notebooks for experimentation

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd RAG-Systems
uv sync  # or: pip install -e .

# Start all services
cd docker
make quickstart
```

## Services

Complete local development stack with graph databases and local LLMs.

**Available Services:**
- **MemGraph** (bolt://localhost:7687, UI: http://localhost:3000)
- **Neo4j** (bolt://localhost:7688, UI: http://localhost:7474, auth: neo4j/testpass)
- **Weaviate** (http://localhost:8080)
- **Ollama** (http://localhost:11434)

**Features:**
- Docker Compose orchestration
- Pre-configured APOC plugins for Neo4j
- GPU acceleration support
- Embedding models included
- Makefile commands for easy management

See [docker/README.md](docker/README.md) for detailed documentation.

## Common Commands

```bash
cd docker

make up          # Start all services
make down        # Stop all services
make status      # Check status
make logs        # View logs
make help        # Show all commands
```

## Project Structure

```
RAG-Systems/
├── docker/              # Docker Compose setup & orchestration
│   ├── docker-compose.yml
│   ├── Makefile
│   ├── README.md
│   ├── models/          # Ollama models directory
│   └── volumes/         # Persistent data
│       ├── memgraph/
│       ├── neo4j/
│       ├── ollama/
│       └── weaviate/
├── notebooks/           # Jupyter notebooks for experimentation
│   ├── Quick-Intro_MemGraph.ipynb
│   └── Quick-Intro_Neo4j.ipynb
├── docs/                # Additional documentation
├── pyproject.toml       # Python dependencies
└── README.md
```

## Resources

- [Docker Setup Documentation](docker/README.md)
- [MemGraph Docs](https://memgraph.com/docs)
- [Neo4j Docs](https://neo4j.com/docs/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [Ollama Docs](https://github.com/ollama/ollama)
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
