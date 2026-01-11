# RAG Systems

Local development environments for building Retrieval-Augmented Generation systems with Knowledge Graphs.

## Overview

Complete development stack featuring:
- **Neo4j** for Knowledge Graph RAG
- **Weaviate** for vector search
- **Ollama** for local LLM inference
- Jupyter notebooks for experimentation

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd RAG-Systems
uv sync  # or: pip install -e .

# Start services (choose one):
cd docker
make quickstart      # KG-RAG stack (Neo4j, Ollama)
make quickstart-rag  # Vector RAG stack (Weaviate, Ollama)
make up-all          # All services
```

## Services

Complete local development stack with graph databases and local LLMs, organized by profiles:

**KG-RAG Profile** (Knowledge Graph RAG):
- **Neo4j** (bolt://localhost:7687, UI: http://localhost:7474, auth: neo4j/testpass)
- **Ollama** (http://localhost:11434)

**RAG Profile** (Vector RAG):
- **Weaviate** (http://localhost:8080)
- **Ollama** (http://localhost:11434)

**Features:**
- Profile-based service selection
- Docker Compose orchestration
- Pre-configured APOC plugins for Neo4j
- GPU acceleration support
- Embedding models included
- Makefile commands for easy management

See [docker/README.md](docker/README.md) for detailed documentation.

## Common Commands

```bash
cd docker

# Start services by profile
make up-kg-rag   # Knowledge Graph RAG (Neo4j, Ollama)
make up-rag      # Vector RAG (Weaviate, Ollama)
make up-all      # All services

# Management
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
│       ├── neo4j/
│       ├── ollama/
│       └── weaviate/
├── notebooks/           # Jupyter notebooks for experimentation
│   └── Quick-Intro_Neo4j.ipynb
├── docs/                # Additional documentation
├── pyproject.toml       # Python dependencies
└── README.md
```

## Resources

- [Docker Setup Documentation](docker/README.md)
- [Neo4j Docs](https://neo4j.com/docs/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [Ollama Docs](https://github.com/ollama/ollama)
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
