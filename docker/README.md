# KG-RAG Cluster Docker Setup

This directory contains Docker Compose configuration for running a complete Knowledge Graph RAG system with:
- **MemGraph** - In-memory graph database with Lab UI
- **Neo4j** - Popular graph database with APOC support
- **Weaviate** - Vector database with Ollama integration
- **Ollama** - Local LLM runtime with embedding models

## Quick Start

```bash
# Start all services and pull embedding model
make quickstart

# Or manually:
make up              # Start all services
make pull-nomic      # Pull nomic-embed-text model
```

## Services Access

After starting with `make up`:

| Service | Type | URL | Credentials |
|---------|------|-----|-------------|
| **MemGraph Lab** | Web UI | http://localhost:3000 | - |
| **MemGraph Bolt** | Protocol | bolt://localhost:7687 | - |
| **MemGraph HTTP** | REST API | http://localhost:7444 | - |
| **Neo4j Browser** | Web UI | http://localhost:7474 | neo4j/testpass |
| **Neo4j Bolt** | Protocol | bolt://localhost:7688 | neo4j/testpass |
| **Weaviate API** | REST API | http://localhost:8080 | - |
| **Weaviate gRPC** | gRPC | localhost:50051 | - |
| **Ollama API** | REST API | http://localhost:11434 | - |

## Available Commands

Run `make help` to see all commands. Key commands:

```bash
make up              # Start all services
make down            # Stop all services
make status          # Check all services status
make logs            # View all logs (or make logs SERVICE=weaviate)
make restart         # Restart all services
make clean           # Remove all containers and volumes
make quickstart      # Start cluster + pull nomic model
```

### Service-Specific Commands

**MemGraph:**
```bash
make up-memgraph     # Start only MemGraph
make down-memgraph   # Stop only MemGraph
make logs-memgraph   # View MemGraph logs
make shell-memgraph  # Open MemGraph console (mgconsole)
```

**Neo4j:**
```bash
make up-neo4j        # Start only Neo4j
make down-neo4j      # Stop only Neo4j
make logs-neo4j      # View Neo4j logs
make shell-neo4j     # Open Neo4j shell (cypher-shell)
```

**Ollama:**
```bash
make up-ollama       # Start only Ollama
make down-ollama     # Stop only Ollama
make logs-ollama     # View Ollama logs
make shell-ollama    # Open Ollama shell
make list-models     # List installed models
```

**Weaviate:**
```bash
make up-weaviate     # Start only Weaviate
make down-weaviate   # Stop only Weaviate
make logs-weaviate   # View Weaviate logs
make shell-weaviate  # Open Weaviate shell
```

## Ollama Embedding Models

### Pull Models

```bash
make pull-models     # Pull all recommended models
make pull-nomic      # Pull nomic-embed-text (recommended)
make pull-mxbai      # Pull mxbai-embed-large
make pull-minilm     # Pull all-minilm (lightweight)
make list-models     # List installed models
```

### Recommended Models for RAG

| Model | Size | Dimensions | Use Case |
|-------|------|------------|----------|
| **nomic-embed-text** | 274MB | 768 | Best for RAG, high quality |
| **mxbai-embed-large** | 669MB | 1024 | Highest quality embeddings |
| **all-minilm** | 46MB | 384 | Lightweight, fast |

## Python Integration Examples

### Connecting to Graph Databases

#### MemGraph with Python

```python
from neo4j import GraphDatabase

# MemGraph uses Neo4j protocol (Bolt)
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=None  # MemGraph doesn't require auth by default
)

with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as count")
    print(result.single()["count"])

driver.close()
```

#### Neo4j with Python

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://localhost:7688",
    auth=("neo4j", "testpass")
)

with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as count")
    print(result.single()["count"])

driver.close()
```

#### Using LangChain with Neo4j

```python
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph(
    url="bolt://localhost:7688",
    username="neo4j",
    password="testpass"
)

# Run Cypher query
result = graph.query("MATCH (n) RETURN n LIMIT 5")
print(result)
```

### Using Weaviate with Ollama

#### Basic Example

```python
import weaviate
from weaviate.classes.init import Auth

# Connect to Weaviate
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

# Check if ready
print(client.is_ready())

# Close connection
client.close()
```

#### With LangChain

```python
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import OllamaEmbeddings
import weaviate

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Connect to Weaviate
client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051
)

# Create vector store
vector_store = Weaviate(
    client=client,
    index_name="Documents",
    text_key="text",
    embedding=embeddings
)

# Add documents
vector_store.add_texts(["Document 1", "Document 2"])

# Query
results = vector_store.similarity_search("query", k=3)
```

### Using Ollama for Embeddings

#### Basic Example

```python
import requests
import json

def get_embedding(text, model="nomic-embed-text"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]

# Get embedding
embedding = get_embedding("Your text here")
print(f"Embedding dimension: {len(embedding)}")
```

### With LangChain

```python
from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Embed documents
docs_embeddings = embeddings.embed_documents([
    "Document 1",
    "Document 2"
])

# Embed query
query_embedding = embeddings.embed_query("Your query")
```

### With LlamaIndex

```python
from llama_index.embeddings.ollama import OllamaEmbedding

embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0}
)

# Get embeddings
embeddings = embed_model.get_text_embedding("Your text")
```

### Complete RAG Pipeline Example

```python
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

# Initialize graph and embeddings
graph = Neo4jGraph(
    url="bolt://localhost:7688",
    username="neo4j",
    password="testpass"
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Create vector index in Neo4j
vector_store = Neo4jVector.from_documents(
    documents=your_documents,
    embedding=embeddings,
    graph=graph,
    index_name="document_embeddings"
)

# Query the vector store
results = vector_store.similarity_search("your query", k=5)
```

## Configuration

Copy [.env.example](.env.example) to `.env` and adjust settings:

```bash
cp .env.example .env
```

Key configurations:
- **MemGraph**: Ports (7687 Bolt, 7444 HTTP, 3000 Lab), log level
- **Neo4j**: Authentication, memory settings, plugins
- **Weaviate**: Ports (8080 HTTP, 50051 gRPC), Ollama integration
- **Ollama**: Host, port, GPU settings

## GPU Support (Ollama)

GPU acceleration is enabled by default in [docker-compose.yml](docker-compose.yml). To disable, comment out the deploy section:

```yaml
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]
```

Requires: `nvidia-docker` runtime installed.

## API Endpoints

### Ollama
- **Health Check**: `http://localhost:11434/`
- **Generate Embeddings**: `http://localhost:11434/api/embeddings`
- **List Models**: `http://localhost:11434/api/tags`
- **Pull Model**: `http://localhost:11434/api/pull`

### Neo4j
- **Browser**: `http://localhost:7474`
- **Bolt Protocol**: `bolt://localhost:7688`

### MemGraph
- **Lab UI**: `http://localhost:3000`
- **Bolt Protocol**: `bolt://localhost:7687`
- **HTTP API**: `http://localhost:7444`

### Weaviate
- **REST API**: `http://localhost:8080`
- **gRPC**: `localhost:50051`
- **Ready Check**: `http://localhost:8080/v1/.well-known/ready`

## Troubleshooting

### Check service status
```bash
make status
```

### Check if services are running
```bash
# Ollama
curl http://localhost:11434/

# Neo4j
curl http://localhost:7474/

# MemGraph
curl http://localhost:3000/

# Weaviate
curl http://localhost:8080/v1/.well-known/ready
```

### View logs
```bash
make logs                # All services
make logs SERVICE=neo4j  # Specific service
make logs-memgraph       # MemGraph only
make logs-neo4j          # Neo4j only
make logs-ollama         # Ollama only
```

### Restart services
```bash
make restart             # All services
make restart SERVICE=ollama  # Specific service
```

### Clean and start fresh
```bash
make clean  # Warning: removes ALL data and downloaded models
make up
```

### Common Issues

**MemGraph Lab not loading:**
- Check if port 3000 is available: `lsof -i :3000`
- Check logs: `make logs-memgraph`

**Neo4j authentication fails:**
- Default credentials: `neo4j/testpass`
- Change in [.env.example](.env.example) if needed

**Ollama GPU not working:**
- Verify nvidia-docker: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`
- Check GPU access: `make logs-ollama`

## Data Persistence

All data is stored in local volumes:

- **MemGraph**: `./volumes/memgraph/lib`, `./volumes/memgraph/log`, `./volumes/memgraph/etc`
- **Neo4j**: `./volumes/neo4j/data`
- **Weaviate**: `./volumes/weaviate`
- **Ollama**: `./volumes/ollama`

### Backup volumes
```bash
# Backup all data directories
tar czf memgraph_backup.tar.gz volumes/memgraph/
tar czf neo4j_backup.tar.gz volumes/neo4j/
tar czf weaviate_backup.tar.gz volumes/weaviate/
tar czf ollama_backup.tar.gz volumes/ollama/
```

## Additional Models

Pull other models as needed:

```bash
# LLM models for generation
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull mistral
docker exec ollama ollama pull phi3

# More embedding models
docker exec ollama ollama pull snowflake-arctic-embed
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              KG-RAG Cluster (kgrag_net)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌──────────────┐                │
│  │  MemGraph    │      │    Neo4j     │                │
│  │              │      │              │                │
│  │ Bolt:  7687  │      │ Bolt:  7688  │                │
│  │ HTTP:  7444  │      │ HTTP:  7474  │                │
│  │ Lab:   3000  │      │              │                │
│  └──────────────┘      └──────────────┘                │
│                                                         │
│  ┌──────────────┐      ┌──────────────────────────┐    │
│  │  Weaviate    │      │        Ollama            │    │
│  │              │◄─────│                          │    │
│  │ HTTP:  8080  │      │  API: 11434              │    │
│  │ gRPC: 50051  │      │  Models: nomic-embed     │    │
│  └──────────────┘      └──────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Use Cases

### Graph RAG with Neo4j
- Store documents as graph structures
- Semantic search with vector embeddings
- Graph-based retrieval augmentation

### Real-time Graph Analytics with MemGraph
- Stream processing with graph analytics
- High-performance graph queries
- In-memory graph computations
- HTTP API for programmatic access

### Vector Search with Weaviate
- Scalable vector storage and search
- Native Ollama integration for embeddings
- Hybrid search (vector + keyword)
- GraphQL and REST APIs

### Local LLM Embeddings with Ollama
- Privacy-preserving embeddings
- No API costs
- Fast local inference
- GPU acceleration support

## Resources

### MemGraph
- [MemGraph Documentation](https://memgraph.com/docs)
- [MemGraph Lab Guide](https://memgraph.com/docs/memgraph-lab)
- [Cypher Query Language](https://memgraph.com/docs/cypher-manual)

### Neo4j
- [Neo4j Documentation](https://neo4j.com/docs/)
- [APOC Procedures](https://neo4j.com/labs/apoc/)
- [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)

### Ollama
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Available Models](https://ollama.com/library)

### Weaviate
- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Weaviate Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)
- [Weaviate Ollama Integration](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-ollama)

### RAG & LangChain
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Neo4j Integration](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher)
- [LangChain Weaviate Integration](https://python.langchain.com/docs/integrations/vectorstores/weaviate)
- [Building RAG Systems](https://python.langchain.com/docs/use_cases/question_answering/)
