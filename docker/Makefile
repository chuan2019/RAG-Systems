.PHONY: up up-kg-rag up-kg-rag-memgraph up-rag up-all down logs restart status clean quickstart quickstart-memgraph quickstart-rag help
.PHONY: up-neo4j down-neo4j logs-neo4j shell-neo4j
.PHONY: up-memgraph down-memgraph logs-memgraph shell-memgraph lab-memgraph
.PHONY: up-ollama down-ollama logs-ollama shell-ollama pull-models pull-nomic pull-mxbai pull-minilm list-models
.PHONY: up-weaviate down-weaviate logs-weaviate shell-weaviate

# =============================================================================
# Main Commands
# =============================================================================

# Start all services (requires profile selection)
up:
	@echo "ERROR: Please specify a profile:"
	@echo "  make up-kg-rag           - Start Knowledge Graph RAG stack (Neo4j, Ollama)"
	@echo "  make up-kg-rag-memgraph  - Start Knowledge Graph RAG stack (MemGraph, Ollama)"
	@echo "  make up-rag              - Start Vector RAG stack (Weaviate, Ollama)"
	@echo "  make up-all              - Start all services"
	@exit 1

# Start KG-RAG profile (Neo4j, Ollama)
up-kg-rag:
	@echo "Starting KG-RAG profile (Neo4j, Ollama)..."
	docker-compose --profile kg-rag-neo4j up -d --build
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "======================== KG-RAG Profile Ready ========================"
	@echo "Neo4j Browser: http://localhost:7474 (user: neo4j, pass: testpass)"
	@echo "Neo4j Bolt:    bolt://localhost:7687"
	@echo "Ollama API:    http://localhost:11434"
	@echo "======================================================================"

# Start KG-RAG profile with MemGraph (MemGraph, Ollama)
up-kg-rag-memgraph:
	@echo "Starting KG-RAG profile (MemGraph, Ollama)..."
	docker-compose --profile kg-rag-memgraph up -d --build
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "======================== KG-RAG Profile Ready ========================"
	@echo "MemGraph Lab:  http://localhost:3000"
	@echo "MemGraph Bolt: bolt://localhost:7687"
	@echo "Ollama API:    http://localhost:11434"
	@echo "======================================================================"

# Start RAG profile (Weaviate, Ollama)
up-rag:
	@echo "Starting RAG profile (Weaviate, Ollama)..."
	docker-compose --profile rag up -d --build
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "======================== RAG Profile Ready ========================"
	@echo "Weaviate API:  http://localhost:8080"
	@echo "Weaviate gRPC: localhost:50051"
	@echo "Ollama API:    http://localhost:11434"
	@echo "==================================================================="

# Start all services (all profiles)
up-all:
	@echo "Starting all services (kg-rag-neo4j + kg-rag-memgraph + rag profiles)..."
	docker-compose --profile kg-rag-neo4j --profile kg-rag-memgraph --profile rag up -d --build
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "======================== All Services Ready ========================"
	@echo "Neo4j Browser: http://localhost:7474 (user: neo4j, pass: testpass)"
	@echo "Neo4j Bolt:    bolt://localhost:7687"
	@echo "MemGraph Lab:  http://localhost:3000"
	@echo "MemGraph Bolt: bolt://localhost:7687 (conflicts with Neo4j)"
	@echo "Weaviate API:  http://localhost:8080"
	@echo "Weaviate gRPC: localhost:50051"
	@echo "Ollama API:    http://localhost:11434"
	@echo "====================================================================="

# Stop all services
down:
	@echo "Stopping KG-RAG cluster..."
	docker-compose down

# View logs (all services or specific: make logs SERVICE=ollama)
logs:
ifdef SERVICE
	docker-compose logs -f $(SERVICE)
else
	docker-compose logs -f
endif

# Restart all services
restart:
	@echo "Restarting KG-RAG cluster..."
	docker-compose restart

# Check status of all services
status:
	@echo "KG-RAG Cluster Status:"
	@echo "======================"
	@docker-compose ps
	@echo ""
	@echo "Ollama Models:"
	@docker exec ollama ollama list 2>/dev/null || echo "Ollama not running"

# Clean everything (including volumes)
clean:
	@echo "WARNING: This will remove ALL data and downloaded models!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		echo "Cleaned successfully"; \
	fi

# Quick start - start KG-RAG cluster and pull embedding model
quickstart: up-kg-rag
	@echo "Waiting for Ollama to fully start..."
	@sleep 8
	@$(MAKE) pull-nomic
	@echo ""
	@echo "Quickstart complete! KG-RAG services running with nomic-embed-text model."

# Quick start for MemGraph KG-RAG profile
quickstart-memgraph: up-kg-rag-memgraph
	@echo "Waiting for Ollama to fully start..."
	@sleep 8
	@$(MAKE) pull-nomic
	@echo ""
	@echo "Quickstart complete! KG-RAG services (MemGraph) running with nomic-embed-text model."

# Quick start for RAG profile
quickstart-rag: up-rag
	@echo "Waiting for Ollama to fully start..."
	@sleep 8
	@$(MAKE) pull-nomic
	@echo ""
	@echo "Quickstart complete! RAG services running with nomic-embed-text model."

# Display help
help:
	@echo "KG-RAG Cluster Management"
	@echo "========================="
	@echo ""
	@echo "Profile Commands:"
	@echo "  make up-kg-rag           - Start KG-RAG profile (Neo4j, Ollama)"
	@echo "  make up-kg-rag-memgraph  - Start KG-RAG profile (MemGraph, Ollama)"
	@echo "  make up-rag              - Start RAG profile (Weaviate, Ollama)"
	@echo "  make up-all              - Start all services (all profiles)"
	@echo "  make quickstart          - Start KG-RAG (Neo4j) + pull nomic model"
	@echo "  make quickstart-memgraph - Start KG-RAG (MemGraph) + pull nomic model"
	@echo "  make quickstart-rag      - Start RAG + pull nomic model"
	@echo ""
	@echo "Main Commands:"
	@echo "  make down            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make status          - Show status of all services"
	@echo "  make logs            - View logs (all services)"
	@echo "  make logs SERVICE=<name> - View logs for specific service"
	@echo "  make clean           - Remove all containers and volumes"
	@echo ""
	@echo "Note: Services are organized by profiles:"
	@echo "  kg-rag-neo4j:     Neo4j, Ollama"
	@echo "  kg-rag-memgraph:  MemGraph, Ollama"
	@echo "  rag:              Weaviate, Ollama"
	@echo ""
	@echo "Neo4j Commands:"
	@echo "  make up-neo4j        - Start only Neo4j"
	@echo "  make down-neo4j      - Stop only Neo4j"
	@echo "  make logs-neo4j      - View Neo4j logs"
	@echo "  make shell-neo4j     - Open Neo4j shell"
	@echo ""
	@echo "MemGraph Commands:"
	@echo "  make up-memgraph     - Start only MemGraph"
	@echo "  make down-memgraph   - Stop only MemGraph"
	@echo "  make logs-memgraph   - View MemGraph logs"
	@echo "  make shell-memgraph  - Open MemGraph shell"
	@echo "  make lab-memgraph    - Open MemGraph Lab in browser"
	@echo ""
	@echo "Ollama Commands:"
	@echo "  make up-ollama       - Start only Ollama"
	@echo "  make down-ollama     - Stop only Ollama"
	@echo "  make logs-ollama     - View Ollama logs"
	@echo "  make shell-ollama    - Open Ollama shell"
	@echo "  make pull-models     - Pull all embedding models"
	@echo "  make pull-nomic      - Pull nomic-embed-text (recommended)"
	@echo "  make pull-mxbai      - Pull mxbai-embed-large"
	@echo "  make pull-minilm     - Pull all-minilm (lightweight)"
	@echo "  make list-models     - List installed models"

# =============================================================================
# Neo4j Commands
# =============================================================================

up-neo4j:
	@echo "Starting Neo4j..."
	docker-compose --profile kg-rag-neo4j up -d neo4j
	@echo "Neo4j Browser: http://localhost:7474"
	@echo "Neo4j Bolt: bolt://localhost:7687"
	@echo "User: neo4j, Password: testpass"

down-neo4j:
	@echo "Stopping Neo4j..."
	docker-compose stop neo4j

logs-neo4j:
	docker-compose logs -f neo4j

shell-neo4j:
	docker exec -it neo4j-single cypher-shell -u neo4j -p testpass

# =============================================================================
# MemGraph Commands
# =============================================================================

up-memgraph:
	@echo "Starting MemGraph..."
	docker-compose --profile kg-rag-memgraph up -d memgraph
	@echo "MemGraph Lab: http://localhost:3000"
	@echo "MemGraph Bolt: bolt://localhost:7687"
	@echo "Note: Compatible with Neo4j drivers and tools"

down-memgraph:
	@echo "Stopping MemGraph..."
	docker-compose stop memgraph

logs-memgraph:
	docker-compose logs -f memgraph

shell-memgraph:
	docker exec -it memgraph mgconsole

lab-memgraph:
	@echo "Opening MemGraph Lab..."
	@xdg-open http://localhost:3000 || open http://localhost:3000 || echo "Please open http://localhost:3000 in your browser"

# =============================================================================
# Ollama Commands
# =============================================================================

up-ollama:
	@echo "Starting Ollama..."
	docker-compose --profile kg-rag-neo4j up -d ollama
	@echo "Ollama API: http://localhost:11434"
	@echo "Note: Ollama is available in kg-rag-neo4j, kg-rag-memgraph, and rag profiles"

down-ollama:
	@echo "Stopping Ollama..."
	docker-compose stop ollama

logs-ollama:
	docker-compose logs -f ollama

shell-ollama:
	docker exec -it ollama /bin/bash

# Pull all recommended embedding models
pull-models: pull-nomic pull-mxbai pull-minilm
	@echo "All embedding models downloaded successfully!"

# Pull nomic-embed-text (highly recommended for RAG, 274MB)
pull-nomic:
	@echo "Pulling nomic-embed-text model..."
	docker exec ollama ollama pull nomic-embed-text

# Pull mxbai-embed-large (high quality, 669MB)
pull-mxbai:
	@echo "Pulling mxbai-embed-large model..."
	docker exec ollama ollama pull mxbai-embed-large

# Pull all-minilm (lightweight, 46MB)
pull-minilm:
	@echo "Pulling all-minilm model..."
	docker exec ollama ollama pull all-minilm

# List available models
list-models:
	@echo "Available models in Ollama:"
	docker exec ollama ollama list

# =============================================================================
# Weaviate Commands
# =============================================================================

up-weaviate:
	@echo "Starting Weaviate (with Ollama dependency)..."
	docker-compose --profile rag up -d weaviate
	@echo "Weaviate API: http://localhost:8080"
	@echo "Weaviate gRPC: localhost:50051"

down-weaviate:
	@echo "Stopping Weaviate..."
	docker-compose stop weaviate

logs-weaviate:
	docker-compose logs -f weaviate

shell-weaviate:
	docker exec -it weaviate /bin/sh
