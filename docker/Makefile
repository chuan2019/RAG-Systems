.PHONY: up down logs restart status clean quickstart help
.PHONY: up-memgraph down-memgraph logs-memgraph shell-memgraph
.PHONY: up-neo4j down-neo4j logs-neo4j shell-neo4j
.PHONY: up-ollama down-ollama logs-ollama shell-ollama pull-models pull-nomic pull-mxbai pull-minilm list-models

# =============================================================================
# Main Commands
# =============================================================================

# Start all services (MemGraph, Neo4j, Ollama)
up:
	@echo "Starting KG-RAG cluster (MemGraph, Neo4j, Ollama)..."
	docker-compose up -d --build
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo ""
	@echo "======================== Cluster Ready ========================"
	@echo "MemGraph Lab:  http://localhost:3000"
	@echo "MemGraph Bolt: bolt://localhost:7687"
	@echo "Neo4j Browser: http://localhost:7474 (user: neo4j, pass: testpass)"
	@echo "Neo4j Bolt:    bolt://localhost:7688"
	@echo "Ollama API:    http://localhost:11434"
	@echo "==============================================================="

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

# Quick start - start cluster and pull embedding model
quickstart: up
	@echo "Waiting for Ollama to fully start..."
	@sleep 8
	@$(MAKE) pull-nomic
	@echo ""
	@echo "Quickstart complete! All services running with nomic-embed-text model."

# Display help
help:
	@echo "KG-RAG Cluster Management"
	@echo "========================="
	@echo ""
	@echo "Main Commands:"
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make restart         - Restart all services"
	@echo "  make status          - Show status of all services"
	@echo "  make logs            - View logs (all services)"
	@echo "  make logs SERVICE=<name> - View logs for specific service"
	@echo "  make clean           - Remove all containers and volumes"
	@echo "  make quickstart      - Start cluster + pull nomic model"
	@echo ""
	@echo "MemGraph Commands:"
	@echo "  make up-memgraph     - Start only MemGraph"
	@echo "  make down-memgraph   - Stop only MemGraph"
	@echo "  make logs-memgraph   - View MemGraph logs"
	@echo "  make shell-memgraph  - Open MemGraph shell"
	@echo ""
	@echo "Neo4j Commands:"
	@echo "  make up-neo4j        - Start only Neo4j"
	@echo "  make down-neo4j      - Stop only Neo4j"
	@echo "  make logs-neo4j      - View Neo4j logs"
	@echo "  make shell-neo4j     - Open Neo4j shell"
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
# MemGraph Commands
# =============================================================================

up-memgraph:
	@echo "Starting MemGraph..."
	docker-compose up -d memgraph
	@echo "MemGraph Lab: http://localhost:3000"
	@echo "MemGraph Bolt: bolt://localhost:7687"

down-memgraph:
	@echo "Stopping MemGraph..."
	docker-compose stop memgraph

logs-memgraph:
	docker-compose logs -f memgraph

shell-memgraph:
	docker exec -it memgraph-single mgconsole

# =============================================================================
# Neo4j Commands
# =============================================================================

up-neo4j:
	@echo "Starting Neo4j..."
	docker-compose up -d neo4j
	@echo "Neo4j Browser: http://localhost:7474"
	@echo "Neo4j Bolt: bolt://localhost:7688"
	@echo "User: neo4j, Password: testpass"

down-neo4j:
	@echo "Stopping Neo4j..."
	docker-compose stop neo4j

logs-neo4j:
	docker-compose logs -f neo4j

shell-neo4j:
	docker exec -it neo4j-single cypher-shell -u neo4j -p testpass

# =============================================================================
# Ollama Commands
# =============================================================================

up-ollama:
	@echo "Starting Ollama..."
	docker-compose up -d ollama
	@echo "Ollama API: http://localhost:11434"

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
