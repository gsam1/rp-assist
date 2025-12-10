.PHONY: start stop restart build logs clean reindex index-new shell-backend shell-frontend validate-config help

help:
	@echo "RPG Assistant - Available commands:"
	@echo "  make start          - Start all services"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make build          - Build Docker images"
	@echo "  make logs           - Tail logs from all services"
	@echo "  make clean          - Remove containers and volumes"
	@echo "  make reindex        - Full reindex of all documents"
	@echo "  make index-new      - Incremental index of new documents"
	@echo "  make shell-backend  - Open shell in backend container"
	@echo "  make shell-frontend - Open shell in frontend container"
	@echo "  make validate-config- Validate configuration files"

start:
	docker compose up -d

stop:
	docker compose down

restart:
	docker compose restart

build:
	docker compose build

logs:
	docker compose logs -f

clean:
	docker compose down -v
	rm -rf data/vectordb/*
	rm -rf data/metadata/*

reindex:
	docker compose exec backend python -m app.cli reindex --full

index-new:
	docker compose exec backend python -m app.cli reindex --incremental

shell-backend:
	docker compose exec backend /bin/bash

shell-frontend:
	docker compose exec frontend /bin/sh

validate-config:
	docker compose exec backend python -m app.cli validate-config
