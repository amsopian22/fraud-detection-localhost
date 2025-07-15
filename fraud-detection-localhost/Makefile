# Makefile
.PHONY: test lint format setup clean help

# Default environment
ENV ?= development

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Setup development environment
	@echo "Setting up fraud detection system..."
	chmod +x scripts/setup.sh
	./scripts/setup.sh
	docker-compose up -d
	@echo "Setup complete!"

test:  ## Run all tests
	docker-compose exec ml-api python -m pytest tests/ -v

test-unit:  ## Run unit tests only
	docker-compose exec ml-api python -m pytest tests/ -m "unit" -v

test-integration:  ## Run integration tests only
	docker-compose exec ml-api python -m pytest tests/ -m "integration" -v

test-api:  ## Run API tests only
	docker-compose exec ml-api python -m pytest tests/test_api/ -v

lint:  ## Run code linting
	docker-compose exec ml-api python -m flake8 src/ tests/
	docker-compose exec ml-api python -m pylint src/ tests/

format:  ## Format code
	docker-compose exec ml-api python -m black src/ tests/
	docker-compose exec ml-api python -m isort src/ tests/

generate-data:  ## Generate sample data
	docker-compose exec ml-api python scripts/generate_sample_data.py

train-model:  ## Train fraud detection model
	docker-compose exec ml-api python scripts/train_model.py

clean:  ## Clean up containers and data
	docker-compose down -v
	docker system prune -f

logs:  ## Show application logs
	docker-compose logs -f ml-api

logs-all:  ## Show all service logs
	docker-compose logs -f

stop:  ## Stop all services
	docker-compose stop

start:  ## Start all services
	docker-compose start

restart:  ## Restart all services
	docker-compose restart

shell:  ## Open shell in ml-api container
	docker-compose exec ml-api /bin/bash

jupyter:  ## Open Jupyter Lab
	@echo "Jupyter Lab is available at: http://localhost:8888"
	@echo "Token: fraudtoken123"

dashboard:  ## Open dashboard
	@echo "Streamlit dashboard is available at: http://localhost:8501"

api-docs:  ## Open API documentation
	@echo "API documentation is available at: http://localhost:8000/docs"

backup-data:  ## Backup database
	docker-compose exec postgres pg_dump -U frauduser frauddb > backup_$(shell date +%Y%m%d_%H%M%S).sql

restore-data:  ## Restore database (usage: make restore-data FILE=backup.sql)
	docker-compose exec -T postgres psql -U frauduser frauddb < $(FILE)

check-health:  ## Check system health
	@echo "Checking system health..."
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo "\nChecking dashboard..."
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
	@echo " - Dashboard status"

monitoring:  ## Open monitoring dashboard
	@echo "Prometheus is available at: http://localhost:9090"

update:  ## Update system
	git pull
	docker-compose build
	docker-compose up -d

deploy-prod:  ## Deploy to production (placeholder)
	@echo "Production deployment not implemented in localhost version"
	@echo "This would typically involve:"
	@echo "  - Building production images"
	@echo "  - Deploying to cloud infrastructure"
	@echo "  - Running database migrations"
	@echo "  - Health checks and validation"