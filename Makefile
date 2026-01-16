# Multilingual Sentiment Analysis Tool - Makefile

.PHONY: help install test lint format clean run-api run-frontend run-demo docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run tests"
	@echo "  test-fast     - Run fast tests only"
	@echo "  lint          - Run linting"
	@echo "  format        - Format code"
	@echo "  clean         - Clean cache and temporary files"
	@echo "  run-api       - Start API server"
	@echo "  run-frontend  - Start web interface"
	@echo "  run-demo      - Run demo script"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run with Docker Compose"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov black flake8 mypy

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-coverage:
	pytest tests/ --cov=app --cov-report=html --cov-report=term

test-integration:
	pytest tests/ -v -m "integration"

# Code quality
lint:
	flake8 app/ tests/
	mypy app/

format:
	black app/ tests/ run_demo.py

format-check:
	black --check app/ tests/ run_demo.py

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Running services
run-api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	streamlit run app/frontend/streamlit_app.py --server.port 8501

run-demo:
	python run_demo.py

# Docker
docker-build:
	docker build -t multilingual-sentiment-analysis .

docker-run:
	docker-compose up --build

docker-run-detached:
	docker-compose up -d --build

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development
dev-setup: install-dev
	cp .env.example .env
	mkdir -p data results models logs

dev-run:
	# Run API and frontend in parallel
	make run-api & make run-frontend

# Production
prod-install:
	pip install -r requirements.txt --no-dev

prod-test:
	pytest tests/ -v -m "not slow and not integration"

# Documentation
docs-serve:
	python -m http.server 8080 -d docs/

# Benchmarking
benchmark:
	python -m pytest tests/ -v -m "benchmark" --benchmark-only

# Security
security-check:
	pip-audit
	bandit -r app/

# Database (if using)
db-migrate:
	alembic upgrade head

db-reset:
	alembic downgrade base
	alembic upgrade head

# Monitoring
health-check:
	curl -f http://localhost:8000/health || exit 1

# Deployment helpers
deploy-check: prod-test lint security-check
	@echo "✅ All deployment checks passed"

# All-in-one commands
setup: dev-setup
	@echo "✅ Development environment setup complete"

check: test lint format-check
	@echo "✅ All checks passed"

ci: install test lint
	@echo "✅ CI pipeline complete"
