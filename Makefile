.PHONY: dev test lint migrate clean docker-up docker-down help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: docker-up  ## Start development environment
	@echo "Development environment started"

test:  ## Run all tests
	python -m pytest tests/ services/ -x --timeout=60 -q

lint:  ## Run linting
	python -m ruff check . --fix
	python -m ruff format .

typecheck:  ## Run type checking
	python -m mypy src/ --ignore-missing-imports

migrate:  ## Run database migrations
	alembic upgrade head

docker-up:  ## Start Docker services
	docker-compose up -d

docker-down:  ## Stop Docker services
	docker-compose down

clean:  ## Clean up caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
