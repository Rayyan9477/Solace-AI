PYTHON ?= python
PROFILE ?= dev
TAG ?= dev
TREE ?=

.PHONY: help lint test test-unit test-realinfra test-e2e test-clinical test-contract images clean-checkout release-gate-closure migrate ci-local

help:
	$(PYTHON) -c "import re;[print(m) for m in re.findall(r'^[a-z-]+:', open('Makefile').read(), re.M)]"

lint:
	$(PYTHON) -m ruff check . --select E9,F63,F7,F82 --no-fix --no-cache

test: test-unit

test-unit:
	$(PYTHON) scripts/ci/run_suites.py --tier unit

test-realinfra:
	$(PYTHON) scripts/ci/run_suites.py --tier realinfra --require-realinfra

test-e2e:
	$(PYTHON) scripts/ci/run_suites.py --tier e2e --profile $(PROFILE)

test-clinical:
	$(PYTHON) scripts/ci/run_suites.py --tier clinical

test-contract:
	$(PYTHON) scripts/ci/run_suites.py --tier contract

images:
	$(PYTHON) -m scripts.build.images --tag $(TAG)

clean-checkout:
	$(PYTHON) scripts/ci/clean_checkout.py --tree $(TREE) --require-registry-match

release-gate-closure:
	$(PYTHON) scripts/ci/check_tracked_closure.py --strict

migrate:
	$(PYTHON) -m scripts.db.migrate upgrade head

ci-local:
	$(PYTHON) scripts/ci/ci_local.py
