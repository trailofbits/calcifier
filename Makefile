.PHONY: install lockcheck lint typecheck portable test generation-reliability-test mutate mutate-changed mutate-stats check verify clean

# CI reports each module's mutation kill rate against 90% without enforcing it.
MUTATION_TARGETS := src/calcifier
UV_RUN := uv run --frozen
# Detect platform-dependent text I/O in exercised core paths, including subprocesses.
export PYTHONWARNDEFAULTENCODING := 1
CHECK_PATHS := src/ tests/ scripts/
GREMLIN_WORKERS ?= 4
GREMLIN_OPERATORS ?=
GREMLIN_CACHE ?= 0
MUTATION_BASE ?= HEAD
MUTATION_FILES ?=
MUTATION_TESTS ?=
MUTATION_RUN_TARGETS ?= $(MUTATION_TARGETS)

comma := ,
empty :=
space := $(empty) $(empty)
GREMLIN_OPERATOR_ARG := $(if $(strip $(GREMLIN_OPERATORS)),--gremlin-operators=$(GREMLIN_OPERATORS))
GREMLIN_CACHE_ARG := $(if $(filter 1 true yes,$(GREMLIN_CACHE)),--gremlin-cache,--gremlin-clear-cache)
CHANGED_SOURCE_PATHS := $(sort $(shell git diff --name-only --diff-filter=ACMR $(MUTATION_BASE) -- src/calcifier) $(shell git ls-files --others --exclude-standard -- src/calcifier))
INCREMENTAL_SOURCE_PATHS := $(if $(strip $(MUTATION_FILES)),$(strip $(MUTATION_FILES)),$(CHANGED_SOURCE_PATHS))
CHANGED_MUTATION_PATHS := $(filter src/calcifier/%.py,$(INCREMENTAL_SOURCE_PATHS))
CHANGED_MUTATION_TARGETS := $(subst $(space),$(comma),$(strip $(CHANGED_MUTATION_PATHS)))
INCREMENTAL_REPOSITORY_TESTS := $(if $(strip $(MUTATION_TESTS)),$(strip $(MUTATION_TESTS)),tests/)
install:
	uv --no-config sync --all-groups --locked

# Validate the committed lock without inheriting workstation configuration files.
# Frozen test runs alone do not check that a clean CI runner can install it.
lockcheck:
	uv --no-config lock --check

lint:
	$(UV_RUN) ruff check $(CHECK_PATHS)
	$(UV_RUN) ruff check --preview --select PLW1514 src/calcifier
	$(UV_RUN) ruff format --check $(CHECK_PATHS)

typecheck:
	$(UV_RUN) ty check $(CHECK_PATHS)

portable:
	$(UV_RUN) calcifier lint plugins/calcifier/skills/calcifier/.calcifier/recipe.py --portable
	$(UV_RUN) calcifier lint plugins/calcifier/skills/debug/.calcifier/recipe.py --portable
	$(UV_RUN) calcifier lint plugins/calcifier/skills/exec/.calcifier/recipe.py --portable
	$(UV_RUN) calcifier lint plugins/calcifier/skills/update/.calcifier/recipe.py --portable

test:
	$(UV_RUN) pytest -q -n auto

generation-reliability-test:
	$(UV_RUN) pytest -q tests/test_corpus.py

# Mutation testing via pytest-gremlins. The `subprocess` executor is the
# working option on macOS Tahoe (26+); `inprocess` is faster but does not
# actually re-run tests per mutant, and `fork` (the unix default) hits the
# same asyncio-fork-safety bug that drove us off mutmut.
# tests/conftest.py forces the subprocess shortcut through real pytest so
# fixtures, parameterized cases, and offline guards run for every mutant.
mutate:
	$(UV_RUN) pytest --gremlins \
		--gremlin-executor=subprocess \
		--gremlin-workers=$(GREMLIN_WORKERS) \
		--gremlin-targets=$(MUTATION_RUN_TARGETS) \
		$(GREMLIN_OPERATOR_ARG) \
		--gremlin-report=json \
		$(GREMLIN_CACHE_ARG) \
		tests/

mutate-changed:
	@if [ -z "$(CHANGED_MUTATION_TARGETS)" ]; then \
		echo "No changed mutation targets relative to $(MUTATION_BASE)."; \
	else \
		echo "Mutating changed targets: $(CHANGED_MUTATION_TARGETS)"; \
		$(UV_RUN) pytest --gremlins \
			--gremlin-executor=subprocess \
			--gremlin-workers=$(GREMLIN_WORKERS) \
			--gremlin-targets=$(CHANGED_MUTATION_TARGETS) \
			--gremlin-report=json \
			--gremlin-cache \
			$(INCREMENTAL_REPOSITORY_TESTS) && \
		$(UV_RUN) python scripts/mutation_threshold.py coverage/gremlins/gremlins.json --min 90; \
	fi

mutate-stats: mutate
	$(UV_RUN) python scripts/mutation_threshold.py coverage/gremlins/gremlins.json --min 90

check: lockcheck lint typecheck portable test

verify: check

clean:
	rm -rf coverage/ dist/ build/ htmlcov/ .coverage .coverage.* coverage.xml \
		.pytest_cache/ .ruff_cache/ .hypothesis/ .gremlins_cache/ .uv-cache/ .calcifier-runtime/
