#!/bin/sh
# Convenience wrapper; Windows users invoke acceptance.py with the repo Python.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec uv run --frozen --project "$script_dir/.." python "$script_dir/acceptance.py" "$@"
