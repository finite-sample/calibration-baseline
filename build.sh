#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python make_tables.py "$@"
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
