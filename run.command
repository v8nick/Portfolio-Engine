#!/bin/bash
set -e

cd "$(dirname "$0")"

PYTHON_BIN=python3

if ! command -v $PYTHON_BIN >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3 first."
  exit 1
fi

if [ ! -d ".venv" ]; then
  $PYTHON_BIN -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py