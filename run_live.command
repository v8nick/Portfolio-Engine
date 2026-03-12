#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -x "venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Upgrading pip..."
    ./venv/bin/python -m pip install --upgrade pip
fi

echo "Syncing requirements..."
./venv/bin/python -m pip install -r requirements.txt

echo "Running live engine..."
./venv/bin/python main_live.py

echo
echo "Live engine finished."
read -r -n 1 -s -p "Press any key to close..."
echo
