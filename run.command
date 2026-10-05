#!/bin/bash
# macOS: double-click this file. Linux: bash run.command.
set -Eeuo pipefail
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

finish() {
    status=$?
    if [ "$status" -ne 0 ] && [ "$status" -ne 130 ]; then
        echo
        echo "Could not start Portfolio Engine. See the error above."
        echo "Check your internet connection and that this folder is writable."
        if [ -t 0 ]; then read -r -p "Press Enter to close..." || true; fi
    fi
}
trap finish EXIT

MODE="${1:-app}"
if [ "$#" -gt 0 ]; then shift; fi
case "$MODE" in
    app|--app|--live|--research|--check) ;;
    *) echo "Usage: bash run.command [--app|--live|--research|--check]"; exit 2 ;;
esac

echo "Portfolio Engine"
echo "First launch downloads Python 3.11 and app dependencies; later launches reuse them."
echo "Setup stays in .runtime inside this folder. No administrator access is needed."

UV_BIN="$PROJECT_DIR/.runtime/uv/uv"
if [ ! -x "$UV_BIN" ]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    else
        echo "Downloading the uv setup tool from astral.sh..."
        mkdir -p "$PROJECT_DIR/.runtime/uv"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$PROJECT_DIR/.runtime/uv" sh
    fi
fi
if [ ! -x "$UV_BIN" ]; then echo "The uv setup tool was not installed successfully."; exit 1; fi

export UV_PYTHON_INSTALL_DIR="$PROJECT_DIR/.runtime/python"
export UV_CACHE_DIR="$PROJECT_DIR/.runtime/cache"
export UV_PYTHON_DOWNLOADS=automatic
UV_ARGS=(run --isolated --no-project --managed-python --python 3.11 --with-requirements "$PROJECT_DIR/requirements.txt")
case "$MODE" in
    app|--app)
        echo "The dashboard will open in your browser at http://localhost:8501."
        echo "Keep this window open. Press Control+C here to stop the app."
        "$UV_BIN" "${UV_ARGS[@]}" python -m streamlit run ui/app.py --server.address 127.0.0.1 --browser.gatherUsageStats false "$@"
        ;;
    --live) "$UV_BIN" "${UV_ARGS[@]}" python main_live.py "$@" ;;
    --research) "$UV_BIN" "${UV_ARGS[@]}" python main_research.py "$@" ;;
    --check)
        "$UV_BIN" "${UV_ARGS[@]}" python -c "import sys, streamlit, plotly, portfolio_decision, main_live, main_research; from sklearn.covariance import LedoitWolf; print('Setup verified. Python ' + sys.version.split()[0] + '; app, engine and covariance dependencies OK.')"
        ;;
esac
