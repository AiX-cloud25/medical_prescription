#!/bin/bash
# One-shot startup for a fresh Jarvis Labs container: installs/starts Ollama
# with the validated tuning env vars, waits until it's actually ready, then
# hands off to run.sh for the backend. Only command needed: bash start.sh
#
# Override any tuning var without editing this file, e.g.:
#   OLLAMA_NUM_PARALLEL=4 bash start.sh

set -e

OLLAMA_HOST_URL=${OLLAMA_HOST:-http://localhost:11434}
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-4}
export OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-q8_0}
export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:--1}

echo "Ollama tuning: NUM_PARALLEL=${OLLAMA_NUM_PARALLEL} KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE} KEEP_ALIVE=${OLLAMA_KEEP_ALIVE}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama not found, installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed, skipping install."
fi

echo "Starting ollama serve in the background (log: ollama.log)..."
nohup ollama serve > ollama.log 2>&1 &

echo "Waiting for Ollama to be ready at ${OLLAMA_HOST_URL}..."
READY=0
for i in $(seq 1 60); do
    if curl -sf "${OLLAMA_HOST_URL}" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 2
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Ollama did not become ready within 120s — check ollama.log" >&2
    exit 1
fi

echo "Ollama is ready. Installed models:"
ollama list

if [ -f ".venv/bin/activate" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
fi

echo "Starting backend via run.sh..."
exec ./run.sh
