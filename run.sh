#!/bin/bash
# Start the Offline Prescription Extractor on Jarvis Labs (Linux).
# Binds to 0.0.0.0 so the Jarvis proxy can reach it.
#
# Optional env vars:
#   PORT=8002          which port to listen on
#   ROOT_PATH=""       set if Jarvis proxies under a sub-path e.g. /proxy/8002

PORT=${PORT:-8002}
ROOT_PATH=${ROOT_PATH:-""}

# Kill anything already on the port
fuser -k ${PORT}/tcp 2>/dev/null || true

echo "Starting on 0.0.0.0:${PORT} root_path='${ROOT_PATH}'"

uvicorn backend:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --root-path "${ROOT_PATH}" \
    --proxy-headers \
    --forwarded-allow-ips="*" \
    --timeout-keep-alive 300 \
    --h11-max-incomplete-event-size 10485760
