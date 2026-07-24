#!/bin/bash
# Start the Offline Prescription Extractor on Jarvis Labs (Linux).
# Binds to 0.0.0.0 so the Jarvis proxy can reach it.
# --proxy-headers: trust X-Forwarded-* headers from the Jarvis reverse proxy.
# --forwarded-allow-ips="*": accept forwarded headers from any upstream IP.

PORT=${PORT:-8002}

# Kill anything already on the port
fuser -k ${PORT}/tcp 2>/dev/null || true

uvicorn backend:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --proxy-headers \
    --forwarded-allow-ips="*" \
    --timeout-keep-alive 300 \
    --h11-max-incomplete-event-size 10485760
