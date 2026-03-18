#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
BUNDLE_NAME="sap-sfim-ai-agent-prod"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_PATH="$DIST_DIR/${BUNDLE_NAME}-${TIMESTAMP}.tar.gz"

cd "$ROOT_DIR"

echo "Validating production compose file..."
docker compose -f docker-compose.prod.yml config >/dev/null

echo "Building production image..."
docker build -t aiagentcomm-app-slim:latest .

echo "Running test suite in container..."
docker run --rm aiagentcomm-app-slim:latest python -m pytest -q

echo "Creating release bundle..."
mkdir -p "$DIST_DIR"
tar \
  --exclude='app/__pycache__' \
  --exclude='app/*/__pycache__' \
  --exclude='tests/__pycache__' \
  --exclude='*.pyc' \
  --exclude='chromadb_store' \
  --exclude='data/raw' \
  --exclude='data/processed' \
  -czf "$BUNDLE_PATH" \
  app \
  scripts \
  tests \
  Dockerfile \
  README.md \
  requirements.txt \
  gunicorn.conf.py \
  docker-compose.yml \
  docker-compose.prod.yml \
  .env.example \
  .dockerignore

echo "Bundle created: $BUNDLE_PATH"
