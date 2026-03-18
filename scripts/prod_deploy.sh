#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "Validating production compose file..."
docker compose -f docker-compose.prod.yml config >/dev/null

echo "Building production services..."
docker compose -f docker-compose.prod.yml build app ingest

echo "Running ingest before bringing the API online..."
docker compose -f docker-compose.prod.yml --profile ingest run --rm ingest

echo "Starting production API..."
docker compose -f docker-compose.prod.yml up -d app

echo "Production stack is up."
