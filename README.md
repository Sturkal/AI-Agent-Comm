# SAP SFIM AI Agent

AI agent for SAP SuccessFactors Incentive Management (SFIM) that ingests official PDFs and XML rule exports, stores them in a local vector database, and answers questions through a Flask webhook.

## What is already in place

- XML parsing for `Rule.xml`, `Formula.xml`, `Plan.xml`, and related exports
- Nested SAP XML handling for `FUNCTION`, `OPERATOR`, `RULE_ELEMENT_REF`, and related structures
- PDF chunking and indexing
- Local vector store with ChromaDB-compatible storage
- Flask webhook at `/webhook/whatsapp`
- Ollama local LLM support through the Docker `ollama` service (`http://ollama:11434`)
- Self-evaluation notes stored as a lightweight JSONL knowledge memory
- Dockerized runtime with `docker-compose`
- Default hashing embeddings for deterministic local Docker runs, with optional `sentence-transformers`
- Configurable text or JSON logging for Docker and local runs

## Project Layout

- `app/data_pipeline/xml_parser.py` parses SAP XML into flat JSON-like records
- `app/data_pipeline/ingest.py` ingests raw files from `data/raw`
- `app/data_pipeline/indexer.py` stores and searches knowledge entries
- `app/agent/llm_engine.py` routes questions, self-evaluates answers, and calls Ollama
- `app/agent/refinement_memory.py` stores self-evaluation memory for later refinement rounds
- `app/api/routes.py` exposes the WhatsApp webhook
- `app/cli.py` provides manual CLI ingestion, knowledge-base queries, and memory admin commands

## Quick Start with Docker

1. Copy the environment file:

```bash
cp .env.example .env
```

2. Build the image:

```bash
docker build -t aiagentcomm-app-slim .
```

3. Ingest the raw files into the vector store:

```bash
docker compose --profile ingest run --rm ingest
```

4. Start the web app:

```bash
docker compose up -d
```

5. Check health:

```bash
curl http://localhost:5000/health
curl http://localhost:5000/ready
```

6. Test the webhook:

```bash
curl -X POST http://localhost:5000/webhook/whatsapp \
  -H "Content-Type: application/json" \
  -d '{"user_phone_number":"+10000000000","message_body":"How is F_Direct_Credit_Before_Termination_Check calculated?"}'
```

The webhook response also includes:

- `self_evaluation_summary`: a short explanation of why the draft was trusted or refined
- `self_evaluation`: the structured confidence / refinement metadata
- `refinement_memory`: prior self-evaluation notes that influenced the current answer

Expected n8n payload shape:

```json
{
  "user_phone_number": "+10000000000",
  "message_body": "How is F_Direct_Credit_Before_Termination_Check calculated?"
}
```

## Manual Ingestion

Run ingestion without starting Flask:

```bash
python -m app.cli ingest
```

Custom paths are supported:

```bash
python -m app.cli ingest --raw-dir data/raw --processed-dir data/processed --chroma-dir chromadb_store
```

You can also query the knowledge base directly:

```bash
python -m app.cli query "How is F_Direct_Credit_Before_Termination_Check calculated?" --show-hits
```

You can inspect or export self-evaluation memory:

```bash
python -m app.cli memory show --limit 10
python -m app.cli memory export --output exports/self_eval_memory.json
```

## Docker Notes

- Docker Compose now includes a dedicated `ollama` service. The app and ingest services call `http://ollama:11434` internally.
- Run `docker compose --profile setup run --rm ollama-pull` once to download both the chat model (`OLLAMA_MODEL`) and embedding model (`OLLAMA_EMBED_MODEL`).
- Ollama model data persists in the `ollama_data` volume.
- Ingestion is now fully separate from Flask startup and runs through the dedicated `ingest` service or `python -m app.cli ingest`.
- Self-evaluation memory defaults to `data/processed/self_eval_memory.jsonl` and is reused on later answers when the query overlaps.
- Embeddings default to the lightweight hashing backend. If you later want higher semantic recall, set `EMBEDDING_BACKEND=sentence-transformers` and install the extra dependency.
- Set `LOG_FORMAT=json` if you want structured logs from the app container.
- The app service now has a Docker healthcheck and `restart: unless-stopped` for a more production-like setup.

## Production Deploy

Use the production compose file when you deploy to a server:

```bash
cp .env.example .env
docker compose -f docker-compose.prod.yml up -d --build
```

For production, set `OLLAMA_BASE_URL` to the actual Ollama endpoint you want the app to use. The production compose file defaults to structured JSON logs, a `/ready` healthcheck, and named volumes for persistent data.

You can also use the deployment scripts:

```bash
bash scripts/prod_package.sh
bash scripts/prod_deploy.sh
```

`prod_package.sh` validates the production compose file, builds the image, runs tests, and creates a release bundle in `dist/`. `prod_deploy.sh` builds the services, runs ingest first, and only then starts the API.

The release bundle is saved as `dist/sap-sfim-ai-agent-prod-<timestamp>.tar.gz`.

## Useful Commands

```bash
docker compose up -d ollama
docker compose --profile setup run --rm ollama-pull
docker compose --profile ingest run --rm ingest
docker compose up -d
docker compose -f docker-compose.prod.yml up -d --build
bash scripts/prod_package.sh
bash scripts/prod_deploy.sh
docker compose logs -f app
python -m app.cli query "How is Rule A calculated?" --show-hits
python -m app.cli memory show --limit 10
python -m app.cli memory export --output exports/self_eval_memory.json
```

## Status

- Core ingest, retrieval, webhook, memory, and Docker flows are implemented.
- Remaining work is mostly product hardening and quality tuning, such as richer rule parsing for more edge-case SAP exports and optional higher-quality embeddings.
