# SHIELD Prototype API (Production-Hardened)

Asynchronous multimodal scoring service using:

- FastAPI for request handling.
- Celery + Redis for background execution.
- SQLite (WAL mode) for job state persistence.

The service now includes health/readiness checks, idempotency controls, stronger upload validation, richer job lifecycle tracking, and operational metrics endpoints.

## Architecture

1. Client submits a job (text or file).
2. API validates payload and stores a durable queued job record.
3. Celery worker processes the job asynchronously.
4. Job state transitions: `queued -> processing -> completed|failed|canceled`.
5. Client polls status or fetches final result.

## Quick Start

1. Create and activate environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure environment.

```bash
cp .env.example .env
# edit values as needed
```

4. Start Redis.

```bash
redis-server
```

5. Start API.

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

6. Start worker.

```bash
celery -A app.celery_app worker -Q text,image,video,audio -l info
```

## Production Run Commands

API (multi-worker):

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 2
```

Worker with controlled concurrency:

```bash
celery -A app.celery_app worker -Q text,image,video,audio --concurrency 4 -l info
```

## Security and Controls

- Optional API key enforcement via `REQUIRE_API_KEY=true` and `API_KEY`.
- CORS allowlist via `ALLOWED_ORIGINS`.
- Upload size enforcement via `MAX_UPLOAD_SIZE_BYTES`.
- Modality-specific allowed file extensions.
- Request ID propagation and process-time headers for tracing.
- Idempotency support using:
   - `idempotency_key` field for text jobs.
   - `X-Idempotency-Key` header for file jobs.

## API Endpoints

Health:

- `GET /health/live`
- `GET /health/ready`

Jobs:

- `POST /v1/jobs/text`
- `POST /v1/jobs/file?modality=image|video|audio`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs?status=&modality=&limit=&offset=`
- `POST /v1/jobs/{job_id}/cancel`
- `GET /v1/results/{job_id}`
- `GET /v1/metrics`

## Examples

Create text job:

```bash
curl -X POST http://localhost:8000/v1/jobs/text \
   -H "Content-Type: application/json" \
   -d '{"text":"I will help people with support resources", "idempotency_key":"txt-001"}'
```

Create file job:

```bash
curl -X POST "http://localhost:8000/v1/jobs/file?modality=image" \
   -H "X-Idempotency-Key: img-001" \
   -F "file=@/path/to/image.jpg"
```

Get job status:

```bash
curl http://localhost:8000/v1/jobs/<job_id>
```

Get result:

```bash
curl http://localhost:8000/v1/results/<job_id>
```

List jobs:

```bash
curl "http://localhost:8000/v1/jobs?status=queued&limit=20&offset=0"
```

## Environment Variables

Copy defaults from `.env.example`. Main settings:

- `APP_ENV`: `development|production`
- `REDIS_URL`: Celery broker and backend URL
- `DATA_DIR`: local storage root
- `DATABASE_PATH`: SQLite DB path
- `REQUIRE_API_KEY`: enable API-key auth
- `API_KEY`: expected API key when auth is enabled
- `ALLOWED_ORIGINS`: comma-separated CORS origin list (`*` for all)
- `MAX_UPLOAD_SIZE_BYTES`: upload hard limit
- `TASK_SOFT_TIME_LIMIT_SEC`: Celery soft timeout
- `TASK_TIME_LIMIT_SEC`: Celery hard timeout
- `RESULT_TTL_SECONDS`: result expiration in broker backend

## Operational Notes

- SQLite is configured with WAL and busy timeout for better concurrent access.
- Job records include queue name, attempts, started/completed timestamps, and task ID.
- Worker scoring is deterministic heuristic logic (`analysis_mode=heuristic_v2`) intended as a production-safe baseline, not a final ML moderation model.
