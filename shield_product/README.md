# Guard (SHIELD) Product

Guard is a multimodal harmful-content analysis system for short-form videos.
It combines feature extraction, vector scoring, optional LLM scoring, temporal risk modeling,
and policy-style output signals.

## What Is Included

- Upload frontend for full Guard video analysis.
- Guard CLI for file or folder analysis.
- Training pipeline for binary and multi-label models.
- Legacy SHIELD engine and dashboard entrypoints.

## Prerequisites

- Python 3.10+
- ffmpeg and ffprobe recommended for media metadata and frame extraction

Optional:

- Whisper for transcription
- Tesseract plus pytesseract for OCR

## Quick Start (Recommended)

Run from the shield_product directory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Guard Studio (redesigned upload frontend):

```bash
python -m streamlit run training/guard_app.py
```

Or use the one-command launcher:

```bash
./run_guard.sh ui
```

Run environment diagnostics (recommended before production deployment):

```bash
./run_guard.sh doctor
```

Enforce recommended-readiness gate (non-zero exit when optional production features are missing):

```bash
./run_guard.sh doctor --strict-recommended
```

## Production Guard Service (Async API)

This repository now includes a production-style asynchronous moderation service that upgrades Guard beyond local CLI/UI usage.

### What Was Built

- API gateway with FastAPI for async job submission and retrieval.
- Queue-backed processing with Celery and Redis.
- Durable job lifecycle in SQLite with retries, timestamps, and webhook state.
- Separate modality workers and queue routing for text, image, video, and audio.
- Real Guard V2 pipeline usage for video moderation.
- Semantic scoring + risk engine fallback flow for text/image/audio.
- Idempotent job creation for safe client retries.
- Health and readiness probes for orchestration systems.
- Structured metrics endpoint for service telemetry.
- Request tracing headers and structured logging.
- Launcher integration (`run_guard.sh serve` and `run_guard.sh worker`).

### Build Inventory (Files Added/Used)

- `service/api.py`: REST API, auth checks, validation, idempotency logic, health, metrics.
- `service/tasks.py`: Celery task handlers and job status transitions.
- `service/analyzers.py`: modality analysis orchestration and risk composition.
- `service/jobs.py`: job CRUD, lifecycle transitions, metrics aggregation.
- `service/db.py`: SQLite schema, indexes, migrations for missing columns.
- `service/storage.py`: upload validation, extension checks, max-size enforcement, storage writes.
- `service/notifications.py`: webhook payload creation and delivery retries.
- `service/celery_app.py`: Celery app config, queues, routes, task reliability options.
- `service/config.py`: centralized typed settings loaded from environment variables.
- `service/logging_utils.py`: logging initialization for API and worker.
- `.env.service.example`: production service environment template.
- `docs/production-service-architecture.md`: architecture flow and key improvements.

### Runtime Architecture (End-to-End)

1. Client sends job request to API.
2. API validates auth, payload, modality, size, and extensions.
3. API checks idempotency key (if provided).
4. API writes job row in SQLite with `queued` status.
5. API stores uploaded file under service storage path (file jobs).
6. API dispatches Celery task to modality queue.
7. Worker fetches job and marks it `processing`.
8. Worker runs modality-specific analyzer and computes result.
9. Worker saves result, marks `completed` or `failed`, and attempts webhook delivery.

### Persistence and Paths

- Job database: `service_data/jobs.db` by default.
- Uploaded files: `service_data/inputs/` by default.
- Redis broker/backend: `redis://localhost:6379/15` by default.
- Redis DB `15` is used by default to avoid collisions with other local apps that use DB `0`.

### Job Lifecycle and States

Job status values:

- `queued`
- `processing`
- `completed`
- `failed`
- `canceled`

Webhook status values:

- `not_configured`
- `pending`
- `delivered`
- `failed`

Lifecycle details:

- `attempts` increments each time work starts.
- `started_at` set when processing begins.
- `completed_at` set for terminal states.
- Terminal statuses: `completed`, `failed`, `canceled`.

### Modality Processing Behavior

- `video`: uses Guard V2 (`GuardV2Pipeline`) with configured frame count, OCR, and whisper toggles.
- `text`: uses semantic scorer and risk engine composition.
- `image` and `audio`: use filename and optional sidecar `.txt` context plus semantic/risk flow.

Reliability behavior in semantic mode:

- Circuit breaker opens after repeated semantic failures.
- Keyword fallback scoring is used while circuit is open.
- Insufficient evidence mode increases conservative risk weighting.

### API Reference

Public health endpoints:

- `GET /health/live`
- `GET /health/ready`

Versioned service endpoints (`API_PREFIX` defaults to `/v1`):

- `POST /v1/jobs/text`: create text moderation job.
- `POST /v1/jobs/file?modality=image|video|audio`: create file moderation job.
- `GET /v1/jobs`: list jobs with filters and pagination.
- `GET /v1/jobs/{job_id}`: fetch job status and metadata.
- `POST /v1/jobs/{job_id}/cancel`: cancel non-terminal jobs.
- `GET /v1/results/{job_id}`: fetch result payload for completed jobs.
- `GET /v1/metrics`: aggregated service counters.

OpenAPI docs (FastAPI default):

- `GET /docs`
- `GET /openapi.json`

### Authentication

- Optional API key auth is controlled by `REQUIRE_API_KEY`.
- If enabled, clients must send header `X-API-Key` (or custom `API_KEY_HEADER`).

### Idempotency Behavior

- Text jobs: send `idempotency_key` in JSON body.
- File jobs: send `X-Idempotency-Key` header.
- Duplicate key returns existing job ID with `created_new: false`.

### Webhook Behavior

- Text jobs: pass `webhook_url` in JSON body.
- File jobs: pass `X-Webhook-Url` header or `webhook_url` query param.
- Webhook payload includes job metadata, status, result, and error message.
- Delivery attempts are tracked in job row fields.

### Service Result Shape

`GET /v1/results/{job_id}` returns:

- `job_id`
- `modality`
- `result` object

`result` commonly includes:

- `processing_status`
- `overall_risk_score_out_of_100`
- `overall_safety_score_out_of_100`
- `good_for_society_percentage`
- `safe_to_watch`
- `content_age_rating`
- `analysis_confidence`
- `insufficient_evidence`
- `evidence_mode`
- `category_breakdown`
- `top_risk_categories`
- `verdict`
- `analysis_mode`

### Environment Variables (Service)

Core:

- `APP_NAME`
- `APP_ENV`
- `API_PREFIX`
- `LOG_LEVEL`

Storage/database/broker:

- `DATA_DIR`
- `DATABASE_PATH`
- `REDIS_URL`

Auth/request:

- `REQUIRE_API_KEY`
- `API_KEY`
- `API_KEY_HEADER`
- `REQUEST_ID_HEADER`
- `ALLOWED_ORIGINS`

Limits/pagination/timeouts:

- `MAX_UPLOAD_SIZE_BYTES`
- `DEFAULT_PAGE_SIZE`
- `MAX_PAGE_SIZE`
- `TASK_SOFT_TIME_LIMIT_SEC`
- `TASK_TIME_LIMIT_SEC`
- `RESULT_TTL_SECONDS`
- `TEXT_MAX_CHARS`

Webhook:

- `WEBHOOK_TIMEOUT_SECONDS`
- `WEBHOOK_MAX_ATTEMPTS`
- `WEBHOOK_USER_AGENT`

Guard analysis controls:

- `GUARD_FRAME_COUNT`
- `GUARD_ENABLE_WHISPER`
- `GUARD_ENABLE_OCR`
- `GUARD_REGION`

File extension allowlists:

- `ALLOWED_IMAGE_EXTENSIONS`
- `ALLOWED_VIDEO_EXTENSIONS`
- `ALLOWED_AUDIO_EXTENSIONS`

Environment loading note:

- Service components auto-load `.env`, `.env.service`, and `.env.local` from the project root.
- Existing process environment variables always take precedence over file values.
- You can point to a custom file with `ENV_FILE=/path/to/file.env`.

### Queue and Worker Configuration

- Queues: `text`, `image`, `video`, `audio`, `default`.
- Routes:
  - `service.tasks.score_text` -> `text`
  - `service.tasks.score_image` -> `image`
  - `service.tasks.score_video` -> `video`
  - `service.tasks.score_audio` -> `audio`
- Reliability settings include `acks_late`, `reject_on_worker_lost`, `prefetch_multiplier=1`, soft and hard task limits, and Redis result expiry.

### Full Runbook (From Zero)

1. Create and activate venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create service env file:

```bash
cp .env.service.example .env
```

4. Optional: override config via shell variables (these override `.env` values):

```bash
set -a
source .env
set +a
```

5. Start Redis (if not already running):

```bash
redis-server --save "" --appendonly no
```

6. Start worker:

```bash
./run_guard.sh worker
```

7. Start API:

```bash
./run_guard.sh serve
```

8. Verify health:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

### Practical API Examples

Create text job:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs/text \
  -H "Content-Type: application/json" \
  -d '{"text":"sample moderation payload", "idempotency_key":"demo-text-1"}'
```

Create file job:

```bash
curl -X POST "http://127.0.0.1:8000/v1/jobs/file?modality=video" \
  -H "X-Idempotency-Key: demo-file-1" \
  -H "X-Webhook-Url: https://example.com/callback" \
  -F "file=@/path/to/video.mp4"
```

List jobs:

```bash
curl "http://127.0.0.1:8000/v1/jobs?status=completed&limit=20&offset=0"
```

Get status:

```bash
curl http://127.0.0.1:8000/v1/jobs/<job_id>
```

Get result:

```bash
curl http://127.0.0.1:8000/v1/results/<job_id>
```

Cancel job:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs/<job_id>/cancel
```

Get metrics:

```bash
curl http://127.0.0.1:8000/v1/metrics
```

### Troubleshooting

- `zsh: no such file or directory: ./run_guard.sh`
  - Run from `shield_product` directory or use absolute path.

- Redis bind error: `Address already in use` on port 6379.
  - Redis is already running. Reuse it, or stop existing Redis before starting another.

- Worker receives unregistered tasks like `app.tasks.score_text`.
  - This usually means stale tasks from another app in shared Redis DB.
  - Use isolated `REDIS_URL` (default is DB 15) and clear stale queues if needed.

- `GET /v1/results/{job_id}` returns 409.
  - Job is not completed yet. Poll `/v1/jobs/{job_id}` first.

- API key errors (401).
  - Check `REQUIRE_API_KEY`, `API_KEY`, and request header name.

### Process Control

Start API:

```bash
./run_guard.sh serve
```

Start worker:

```bash
./run_guard.sh worker
```

Stop processes:

- Press `Ctrl+C` in each terminal running API/worker/redis.

### Architecture Reference

- Detailed architecture note: `docs/production-service-architecture.md`

## Guard Upload Frontend

Command:

```bash
python -m streamlit run training/guard_app.py
```

Or via launcher:

```bash
./run_guard.sh ui
```

Run on a custom port:

```bash
./run_guard.sh ui --server.port 8519
```

Capabilities:

- Guard Studio visual theme with a hero summary and card-based metrics.
- Upload one video and preview file details (name, format, size).
- Control Deck sidebar with grouped controls:
  - Signal Extraction
  - Temporal Dynamics
  - Advanced Paths
- Decision Dashboard tabs:
  - Overview
  - Category Map
  - Temporal View
  - Evidence
  - Raw JSON
- One-click JSON report download from the Raw JSON tab.
- Optional artifact persistence to guard_outputs (disabled by default; toggle to enable).

Recommended UI workflow:

1. Upload a video.
2. Tune extraction and temporal settings in Control Deck.
3. Click `Run Full Guard Analysis`.
4. Review Decision Dashboard tabs and download JSON if needed.

## Guard CLI

Basic analysis:

```bash
python -m training.guard_analyze \
  --input /path/to/video_or_folder \
  --output-dir guard_outputs
```

Equivalent launcher command:

```bash
./run_guard.sh analyze --input /path/to/video_or_folder --output-dir guard_outputs
```

With phase-1 model:

```bash
python -m training.guard_analyze \
  --input /path/to/video_or_folder \
  --output-dir guard_outputs \
  --phase1-model training_runs_all_videos/model.pt \
  --phase1-run-summary training_runs_all_videos/run_summary.json
```

Fast batch mode:

```bash
python -m training.guard_analyze \
  --input /path/to/video_or_folder \
  --output-dir guard_outputs \
  --disable-whisper \
  --disable-ocr
```

Strict mode (fail fast on first item error):

```bash
python -m training.guard_analyze \
  --input /path/to/video_or_folder \
  --output-dir guard_outputs \
  --fail-on-item-error
```

Allow empty inputs (emit empty artifacts instead of failing):

```bash
python -m training.guard_analyze \
  --input /path/to/video_or_folder \
  --output-dir guard_outputs \
  --allow-empty-input
```

Export phase-2 labels:

```bash
python -m training.guard_analyze \
  --input /path/to/video_or_folder \
  --output-dir guard_outputs \
  --phase2-labels-output guard_outputs/phase2_labels.json
```

## Guard V2 (Rebuilt Architecture)

Guard V2 is a clean-slate pipeline with layered modules, dependency injection, and tests.
It is designed to be easier to maintain and safer to evolve than the legacy analyzer flow.

Run V2 directly:

```bash
python -m training.guard_v2_analyze \
  --input /path/to/video_or_folder \
  --output-dir guard_outputs
```

Or via launcher:

```bash
./run_guard.sh analyze-v2 --input /path/to/video_or_folder --output-dir guard_outputs
```

Key V2 modules:

- `training/v2/dataset.py`: media discovery and indexing
- `training/v2/risk_engine.py`: scoring formulas and risk composition
- `training/v2/adapters.py`: integration adapters for existing extraction/scoring modules
- `training/v2/pipeline.py`: orchestration and report object generation
- `training/v2/reporting.py`: summary/csv/json output writing

V2 tests:

```bash
pip install -r requirements-dev.txt
pytest -q tests/v2
```

## Outputs

Main Guard outputs:

- guard_outputs/guard_analysis.json
- guard_outputs/guard_summary.json
- guard_outputs/guard_quality_summary.json
- guard_outputs/guard_run_metadata.json
- guard_outputs/guard_results.csv
- guard_outputs/phase2_labels.json (if enabled)
- guard_outputs/creator_profiles.json (if profiling is used)

guard_analysis.json includes:

- results (successfully processed videos)
- failed_results (videos that failed with error_message)

Each report now includes:

- analysis_confidence (0-100)
- insufficient_evidence (boolean)
- processing_status
- error_message (for failed items)

## Training Workflow

Install training extras:

```bash
pip install -r requirements-train.txt
```

Use training extras when you need model training or phase-1 model inference in Guard CLI/frontend.

Binary training:

```bash
python -m training.train \
  --dataset /path/to/data \
  --modality video \
  --output-dir training_runs
```

Feature preprocessing:

```bash
python -m training.preprocess \
  --dataset /path/to/data \
  --output-json preprocess_videos.json \
  --output-csv preprocess_videos.csv \
  --no-frames
```

Multi-label training:

```bash
python -m training.train \
  --dataset /path/to/data \
  --modality video \
  --multi-label \
  --labels-file /path/to/labels.json \
  --output-dir training_runs
```

Train phase-2 from Guard export:

```bash
python -m training.train \
  --dataset /path/to/reels_or_folder \
  --modality video \
  --multi-label \
  --labels-file guard_outputs/phase2_labels.json \
  --output-dir training_runs_phase2
```

## Legacy SHIELD Entry Points

CLI runner:

```bash
python -m Engine.Launcher.LauncherMain --dataset /path/to/data --output-dir run_outputs
```

Legacy dashboard:

```bash
python -m streamlit run Engine/Launcher/StreamLitLauncher.py
```

Legacy analyzer dashboard:

```bash
python -m streamlit run training/video_app.py
```

## Optional Environment Variables

For LLM scoring:

- LLM_API_BASE
- LLM_API_KEY
- LLM_MODEL
- GROQ_API_KEY
- GROQ_MODEL
- GROQ_API_BASE

## Notes

- Guard is the user-facing name.
- Internal modules are now guard_ native, with reelguard_ compatibility shims retained for legacy imports.
- Edit category bags in Engine/VectorHandler/CategoryBags/category_terms.json.
