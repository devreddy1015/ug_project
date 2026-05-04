# SHIELD Production Service Architecture (Surpass Plan)

This architecture upgrades Guard from local CLI/UI flows to an authenticated, queue-backed moderation platform with durable job state and callback delivery.

## End-to-End Flow

```mermaid
flowchart TD
    A[Client] --> B[FastAPI Gateway service/api.py]
    B --> C[Auth + validation + idempotency]
    C --> D[(SQLite Job Store)]
    C --> E[(Object Storage service_data/inputs)]
    C --> F[Celery broker dispatch]
    F --> G[Redis Queue]
    G --> H[Celery Workers service/tasks.py]

    H --> I{Modality}
    I -->|video| J[Guard V2 Pipeline service/analyzers.py]
    I -->|text| K[Semantic Scoring + Risk Engine]
    I -->|image/audio| L[Semantic Fallback + Risk Engine]

    J --> D
    K --> D
    L --> D

    D --> M[Status API /v1/jobs/{id}]
    D --> N[Results API /v1/results/{id}]
    D --> O[Metrics API /v1/metrics]

    H --> P[Webhook delivery]
    P --> Q[External callback consumer]
```

## Key Improvements Over PDF Baseline

- Adds durable lifecycle state with attempts, timestamps, and webhook status.
- Supports idempotent submission and replay-safe processing.
- Separates API gateway and worker execution through queues.
- Adds readiness checks for database, broker, and storage.
- Adds callback push plus polling pull for result delivery.
- Uses real Guard V2 analysis for video moderation.
