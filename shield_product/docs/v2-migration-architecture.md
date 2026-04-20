# Guard V2 Migration Architecture

This page shows the target V2 runtime and how legacy components map into the new layered pipeline.

## Target V2 Runtime

```mermaid
flowchart TD
    A[run_guard.sh analyze-v2] --> B[training/guard_v2_analyze.py]

    B --> C[discover_video_assets]
    C --> D[GuardV2Config]
    D --> E[GuardV2Pipeline.analyze_assets]

    E --> F[SignalExtractor Protocol]
    E --> G[SemanticScorer Protocol]
    E --> H[EngagementProvider Protocol]
    E --> I[CulturalAdapter Protocol]

    F --> F1[LegacySignalExtractor default]
    G --> G1[LegacySemanticScorer default]
    H --> H1[LegacyEngagementProvider default]
    I --> I1[LegacyCulturalAdapter default]

    F1 --> J[SignalPayload]
    G1 --> K[SemanticResult]
    H1 --> L[Engagement metadata]
    I1 --> M[Region-adapted category scores]

    J --> N[risk_engine formulas]
    K --> N
    L --> N
    M --> N

    N --> N1[final risk]
    N --> N2[network diffusion risk]
    N --> N3[analysis confidence and evidence mode]

    N1 --> O[policy decision\nsafe_to_watch age rating verdict]
    N2 --> O
    N3 --> O

    O --> P[write_outputs reporting]
    P --> P1[guard_analysis.json]
    P --> P2[guard_summary.json]
    P --> P3[guard_quality_summary.json]
    P --> P4[guard_results.csv]
    P --> P5[guard_run_metadata.json]
```

## Migration Map (Legacy to V2)

```mermaid
flowchart LR
    A[Legacy entry\ntraining/guard_analyze.py] --> B[V2 entry\ntraining/guard_v2_analyze.py]
    C[Legacy in-function orchestration] --> D[V2 pipeline module\ntraining/v2/pipeline.py]
    E[Legacy direct module coupling] --> F[Protocol + adapter boundaries\ntraining/v2/pipeline.py]
    G[Legacy scoring/risk spread across modules] --> H[Centralized formulas\ntraining/v2/risk_engine.py]
    I[Legacy output writing in analyzer] --> J[Dedicated reporting layer\ntraining/v2/reporting.py]
    K[Legacy mixed data objects] --> L[Typed contracts\ntraining/v2/types.py]
    M[Legacy video collection helper] --> N[Dedicated dataset discovery\ntraining/v2/dataset.py]
```

## Why this migration architecture matters

- Clear layered boundaries reduce regression risk.
- Dependency injection enables safer module replacement.
- Risk formulas are centralized and auditable.
- Reporting is isolated from scoring logic.
- Failed items and insufficient evidence are explicitly handled.
