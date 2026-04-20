# Guard Legacy Runtime Architecture

This diagram reflects the currently active production path that generates fields such as average_good_for_society and average_sbi in guard_summary.json.

## Diagram

```mermaid
flowchart TD
    A[run_guard.sh mode: ui or analyze] --> B{Entry path}

    B -->|ui| C[training/guard_app.py]
    B -->|analyze| D[training/guard_analyze.py]

    C --> E[extract_reel_signals\ntranscript OCR caption frame_signals]
    D --> E

    E --> F[DualScoringEngine.score\nvector + optional LLM]
    E --> G[temporal_segment_analysis]
    G --> H[blend_temporal_scores]
    F --> H

    H --> I[phase1 prior injection optional]
    I --> J[apply_cultural_adapter by region]

    J --> K[Risk features]
    K --> K1[compute_viral_harm_potential]
    K --> K2[compute_cross_modal_contradiction]
    K --> K3[adversarial_evasion_score]
    K --> K4[cognitive_manipulation_index]
    K --> K5[compute_network_diffusion_risk]

    K1 --> L[build_guard_report]
    K2 --> L
    K3 --> L
    K4 --> L
    K5 --> L

    L --> M[policy outputs\nsafe_to_watch age rating verdict]
    M --> N[creator profile update optional federated export]

    N --> O[guard_analysis.json]
    N --> P[guard_summary.json]
    N --> Q[guard_quality_summary.json]
    N --> R[guard_results.csv]
    N --> S[guard_run_metadata.json]
```

## Main Modules

- Entry and mode routing: training/guard_analyze.py and training/guard_app.py
- Multimodal extraction: training/guard_multimodal.py
- Semantic scoring engine: training/guard_scoring.py
- Temporal risk blending: training/guard_temporal.py
- Platform risk and adaptation: training/guard_platform.py
- Final report shaping: training/guard_report.py
- Summary and artifact writing: training/guard_analyze.py

## Why this is the active path

The summary file currently open in your editor contains keys from this path:
- average_good_for_society
- average_sbi

Those fields are built by _build_summary in training/guard_analyze.py, not by V2 reporting.
