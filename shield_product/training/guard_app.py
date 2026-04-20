import csv
import importlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

if __package__:
    from . import guard_analyze as _guard_analyze
    from .guard_scoring import DualScoringEngine
else:
    # Support: streamlit run training/guard_app.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import training.guard_analyze as _guard_analyze
    from training.guard_scoring import DualScoringEngine


_build_phase1_analyzer = _guard_analyze._build_phase1_analyzer
_build_quality_summary = _guard_analyze._build_quality_summary
_build_summary = _guard_analyze._build_summary
analyze_single_video_report = getattr(_guard_analyze, "analyze_single_video_report", None)

# Streamlit can cache module objects across reruns; force one reload if the new symbol is missing.
if analyze_single_video_report is None:
    _guard_analyze = importlib.reload(_guard_analyze)
    _build_phase1_analyzer = _guard_analyze._build_phase1_analyzer
    _build_quality_summary = _guard_analyze._build_quality_summary
    _build_summary = _guard_analyze._build_summary
    analyze_single_video_report = getattr(_guard_analyze, "analyze_single_video_report", None)

if analyze_single_video_report is None:
    raise ImportError(
        "training.guard_analyze does not expose analyze_single_video_report. "
        "Restart Streamlit and ensure the latest shield_product code is active."
    )


VIDEO_TYPES = ["mp4", "mov", "mkv", "avi", "webm"]


def _load_ui_modules():
    st = importlib.import_module("streamlit")
    return st


def _analyze_video(
    video_path: Path,
    output_dir: Path,
    frame_count: int,
    disable_whisper: bool,
    disable_ocr: bool,
    phase1_model: Optional[str],
    phase1_run_summary: Optional[str],
    temporal_window_sec: float,
    temporal_recency_bias: float,
    temporal_max_segments: int,
    region_arg: str,
    creator_profile_path: Optional[str],
    federated_export_path: Optional[str],
    source_video_name: Optional[str] = None,
) -> Dict[str, object]:
    scorer = DualScoringEngine()
    phase1 = _build_phase1_analyzer(phase1_model, phase1_run_summary)

    profile_path = (
        Path(creator_profile_path).expanduser().resolve()
        if creator_profile_path
        else output_dir / "creator_profiles.json"
    )
    federated_path = (
        Path(federated_export_path).expanduser().resolve()
        if federated_export_path
        else None
    )
    display_video_path = str(source_video_name).strip() if source_video_name else str(video_path)

    return analyze_single_video_report(
        video_path=video_path,
        output_dir=output_dir,
        scorer=scorer,
        phase1=phase1,
        frame_count=int(frame_count),
        disable_whisper=bool(disable_whisper),
        disable_ocr=bool(disable_ocr),
        temporal_window_sec=float(temporal_window_sec),
        temporal_recency_bias=float(temporal_recency_bias),
        temporal_max_segments=int(temporal_max_segments),
        region_arg=str(region_arg),
        profile_path=profile_path,
        federated_path=federated_path,
        display_video_path=display_video_path,
    )


def _save_outputs(output_dir: Path, report: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary([report])
    quality_summary = _build_quality_summary([report])

    (output_dir / "guard_analysis.json").write_text(
        json.dumps({"results": [report]}, indent=2), encoding="utf-8"
    )
    (output_dir / "guard_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "guard_quality_summary.json").write_text(
        json.dumps(quality_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "guard_run_metadata.json").write_text(
        json.dumps(
            {
                "generated_at": _utc_now(),
                "total_videos": 1,
                "completed_videos": 1,
                "failed_videos": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with (output_dir / "guard_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "processing_status",
            "error_message",
            "video_path",
            "overall_safety_score_out_of_100",
            "overall_risk_score_out_of_100",
            "good_for_society_percentage",
            "societal_benefit_index",
            "viral_harm_potential",
            "network_diffusion_risk",
            "safe_to_watch",
            "content_age_rating",
            "phase1_harm_probability",
            "cross_modal_contradiction_score",
            "adversarial_evasion_score",
            "cognitive_manipulation_index",
            "analysis_confidence",
            "insufficient_evidence",
            "evidence_mode",
            "creator_id",
            "verdict",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "processing_status": report.get("processing_status", "completed"),
                "error_message": report.get("error_message"),
                "video_path": report.get("video_path"),
                "overall_safety_score_out_of_100": report.get("overall_safety_score_out_of_100"),
                "overall_risk_score_out_of_100": report.get("overall_risk_score_out_of_100"),
                "good_for_society_percentage": report.get("good_for_society_percentage"),
                "societal_benefit_index": report.get("societal_benefit_index"),
                "viral_harm_potential": report.get("viral_harm_potential"),
                "network_diffusion_risk": report.get("network_diffusion_risk"),
                "safe_to_watch": report.get("safe_to_watch"),
                "content_age_rating": report.get("content_age_rating"),
                "phase1_harm_probability": report.get("phase1_harm_probability"),
                "cross_modal_contradiction_score": report.get("cross_modal_contradiction_score"),
                "adversarial_evasion_score": report.get("adversarial_evasion_score"),
                "cognitive_manipulation_index": report.get("cognitive_manipulation_index"),
                "analysis_confidence": report.get("analysis_confidence"),
                "insufficient_evidence": report.get("insufficient_evidence"),
                "evidence_mode": report.get("evidence_mode"),
                "creator_id": report.get("creator_id"),
                "verdict": report.get("verdict"),
            }
        )


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return default


def _inject_ui_theme(st) -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
  --g-bg:          #07090e;
  --g-surface:     #0d1117;
  --g-surface-2:   #121c28;
  --g-border:      rgba(255,255,255,0.055);
  --g-border-lit:  rgba(0,235,185,0.2);
  --g-text:        #c5d5e3;
  --g-text-soft:   #4e6478;
  --g-teal:        #00ebc0;
  --g-teal-dim:    rgba(0,235,192,0.08);
  --g-teal-glow:   rgba(0,235,192,0.18);
  --g-amber:       #ffb347;
  --g-red:         #ff4060;
  --g-green:       #00e5a0;
  --g-blue:        #46b3ff;
}

/* ── Page background ─────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 60% 40% at 15% 0%,  rgba(0,235,192,0.055) 0%, transparent 60%),
    radial-gradient(ellipse 50% 35% at 90% 90%, rgba(255,179,71,0.04)  0%, transparent 55%),
    var(--g-bg);
  color: var(--g-text);
}

[data-testid="stHeader"] { background: transparent; }

.main .block-container {
  max-width: 1200px;
  padding-top: 1.4rem;
  padding-bottom: 2.4rem;
}

h1, h2, h3, h4 { color: var(--g-text); }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1019 0%, #0e151f 100%);
  border-right: 1px solid var(--g-border);
  font-family: "DM Sans", sans-serif;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: #8aaabf;
}

[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
  background: var(--g-teal) !important;
}

/* ── File uploader ───────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed rgba(0,235,192,0.25);
  border-radius: 14px;
  background: rgba(0,235,192,0.03);
  transition: transform 260ms ease, box-shadow 260ms ease, border-color 260ms ease, background 260ms ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
  transform: translateY(-2px);
  border-color: rgba(0,235,192,0.5);
  background: rgba(0,235,192,0.06);
  box-shadow: 0 0 28px rgba(0,235,192,0.1);
}

/* ── Metric cards ────────────────────────────────── */
div[data-testid="stMetric"] {
  background: var(--g-surface);
  border: 1px solid var(--g-border);
  border-radius: 12px;
  padding: 0.7rem 0.85rem;
  position: relative;
  overflow: hidden;
  animation: gFadeUp 380ms ease-out both;
  transition: transform 220ms ease, box-shadow 220ms ease;
}

div[data-testid="stMetric"]::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg,
    transparent 0%,
    var(--g-teal) 40%,
    var(--g-teal) 60%,
    transparent 100%);
  opacity: 0;
  animation: gScanLine 3.6s ease-in-out infinite;
  animation-delay: var(--scan-delay, 0s);
}

div[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  border-color: var(--g-border-lit);
  box-shadow: 0 0 22px rgba(0,235,192,0.08);
}

div[data-testid="stMetricLabel"] {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--g-text-soft) !important;
}

div[data-testid="stMetricValue"] {
  font-family: "Syne", sans-serif;
  font-weight: 700;
  color: var(--g-text) !important;
}

/* ── Primary button ──────────────────────────────── */
.stButton > button {
  border: 1px solid rgba(0,235,192,0.3);
  border-radius: 10px;
  background: linear-gradient(130deg, rgba(0,235,192,0.12) 0%, rgba(0,235,192,0.06) 100%);
  color: var(--g-teal);
  font-family: "DM Sans", sans-serif;
  font-weight: 600;
  letter-spacing: 0.04em;
  box-shadow: 0 0 20px rgba(0,235,192,0.08), inset 0 1px 0 rgba(0,235,192,0.1);
  transition: transform 220ms ease, box-shadow 220ms ease, background 220ms ease;
}

.stButton > button:hover {
  transform: translateY(-2px);
  background: linear-gradient(130deg, rgba(0,235,192,0.2) 0%, rgba(0,235,192,0.1) 100%);
  box-shadow: 0 0 32px rgba(0,235,192,0.2), inset 0 1px 0 rgba(0,235,192,0.15);
}

/* ── Download button ─────────────────────────────── */
div.stDownloadButton > button {
  border: 1px solid rgba(255,179,71,0.28);
  border-radius: 10px;
  background: rgba(255,179,71,0.06);
  color: var(--g-amber);
  font-family: "DM Sans", sans-serif;
}

/* ── Tab buttons ─────────────────────────────────── */
button[data-baseweb="tab"] {
  border-radius: 8px;
  font-family: "DM Sans", sans-serif;
  font-weight: 500;
  color: var(--g-text-soft);
  transition: background 200ms ease, color 200ms ease, transform 180ms ease;
}

button[data-baseweb="tab"][aria-selected="true"] {
  background: var(--g-teal-dim);
  color: var(--g-teal) !important;
  transform: translateY(-1px);
  box-shadow: 0 0 16px rgba(0,235,192,0.12);
}

/* ── Status widget ───────────────────────────────── */
div[data-testid="stStatusWidget"] {
  border: 1px solid var(--g-border);
  border-radius: 12px;
  background: var(--g-surface);
}

/* ── Dataframe ───────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--g-border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--g-surface);
}

/* ── Expander ────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--g-border) !important;
  border-radius: 12px;
  background: var(--g-surface);
}

[data-testid="stExpander"] summary {
  color: var(--g-text-soft);
  font-family: "DM Sans", sans-serif;
}

/* ── Info / success / error alerts ──────────────── */
[data-testid="stAlert"] {
  border-radius: 12px;
  border-left-width: 3px;
}

/* ════════════════════════════════════════════════ */
/*  HERO COMPONENT                                  */
/* ════════════════════════════════════════════════ */

.g-hero {
  position: relative;
  border: 1px solid var(--g-border);
  border-radius: 20px;
  padding: 1.6rem 1.8rem;
  background:
    linear-gradient(135deg, rgba(0,235,192,0.04) 0%, transparent 60%),
    var(--g-surface);
  overflow: hidden;
  animation: gFadeUp 500ms ease-out;
  margin-bottom: 0.6rem;
}

/* Subtle grid overlay */
.g-hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,235,192,0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,235,192,0.04) 1px, transparent 1px);
  background-size: 36px 36px;
  pointer-events: none;
  border-radius: inherit;
}

/* Sweep shimmer */
.g-hero::after {
  content: "";
  position: absolute;
  top: -60%;
  left: -30%;
  width: 35%;
  height: 220%;
  background: linear-gradient(115deg, transparent, rgba(0,235,192,0.05), transparent);
  transform: rotate(20deg);
  animation: gSweep 7s linear infinite;
  pointer-events: none;
}

.g-hero-inner {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr);
  gap: 1.4rem;
  align-items: center;
  position: relative;
  z-index: 1;
}

.g-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: var(--g-teal-dim);
  border: 1px solid var(--g-border-lit);
  border-radius: 999px;
  padding: 0.22rem 0.7rem;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
  color: var(--g-teal);
  letter-spacing: 0.05em;
  margin-bottom: 0.6rem;
}

.g-badge-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--g-teal);
  animation: gDotPulse 1.8s ease-out infinite;
}

.g-hero-title {
  margin: 0 0 0.4rem 0;
  font-family: "Syne", sans-serif;
  font-weight: 800;
  font-size: clamp(2rem, 3.2vw, 3.2rem);
  line-height: 1.04;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #e0eef8 0%, var(--g-teal) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.g-hero-sub {
  margin: 0;
  font-family: "DM Sans", sans-serif;
  font-weight: 300;
  color: var(--g-text-soft);
  font-size: 0.96rem;
  line-height: 1.55;
  max-width: 440px;
}

.g-hero-note {
  margin-top: 0.55rem;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.72rem;
  color: #304050;
  letter-spacing: 0.04em;
}

/* Right panel: telemetry */
.g-telem-panel {
  border: 1px solid var(--g-border);
  border-radius: 16px;
  background: rgba(255,255,255,0.018);
  backdrop-filter: blur(4px);
  padding: 1rem 1.1rem;
}

.g-telem-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.75rem;
}

.g-live-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--g-teal);
  flex-shrink: 0;
  animation: gDotPulse 1.9s ease-out infinite;
}

.g-telem-title {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
  color: var(--g-text-soft);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.g-telem-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.5rem;
}

.g-telem-cell {
  border: 1px solid var(--g-border);
  border-radius: 10px;
  background: rgba(0,235,192,0.03);
  padding: 0.5rem 0.6rem;
  transition: border-color 300ms ease;
}

.g-telem-cell:hover {
  border-color: var(--g-border-lit);
}

.g-telem-cell-label {
  display: block;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.62rem;
  color: var(--g-text-soft);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  margin-bottom: 0.2rem;
}

.g-telem-cell-val {
  display: block;
  font-family: "DM Sans", sans-serif;
  font-weight: 600;
  font-size: 0.82rem;
  color: var(--g-text);
}

/* ════════════════════════════════════════════════ */
/*  TICKER STRIP                                    */
/* ════════════════════════════════════════════════ */

.g-ticker-strip {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  border: 1px solid var(--g-border);
  border-radius: 10px;
  background: var(--g-surface);
  padding: 0.42rem 0.7rem;
  overflow: hidden;
  margin: 0.5rem 0;
}

.g-ticker-overflow {
  overflow: hidden;
  white-space: nowrap;
  flex: 1;
}

.g-ticker-track {
  display: inline-flex;
  gap: 0.5rem;
  animation: gTicker 20s linear infinite;
}

.g-ticker-chip {
  border: 1px solid rgba(0,235,192,0.15);
  background: rgba(0,235,192,0.05);
  color: rgba(0,235,192,0.7);
  border-radius: 999px;
  padding: 0.14rem 0.55rem;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.65rem;
  letter-spacing: 0.04em;
  white-space: nowrap;
}

/* ════════════════════════════════════════════════ */
/*  SECTION TITLE                                   */
/* ════════════════════════════════════════════════ */

.g-section-title {
  margin-top: 0.5rem;
  margin-bottom: 0.1rem;
  font-family: "Syne", sans-serif;
  font-weight: 700;
  font-size: 1.25rem;
  color: var(--g-text);
  letter-spacing: -0.01em;
  animation: gFadeUp 360ms ease-out;
}

/* ════════════════════════════════════════════════ */
/*  SIGNAL BAND                                     */
/* ════════════════════════════════════════════════ */

.g-signal-panel {
  border: 1px solid var(--g-border);
  border-radius: 16px;
  background: var(--g-surface);
  padding: 1rem 1.2rem;
  margin: 0.5rem 0 1rem;
  animation: gFadeUp 400ms ease-out;
}

.g-signal-panel-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.65rem;
  border-bottom: 1px solid var(--g-border);
}

.g-signal-panel-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  color: var(--g-text-soft);
  letter-spacing: 0.07em;
  text-transform: uppercase;
}

.g-bar-row {
  display: grid;
  grid-template-columns: 108px 1fr 52px;
  gap: 0.65rem;
  align-items: center;
  margin: 0.55rem 0;
}

.g-bar-label {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.68rem;
  color: var(--g-text-soft);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.g-bar-track {
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.05);
  overflow: hidden;
  position: relative;
}

.g-bar-fill {
  height: 100%;
  border-radius: 999px;
  position: relative;
  transform-origin: left center;
  animation: gFill 820ms cubic-bezier(0.22,1,0.36,1) both;
}

.g-bar-fill.safety   { background: linear-gradient(90deg, #00b87a, #00e5a0); box-shadow: 0 0 12px rgba(0,229,160,0.35); }
.g-bar-fill.risk     { background: linear-gradient(90deg, #ffb347, #ff4060); box-shadow: 0 0 12px rgba(255,64,96,0.3); }
.g-bar-fill.conf     { background: linear-gradient(90deg, #46b3ff, #00ebc0); box-shadow: 0 0 12px rgba(0,235,192,0.3); }
.g-bar-fill.diffuse  { background: linear-gradient(90deg, #ff8f40, #ff4060); box-shadow: 0 0 12px rgba(255,64,96,0.25); }

.g-bar-val {
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
  color: var(--g-text);
  text-align: right;
}

/* ════════════════════════════════════════════════ */
/*  VERDICT CARD                                    */
/* ════════════════════════════════════════════════ */

.g-verdict-card {
  border-radius: 16px;
  padding: 1.25rem 1.5rem;
  margin: 0.6rem 0;
  display: flex;
  align-items: flex-start;
  gap: 1.1rem;
  position: relative;
  overflow: hidden;
  animation: gFadeUp 450ms ease-out;
}

.g-verdict-card.safe {
  border: 1px solid rgba(0,229,160,0.3);
  background: linear-gradient(135deg, rgba(0,229,160,0.07) 0%, rgba(0,235,192,0.03) 100%);
}

.g-verdict-card.unsafe {
  border: 1px solid rgba(255,64,96,0.3);
  background: linear-gradient(135deg, rgba(255,64,96,0.07) 0%, rgba(255,64,96,0.03) 100%);
}

.g-verdict-icon {
  font-size: 1.9rem;
  line-height: 1;
  flex-shrink: 0;
  margin-top: 0.1rem;
}

.g-verdict-content { flex: 1; }

.g-verdict-decision {
  font-family: "Syne", sans-serif;
  font-weight: 800;
  font-size: 1.1rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}

.g-verdict-card.safe   .g-verdict-decision { color: var(--g-green); }
.g-verdict-card.unsafe .g-verdict-decision { color: var(--g-red); }

.g-verdict-text {
  font-family: "DM Sans", sans-serif;
  font-size: 0.9rem;
  color: var(--g-text-soft);
  line-height: 1.5;
}

.g-verdict-tag {
  position: absolute;
  top: 1rem;
  right: 1.2rem;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.6rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.45;
}

/* ════════════════════════════════════════════════ */
/*  INSUFFICIENT EVIDENCE                           */
/* ════════════════════════════════════════════════ */

.g-evidence-warn {
  border: 1px solid rgba(255,179,71,0.3);
  border-radius: 12px;
  background: rgba(255,179,71,0.05);
  padding: 0.8rem 1rem;
  font-family: "DM Sans", sans-serif;
  font-size: 0.88rem;
  color: var(--g-amber);
  margin: 0.5rem 0;
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
}

/* ════════════════════════════════════════════════ */
/*  ANIMATIONS                                      */
/* ════════════════════════════════════════════════ */

@keyframes gFadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes gSweep {
  from { transform: translateX(-30%) rotate(20deg); }
  to   { transform: translateX(340%) rotate(20deg); }
}

@keyframes gScanLine {
  0%   { opacity: 0; transform: translateY(0px); }
  10%  { opacity: 0.6; }
  90%  { opacity: 0.6; }
  100% { opacity: 0; transform: translateY(200px); }
}

@keyframes gDotPulse {
  0%   { box-shadow: 0 0 0 0 rgba(0,235,192,0.5); }
  70%  { box-shadow: 0 0 0 7px rgba(0,235,192,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,235,192,0); }
}

@keyframes gTicker {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}

@keyframes gFill {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}

/* ── Responsive ──────────────────────────────── */
@media (max-width: 768px) {
  .g-hero-inner    { grid-template-columns: 1fr; }
  .g-telem-grid    { grid-template-columns: 1fr; }
  .g-bar-row       { grid-template-columns: 80px 1fr 44px; }
  .main .block-container { padding-top: 0.6rem; }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(st) -> None:
    st.markdown(
        """
<div class="g-hero">
  <div class="g-hero-inner">
    <div>
      <div class="g-badge">
        <span class="g-badge-dot"></span>
        Guard Studio · v2
      </div>
      <h1 class="g-hero-title">Multimodal Safety<br>Intelligence</h1>
      <p class="g-hero-sub">
        Upload one video and generate a policy-grade risk brief spanning
        temporal attribution, behavioral signals, cross-modal contradiction,
        and confidence diagnostics.
      </p>
      <p class="g-hero-note">⌘ single-file interactive analysis mode</p>
    </div>
    <div class="g-telem-panel">
      <div class="g-telem-header">
        <span class="g-live-dot"></span>
        <span class="g-telem-title">Pipeline · Live &amp; Ready</span>
      </div>
      <div class="g-telem-grid">
        <div class="g-telem-cell">
          <span class="g-telem-cell-label">Signal Stack</span>
          <span class="g-telem-cell-val">Frame · Audio · OCR</span>
        </div>
        <div class="g-telem-cell">
          <span class="g-telem-cell-label">Reasoning</span>
          <span class="g-telem-cell-val">Vector · LLM · Temporal</span>
        </div>
        <div class="g-telem-cell">
          <span class="g-telem-cell-label">Risk Surface</span>
          <span class="g-telem-cell-val">Behavioral · Diffusion</span>
        </div>
        <div class="g-telem-cell">
          <span class="g-telem-cell-label">Output</span>
          <span class="g-telem-cell-val">Decision Dashboard</span>
        </div>
      </div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_live_surface_strip(st) -> None:
    st.markdown(
        """
<div class="g-ticker-strip">
  <span class="g-live-dot"></span>
  <div class="g-ticker-overflow">
    <div class="g-ticker-track">
      <span class="g-ticker-chip">Signal Extraction</span>
      <span class="g-ticker-chip">Temporal Attribution</span>
      <span class="g-ticker-chip">Cross-modal Contradiction</span>
      <span class="g-ticker-chip">Behavioral Risk Blending</span>
      <span class="g-ticker-chip">Policy Verdict Synthesis</span>
      <span class="g-ticker-chip">Network Diffusion Scoring</span>
      <span class="g-ticker-chip">Creator Profile Lookup</span>
      <span class="g-ticker-chip">Adversarial Evasion Check</span>
      <span class="g-ticker-chip">Signal Extraction</span>
      <span class="g-ticker-chip">Temporal Attribution</span>
      <span class="g-ticker-chip">Cross-modal Contradiction</span>
      <span class="g-ticker-chip">Behavioral Risk Blending</span>
      <span class="g-ticker-chip">Policy Verdict Synthesis</span>
      <span class="g-ticker-chip">Network Diffusion Scoring</span>
      <span class="g-ticker-chip">Creator Profile Lookup</span>
      <span class="g-ticker-chip">Adversarial Evasion Check</span>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _none_if_blank(value: str) -> Optional[str]:
    normalized = str(value or "").strip()
    return normalized or None


def _query_flag(st, key: str) -> bool:
    raw = st.query_params.get(key, "0")
    if isinstance(raw, list):
        raw = raw[0] if raw else "0"
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _upload_size_mb(uploaded) -> float:
    size = getattr(uploaded, "size", None)
    if size is None:
        try:
            size = len(uploaded.getbuffer())
        except Exception:
            size = 0
    return float(size) / (1024.0 * 1024.0)


def _render_uploaded_file_summary(st, uploaded) -> None:
    extension = Path(uploaded.name).suffix.lower().lstrip(".") or "unknown"
    size_mb = _upload_size_mb(uploaded)
    c1, c2, c3 = st.columns(3)
    c1.metric("File Name", uploaded.name)
    c2.metric("File Size (MB)", f"{size_mb:.2f}")
    c3.metric("Format", extension.upper())


def _collect_ui_settings(st) -> Dict[str, object]:
    with st.sidebar:
        st.markdown("## Control Deck")
        st.caption("Tune extraction depth and moderation behavior before running analysis.")

        region = st.selectbox(
            "Region",
            ["auto", "global", "south_asia", "mena", "eu", "us"],
            index=0,
        )
        persist_outputs = st.checkbox("Persist outputs to guard_outputs", value=False)

        with st.expander("Signal Extraction", expanded=True):
            frame_count = st.slider("Frame samples", min_value=1, max_value=12, value=6)
            disable_whisper = st.checkbox("Disable Whisper transcription", value=False)
            disable_ocr = st.checkbox("Disable OCR", value=False)

        with st.expander("Temporal Dynamics", expanded=False):
            temporal_window_sec = st.slider(
                "Window (seconds)",
                min_value=3.0,
                max_value=8.0,
                value=5.0,
                step=0.5,
            )
            temporal_recency_bias = st.slider(
                "Recency bias",
                min_value=0.0,
                max_value=2.5,
                value=1.25,
                step=0.05,
            )
            temporal_max_segments = st.slider("Max segments", min_value=6, max_value=30, value=18)

        with st.expander("Advanced Paths", expanded=False):
            phase1_model = _none_if_blank(st.text_input("Phase-1 model path", value=""))
            phase1_run_summary = _none_if_blank(st.text_input("Phase-1 run summary path", value=""))
            creator_profile_path = _none_if_blank(st.text_input("Creator profile path", value=""))
            federated_export_path = _none_if_blank(st.text_input("Federated export path", value=""))

    return {
        "region": region,
        "persist_outputs": persist_outputs,
        "frame_count": frame_count,
        "disable_whisper": disable_whisper,
        "disable_ocr": disable_ocr,
        "temporal_window_sec": temporal_window_sec,
        "temporal_recency_bias": temporal_recency_bias,
        "temporal_max_segments": temporal_max_segments,
        "phase1_model": phase1_model,
        "phase1_run_summary": phase1_run_summary,
        "creator_profile_path": creator_profile_path,
        "federated_export_path": federated_export_path,
    }


def _render_report_ui(st, report: Dict[str, object]) -> None:
    st.markdown('<div class="g-section-title">Decision Dashboard</div>', unsafe_allow_html=True)

    safety     = _to_float(report.get("overall_safety_score_out_of_100"))
    risk       = _to_float(report.get("overall_risk_score_out_of_100"))
    confidence = _to_float(report.get("analysis_confidence"))
    diffusion  = _to_float(report.get("network_diffusion_risk"))
    age_rating = str(report.get("content_age_rating", "unknown"))

    safety     = max(0.0, min(100.0, safety))
    risk       = max(0.0, min(100.0, risk))
    confidence = max(0.0, min(100.0, confidence))
    diffusion  = max(0.0, min(100.0, diffusion))

    # ── Signal band ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="g-signal-panel">
  <div class="g-signal-panel-header">
    <span class="g-live-dot"></span>
    <span class="g-signal-panel-label">Signal Readout</span>
  </div>

  <div class="g-bar-row">
    <span class="g-bar-label">Safety</span>
    <div class="g-bar-track">
      <div class="g-bar-fill safety" style="width:{safety:.2f}%"></div>
    </div>
    <span class="g-bar-val">{safety:.1f}</span>
  </div>

  <div class="g-bar-row">
    <span class="g-bar-label">Risk</span>
    <div class="g-bar-track">
      <div class="g-bar-fill risk" style="width:{risk:.2f}%"></div>
    </div>
    <span class="g-bar-val">{risk:.1f}</span>
  </div>

  <div class="g-bar-row">
    <span class="g-bar-label">Confidence</span>
    <div class="g-bar-track">
      <div class="g-bar-fill conf" style="width:{confidence:.2f}%"></div>
    </div>
    <span class="g-bar-val">{confidence:.1f}</span>
  </div>

  <div class="g-bar-row">
    <span class="g-bar-label">Diffusion</span>
    <div class="g-bar-track">
      <div class="g-bar-fill diffuse" style="width:{diffusion:.2f}%"></div>
    </div>
    <span class="g-bar-val">{diffusion:.1f}</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metric row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Safety",     f"{safety:.2f}")
    m2.metric("Risk",       f"{risk:.2f}")
    m3.metric("Confidence", f"{confidence:.2f}")
    m4.metric("Diffusion",  f"{diffusion:.2f}")
    m5.metric("Age Rating", age_rating)

    # ── Verdict card ──────────────────────────────────────────────────────────
    verdict        = str(report.get("verdict", "No verdict generated."))
    safe_to_watch  = _to_bool(report.get("safe_to_watch", False), default=False)
    verdict_cls    = "safe" if safe_to_watch else "unsafe"
    verdict_icon   = "🛡️" if safe_to_watch else "⚠️"
    verdict_label  = "SAFE TO WATCH — CLEARED" if safe_to_watch else "NOT SAFE — FLAGGED"

    st.markdown(
        f"""
<div class="g-verdict-card {verdict_cls}">
  <div class="g-verdict-icon">{verdict_icon}</div>
  <div class="g-verdict-content">
    <div class="g-verdict-decision">{verdict_label}</div>
    <div class="g-verdict-text">{verdict}</div>
  </div>
  <span class="g-verdict-tag">Policy Decision</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ── Insufficient evidence notice ──────────────────────────────────────────
    if _to_bool(report.get("insufficient_evidence", False), default=False):
        st.markdown(
            """
<div class="g-evidence-warn">
  <span>⚠</span>
  <span>Marked as <strong>insufficient evidence</strong>. Enable or improve extractors
  (audio / OCR / context) for stronger confidence before acting on this result.</span>
</div>
            """,
            unsafe_allow_html=True,
        )

    overview_tab, category_tab, temporal_tab, evidence_tab, raw_tab = st.tabs(
        ["Overview", "Category Map", "Temporal View", "Evidence", "Raw JSON"]
    )

    with overview_tab:
        left, right = st.columns(2)

        warning_notes = report.get("warning_notes") or []
        left.markdown("#### Risk Notes")
        if warning_notes:
            for note in warning_notes:
                left.write(f"- {note}")
        else:
            left.caption("No high-risk warning notes were emitted.")

        top_positive = report.get("top_positive_signals") or []
        right.markdown("#### Positive Signals")
        if top_positive:
            for signal in top_positive:
                right.write(f"- {signal}")
        else:
            right.caption("No strong positive value signals were found.")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Societal Benefit",  f"{_to_float(report.get('societal_benefit_index')):.2f}")
        q2.metric("Viral Harm",        f"{_to_float(report.get('viral_harm_potential')):.2f}")
        q3.metric("Contradiction",     f"{_to_float(report.get('cross_modal_contradiction_score')):.2f}")
        q4.metric("Evasion",           f"{_to_float(report.get('adversarial_evasion_score')):.2f}")

    with category_tab:
        category_breakdown = report.get("category_breakdown") or {}
        if isinstance(category_breakdown, dict) and category_breakdown:
            rows = [
                {"category": key, "score": _to_float(value)}
                for key, value in category_breakdown.items()
            ]
            rows.sort(key=lambda item: item["score"], reverse=True)
            st.dataframe(
                rows,
                hide_index=True,
                width="stretch",
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "Score",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.2f",
                    )
                },
            )
        else:
            st.caption("No category breakdown is available for this report.")

    with temporal_tab:
        temporal_segments = report.get("temporal_segments") or []
        if temporal_segments:
            segment_rows = []
            for segment in temporal_segments:
                start_sec  = _to_float(segment.get("start_sec"))
                end_sec    = _to_float(segment.get("end_sec"))
                dominant   = ", ".join(segment.get("dominant_categories") or [])
                segment_rows.append(
                    {
                        "window": f"{start_sec:.1f}s → {end_sec:.1f}s",
                        "max_risk_score": _to_float(segment.get("max_risk_score")),
                        "dominant_categories": dominant,
                    }
                )

            st.dataframe(
                segment_rows,
                hide_index=True,
                width="stretch",
                column_config={
                    "max_risk_score": st.column_config.ProgressColumn(
                        "Max Risk",
                        min_value=0.0,
                        max_value=100.0,
                        format="%.2f",
                    )
                },
            )
        else:
            st.caption("Temporal segment attribution was not generated.")

        timestamp_attr = report.get("timestamp_attribution") or []
        if timestamp_attr:
            with st.expander("Top Timestamp Attribution", expanded=False):
                st.json(timestamp_attr)

    with evidence_tab:
        engine    = report.get("engine_details") or {}
        e1, e2, e3 = st.columns(3)
        e1.metric("GROQ LLM",   "ON" if _to_bool(engine.get("used_groq"),        default=False) else "OFF")
        e2.metric("Embeddings", "ON" if _to_bool(engine.get("used_embeddings"),  default=False) else "OFF")
        e3.metric("ChromaDB",   "ON" if _to_bool(engine.get("used_chromadb"),    default=False) else "OFF")
        llm_error = str(engine.get("llm_error") or "").strip()
        if llm_error and not _to_bool(engine.get("used_groq"), default=False):
            st.caption(f"LLM fallback reason: {llm_error}")

        network_details = report.get("network_diffusion_details") or {}
        if network_details:
            st.markdown("#### Network Diffusion Components")
            st.json(network_details)

        metadata = report.get("metadata") or {}
        if metadata:
            with st.expander("Metadata Snapshot", expanded=False):
                st.json(metadata)

    with raw_tab:
        st.json(report)
        report_blob = json.dumps(report, indent=2, ensure_ascii=True).encode("utf-8")
        timestamp   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download Report JSON",
            data=report_blob,
            file_name=f"guard_report_{timestamp}.json",
            mime="application/json",
            width="stretch",
        )


def main() -> None:
    st = _load_ui_modules()

    st.set_page_config(page_title="Guard Studio", layout="wide")
    safe_ui = _query_flag(st, "safeui")
    if safe_ui:
        st.title("Guard Studio")
        st.caption(
            "Safe UI mode is enabled. Animations and custom theme layers are disabled. "
            "Remove safeui=1 from the URL to restore the full animated experience."
        )
    else:
        _inject_ui_theme(st)
        _render_hero(st)
        _render_live_surface_strip(st)

    settings   = _collect_ui_settings(st)
    output_dir = "guard_outputs"

    st.markdown('<div class="g-section-title">Upload Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a video file to begin",
        type=VIDEO_TYPES,
        help="Supported formats: MP4, MOV, MKV, AVI, WEBM",
    )

    if uploaded is None:
        st.info("Select a video file to start analysis.")
        return

    _render_uploaded_file_summary(st, uploaded)

    run_clicked = st.button("Run Full Guard Analysis", type="primary", width="stretch")
    if not run_clicked:
        return

    output_path = Path(output_dir).expanduser().resolve()
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = Path(temp_dir) / uploaded.name
        temp_video.write_bytes(uploaded.getbuffer())

        with st.status("Running Guard analysis pipeline...", expanded=True) as status:
            status.write("Extracting multimodal signals from the uploaded media.")
            status.write("Computing vector, temporal, and behavioral safety metrics.")
            try:
                report = _analyze_video(
                    video_path=temp_video,
                    output_dir=output_path,
                    frame_count=int(settings["frame_count"]),
                    disable_whisper=bool(settings["disable_whisper"]),
                    disable_ocr=bool(settings["disable_ocr"]),
                    phase1_model=settings["phase1_model"],
                    phase1_run_summary=settings["phase1_run_summary"],
                    temporal_window_sec=float(settings["temporal_window_sec"]),
                    temporal_recency_bias=float(settings["temporal_recency_bias"]),
                    temporal_max_segments=int(settings["temporal_max_segments"]),
                    region_arg=str(settings["region"]),
                    creator_profile_path=settings["creator_profile_path"],
                    federated_export_path=settings["federated_export_path"],
                    source_video_name=uploaded.name,
                )
                if bool(settings["persist_outputs"]):
                    _save_outputs(output_path, report)
                    status.write(f"Saved report artifacts to: {output_path}")
                status.update(label="Guard analysis complete", state="complete")
            except Exception as error:
                status.update(label="Guard analysis failed", state="error")
                st.error(f"Analysis failed: {error}")
                return

    st.success("Guard analysis complete. Full output is shown below.")
    if bool(settings["persist_outputs"]):
        st.caption(f"Report files were written to: {output_path}")
    else:
        st.caption("UI-only mode is active. No files were written.")
    _render_report_ui(st, report)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()