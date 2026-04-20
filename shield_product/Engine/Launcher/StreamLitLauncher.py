import importlib
from pathlib import Path
from typing import Dict, List

from .FileSegregator import segregate_files
from .TaskPerformer import TaskPerformer
from Runner.Monitor.LogMonitor import LogMonitor


def _load_ui_modules():
    try:
        st = importlib.import_module("streamlit")
        pd = importlib.import_module("pandas")
    except ImportError as error:
        missing = error.name or "streamlit/pandas"
        raise RuntimeError(
            f"Missing dashboard dependency: {missing}. Install with 'pip install -r requirements.txt'."
        ) from error
    return st, pd


def _compute_metrics(results: List[Dict[str, object]], threshold: float) -> Dict[str, object]:
    tp = fp = tn = fn = 0
    for result in results:
        label = result.get("label")
        if label is None:
            continue
        pred = 1 if float(result.get("harm_score_pct", 0.0)) >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def _run_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    limit: int,
    threshold: float,
    enable_transcription: bool,
    enable_ocr: bool,
) -> List[Dict[str, object]]:
    st, _ = _load_ui_modules()
    items = segregate_files(dataset_dir, max_files=limit if limit > 0 else None)
    monitor = LogMonitor(output_dir=output_dir)
    performer = TaskPerformer(
        monitor=monitor,
        enable_transcription=enable_transcription,
        enable_ocr=enable_ocr,
    )

    results: List[Dict[str, object]] = []
    progress = st.progress(0)
    for idx, item in enumerate(items, start=1):
        results.append(performer.process_item(item))
        progress.progress(int(idx / max(len(items), 1) * 100))

    st.success(f"Processed {len(items)} items")
    metrics = _compute_metrics(results, threshold)
    st.json(metrics)
    return results


def main() -> None:
    st, pd = _load_ui_modules()
    st.set_page_config(page_title="SHIELD Dashboard", layout="wide")
    st.title("SHIELD Dashboard")

    dataset_default = str(Path.cwd().parent)
    dataset_dir = st.text_input("Dataset directory", value=dataset_default)
    output_dir = st.text_input("Output directory", value="run_outputs")
    limit = st.number_input("Max files (0 = no limit)", min_value=0, value=100)
    threshold = st.slider("Harm score threshold (percent)", min_value=0.0, max_value=100.0, value=35.0)
    enable_transcription = st.checkbox("Enable transcription (requires whisper)", value=False)
    enable_ocr = st.checkbox("Enable OCR for images (requires pytesseract)", value=False)

    if st.button("Run analysis"):
        dataset_path = Path(dataset_dir).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        with st.spinner("Running pipeline..."):
            results = _run_pipeline(
                dataset_path,
                output_path,
                int(limit),
                float(threshold),
                enable_transcription,
                enable_ocr,
            )

        if results:
            combined = pd.DataFrame([entry.get("combined_scores_pct", {}) for entry in results])
            if not combined.empty:
                st.subheader("Average combined scores (percent)")
                st.bar_chart(combined.mean().sort_values(ascending=False))

            table = pd.DataFrame(results)
            st.subheader("Sample results")
            st.dataframe(table.head(200))


if __name__ == "__main__":
    main()
