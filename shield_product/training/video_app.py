import importlib
import tempfile
from pathlib import Path

from .inference import InferenceConfig, VideoAnalyzer


def _load_ui_modules():
    st = importlib.import_module("streamlit")
    pd = importlib.import_module("pandas")
    return st, pd


def main() -> None:
    st, pd = _load_ui_modules()
    st.set_page_config(page_title="SHIELD Video Analyzer", layout="wide")
    st.title("SHIELD Full Video Analyzer")

    model_path = st.text_input("Model path", value="training_runs_hate_nonhate_frames/model.pt")
    run_summary = st.text_input(
        "Run summary path (optional)", value="training_runs_hate_nonhate_frames/run_summary.json"
    )
    output_dir = st.text_input("Output directory", value="analysis_outputs")
    threshold = st.slider("Toxicity threshold", min_value=0.0, max_value=100.0, value=50.0)
    preview_frames = st.number_input("Preview frames", min_value=0, max_value=8, value=3)
    save_previews = st.checkbox("Save preview frames", value=True)

    tab_upload, tab_folder = st.tabs(["Analyze Upload", "Analyze Folder"])

    with tab_upload:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "webm"])
        if uploaded and st.button("Analyze uploaded video"):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_video = Path(temp_dir) / uploaded.name
                temp_video.write_bytes(uploaded.getbuffer())

                config = InferenceConfig(
                    model_path=Path(model_path).expanduser().resolve(),
                    run_summary_path=Path(run_summary).expanduser().resolve()
                    if run_summary.strip()
                    else None,
                    threshold=float(threshold),
                    preview_frames=int(preview_frames),
                    save_previews=save_previews,
                )
                analyzer = VideoAnalyzer(config)
                results, summary = analyzer.analyze_input(
                    input_path=temp_video,
                    output_dir=Path(output_dir).expanduser().resolve(),
                )

            if results:
                item = results[0]
                st.video(uploaded)
                st.metric("Harm score", f"{item['harm_score_pct']:.2f}%")
                st.metric("Recommendation", item["view_recommendation"])
                st.metric("Age preference", item["age_preference"])

                category_scores = item.get("category_scores_pct", {})
                if category_scores:
                    series = pd.Series(category_scores).sort_values(ascending=False)
                    st.bar_chart(series)

                st.subheader("Warnings")
                for note in item.get("warning_notes", []):
                    st.write(f"- {note}")

                st.subheader("Summary")
                st.json(summary)

    with tab_folder:
        folder_path = st.text_input("Folder path", value=str(Path.cwd().parent))
        if st.button("Analyze folder"):
            config = InferenceConfig(
                model_path=Path(model_path).expanduser().resolve(),
                run_summary_path=Path(run_summary).expanduser().resolve()
                if run_summary.strip()
                else None,
                threshold=float(threshold),
                preview_frames=int(preview_frames),
                save_previews=save_previews,
            )
            analyzer = VideoAnalyzer(config)
            with st.spinner("Analyzing folder videos..."):
                results, summary = analyzer.analyze_input(
                    input_path=Path(folder_path).expanduser().resolve(),
                    output_dir=Path(output_dir).expanduser().resolve(),
                )

            st.success(f"Processed {summary['total_videos']} videos")
            st.json(summary)

            if results:
                table = pd.DataFrame(results)
                st.dataframe(table[["video_path", "harm_score_pct", "classification", "view_recommendation", "age_preference"]])


if __name__ == "__main__":
    main()
