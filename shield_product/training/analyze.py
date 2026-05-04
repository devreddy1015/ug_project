import argparse
import logging
from pathlib import Path

from .inference import InferenceConfig, VideoAnalyzer


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="SHIELD video analyzer")
    parser.add_argument("--input", type=str, required=True, help="Video file or directory")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model.pt")
    parser.add_argument(
        "--run-summary",
        type=str,
        default=None,
        help="Optional run_summary.json generated during training",
    )
    parser.add_argument("--output-dir", type=str, default="analysis_outputs")
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--preview-frames", type=int, default=3)
    parser.add_argument("--save-previews", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    run_summary = Path(args.run_summary).expanduser().resolve() if args.run_summary else None
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    config = InferenceConfig(
        model_path=model_path,
        run_summary_path=run_summary,
        threshold=float(args.threshold),
        preview_frames=int(args.preview_frames),
        save_previews=bool(args.save_previews),
    )
    analyzer = VideoAnalyzer(config)
    _, summary = analyzer.analyze_input(input_path=input_path, output_dir=output_dir)

    logger.info("Video analysis complete")
    logger.info("Total videos: %s", summary["total_videos"])
    logger.info("Toxic videos: %s", summary["toxic_count"])
    logger.info("Average harm score: %s%%", summary["avg_harm_score_pct"])
    logger.info("Top risk video: %s", summary["max_harm_video"])
    logger.info("Results folder: %s", output_dir)


if __name__ == "__main__":
    main()
