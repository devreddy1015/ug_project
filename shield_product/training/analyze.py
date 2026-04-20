import argparse
from pathlib import Path

from .inference import InferenceConfig, VideoAnalyzer


def main() -> None:
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

    print("Video analysis complete")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Toxic videos: {summary['toxic_count']}")
    print(f"Average harm score: {summary['avg_harm_score_pct']}%")
    print(f"Top risk video: {summary['max_harm_video']}")
    print(f"Results folder: {output_dir}")


if __name__ == "__main__":
    main()
