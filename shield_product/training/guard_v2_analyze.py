import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from .v2 import GuardV2Config, GuardV2Pipeline, discover_video_assets, write_outputs
from .v2.reporting import build_quality_summary, build_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard V2 analyzer (rebuilt architecture)")
    parser.add_argument("--input", type=str, required=True, help="Video file or folder")
    parser.add_argument("--output-dir", type=str, default="guard_outputs")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--disable-whisper", action="store_true")
    parser.add_argument("--disable-ocr", action="store_true")
    parser.add_argument("--region", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-empty-input", action="store_true")
    parser.add_argument("--fail-on-item-error", action="store_true")
    args = parser.parse_args()

    started_at = _utc_now()
    started_perf = time.perf_counter()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    assets = discover_video_assets(input_path, limit=args.limit)
    if not assets and not args.allow_empty_input:
        raise SystemExit("No video files found for analysis")

    config = GuardV2Config(
        frame_count=int(args.frame_count),
        enable_whisper=not bool(args.disable_whisper),
        enable_ocr=not bool(args.disable_ocr),
        region=str(args.region),
    )
    pipeline = GuardV2Pipeline(config)
    completed, failed = pipeline.analyze_assets(assets, fail_on_item_error=bool(args.fail_on_item_error))

    elapsed_seconds = round(time.perf_counter() - started_perf, 4)
    run_metadata = {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "total_detected_videos": len(assets),
        "arguments": {
            "frame_count": int(args.frame_count),
            "disable_whisper": bool(args.disable_whisper),
            "disable_ocr": bool(args.disable_ocr),
            "region": str(args.region),
            "limit": args.limit,
            "allow_empty_input": bool(args.allow_empty_input),
            "fail_on_item_error": bool(args.fail_on_item_error),
        },
    }

    write_outputs(output_dir, completed, failed, run_metadata)

    summary = build_summary(completed, failed)
    quality_summary = build_quality_summary(completed, failed)

    print("Guard V2 analysis complete")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Completed videos: {summary['completed_videos']}")
    print(f"Failed videos: {summary['failed_videos']}")
    print(f"Safe to watch: {summary['safe_to_watch_count']}")
    print(f"Average safety score: {summary['average_safety_score']}")
    print(f"Average risk score: {summary['average_risk_score']}")
    print(f"Average network diffusion risk: {summary['average_network_diffusion_risk']}")
    print(f"Average analysis confidence: {summary['average_analysis_confidence']}")
    print(f"High risk count: {quality_summary['high_risk_count']}")
    print(f"Output directory: {output_dir}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
