from pathlib import Path

from training.v2.dataset import discover_video_assets


def test_discover_video_assets_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "z_video.mp4").write_bytes(b"a")
    (tmp_path / "not_video.txt").write_text("nope", encoding="utf-8")
    (tmp_path / ".hidden.mp4").write_bytes(b"x")

    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a_video.webm").write_bytes(b"b")

    assets = discover_video_assets(tmp_path)

    names = [asset.path.name for asset in assets]
    assert names == ["a_video.webm", "z_video.mp4"]
    assert [asset.source_id for asset in assets] == ["video_a_video", "video_z_video"]


def test_discover_video_assets_limit(tmp_path: Path) -> None:
    for idx in range(5):
        (tmp_path / f"video_{idx}.mp4").write_bytes(b"x")

    assets = discover_video_assets(tmp_path, limit=2)
    assert len(assets) == 2
