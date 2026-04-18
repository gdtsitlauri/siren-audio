from pathlib import Path

from siren.datasets import autodetect_asvspoof_layout


def test_autodetect_asvspoof_layout(tmp_path: Path) -> None:
    flac_dir = tmp_path / "LA" / "ASVspoof2019_LA_dev" / "flac"
    protocol_dir = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols"
    flac_dir.mkdir(parents=True)
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt").write_text("dummy\n", encoding="utf-8")
    detected = autodetect_asvspoof_layout(tmp_path)
    assert detected["audio_root"].endswith("flac")
    assert detected["protocol_path"].endswith(".txt")
