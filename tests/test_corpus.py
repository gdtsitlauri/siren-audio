from pathlib import Path

import numpy as np

from siren.audio import save_audio
from siren.corpus import export_predictions, resolve_records, score_records, summarize_predictions


def _write_tone(path: Path, freq: float, sample_rate: int = 16_000, seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    audio = 0.25 * np.sin(2 * np.pi * freq * t)
    save_audio(path, audio, sample_rate)


def test_corpus_scoring_from_scanned_tree(tmp_path: Path) -> None:
    auth_dir = tmp_path / "test" / "authentic"
    synth_dir = tmp_path / "test" / "synthetic_spoof"
    auth_dir.mkdir(parents=True)
    synth_dir.mkdir(parents=True)
    _write_tone(auth_dir / "speaker1.wav", 220.0)
    _write_tone(synth_dir / "speaker2_fake.wav", 440.0)

    records = resolve_records(dataset_root=str(tmp_path), dataset_name="toy")
    assert len(records) == 2
    predictions = score_records(records)
    summary = summarize_predictions(predictions)
    assert "training_free" in summary
    exported = export_predictions(predictions, tmp_path / "out")
    assert Path(exported["predictions_csv"]).exists()


def test_corpus_scoring_from_manifest(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    _write_tone(wav, 330.0)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,label,dataset,split\n%s,authentic,toy,dev\n" % wav, encoding="utf-8")
    records = resolve_records(manifest_path=str(manifest), dataset_name="toy")
    assert len(records) == 1


def test_corpus_resolve_asvspoof_protocol(tmp_path: Path) -> None:
    wav = tmp_path / "audio" / "LA_T_1000001.wav"
    _write_tone(wav, 220.0)
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("LA_0001 LA_T_1000001 - A07 bonafide\n", encoding="utf-8")
    records = resolve_records(
        dataset_root=str(tmp_path / "audio"),
        dataset_name="asvspoof2019",
        protocol_path=str(protocol),
    )
    assert len(records) == 1
    assert records[0].label == "authentic"
