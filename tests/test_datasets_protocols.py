from pathlib import Path

import numpy as np

from siren.audio import save_audio
from siren.datasets import load_asvspoof_protocol, load_fakeavceleb_records, load_librispeech_records


def _write_audio(path: Path, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    save_audio(path, audio, sample_rate)


def test_load_asvspoof_protocol_reads_records(tmp_path: Path) -> None:
    audio_root = tmp_path / "flac"
    _write_audio(audio_root / "LA_T_0000001.wav")
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("LA_0001 LA_T_0000001 - A01 spoof\n", encoding="utf-8")
    records = load_asvspoof_protocol(protocol, audio_root, split="dev")
    assert len(records) == 1
    assert records[0].label == "synthetic"
    assert records[0].attack_id == "A01"


def test_load_librispeech_records_marks_authentic(tmp_path: Path) -> None:
    _write_audio(tmp_path / "dev-clean" / "1272" / "128104" / "1272-128104-0001.flac")
    records = load_librispeech_records(tmp_path)
    assert len(records) == 1
    assert records[0].label == "authentic"
    assert records[0].speaker_id == "1272"


def test_load_fakeavceleb_records_reads_metadata(tmp_path: Path) -> None:
    _write_audio(tmp_path / "clips" / "sample.wav")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "path,label,split,speaker_id,method\nclips/sample.wav,fake,test,spk1,vocoder_x\n",
        encoding="utf-8",
    )
    records = load_fakeavceleb_records(tmp_path)
    assert len(records) == 1
    assert records[0].label == "synthetic"
    assert records[0].attack_id == "vocoder_x"
