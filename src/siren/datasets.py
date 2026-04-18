from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class DatasetSpec:
    name: str
    purpose: str
    homepage: str
    expected_subdir: str
    download_hint: str


@dataclass(slots=True)
class AudioRecord:
    path: str
    label: str
    dataset: str
    split: str
    speaker_id: str = ""
    utterance_id: str = ""
    attack_id: str = ""


CATALOG: dict[str, DatasetSpec] = {
    "asvspoof2019": DatasetSpec(
        name="ASVspoof2019",
        purpose="Primary spoofing and anti-spoofing benchmark",
        homepage="https://www.asvspoof.org/",
        expected_subdir="data/asvspoof2019",
        download_hint="Download the LA split from the official ASVspoof site and unpack under data/asvspoof2019.",
    ),
    "fakeavceleb": DatasetSpec(
        name="FakeAVCeleb",
        purpose="Cross-modal deepfake dataset used as secondary benchmark",
        homepage="https://sites.google.com/view/fakeavcelebdash-lab/",
        expected_subdir="data/fakeavceleb",
        download_hint="Download the official release and place extracted audio under data/fakeavceleb.",
    ),
    "librispeech": DatasetSpec(
        name="LibriSpeech",
        purpose="Authentic speech baseline",
        homepage="https://www.openslr.org/12",
        expected_subdir="data/librispeech",
        download_hint="Download selected clean subsets and extract under data/librispeech.",
    ),
}


def dataset_status(root: str | Path = ".") -> dict[str, bool]:
    base = Path(root)
    return {key: (base / spec.expected_subdir).exists() for key, spec in CATALOG.items()}


def dataset_manifest(root: str | Path = ".") -> dict[str, dict[str, str | bool]]:
    base = Path(root)
    return {
        key: {
            "name": spec.name,
            "purpose": spec.purpose,
            "homepage": spec.homepage,
            "expected_path": str(base / spec.expected_subdir),
            "download_hint": spec.download_hint,
            "present": (base / spec.expected_subdir).exists(),
        }
        for key, spec in CATALOG.items()
    }


def _infer_label(path: Path) -> str:
    text = str(path).lower()
    synthetic_tokens = ("spoof", "fake", "synth", "cloned", "generated", "tts")
    return "synthetic" if any(token in text for token in synthetic_tokens) else "authentic"


def _infer_split(path: Path) -> str:
    text = str(path).lower()
    for token in ("train", "dev", "test", "eval"):
        if token in text:
            return token
    return "unknown"


def scan_audio_records(root: str | Path, dataset: str = "custom") -> list[AudioRecord]:
    base = Path(root)
    if not base.exists():
        return []
    files = sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
    )
    return [
        AudioRecord(
            path=str(path),
            label=_infer_label(path),
            dataset=dataset,
            split=_infer_split(path),
            speaker_id=path.parent.name,
            utterance_id=path.stem,
        )
        for path in files
    ]


def load_records_from_manifest(path: str | Path, dataset: str = "manifest") -> list[AudioRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    rows: list[AudioRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                AudioRecord(
                    path=row["path"],
                    label=row.get("label", "authentic"),
                    dataset=row.get("dataset", dataset),
                    split=row.get("split", "unknown"),
                    speaker_id=row.get("speaker_id", ""),
                    utterance_id=row.get("utterance_id", Path(row["path"]).stem),
                    attack_id=row.get("attack_id", ""),
                )
            )
    return rows


def records_by_split(records: Iterable[AudioRecord]) -> dict[str, list[AudioRecord]]:
    grouped: dict[str, list[AudioRecord]] = {}
    for record in records:
        grouped.setdefault(record.split, []).append(record)
    return grouped


def _resolve_relative_audio(base_dir: Path, candidate: str) -> Path:
    path = Path(candidate)
    return path if path.is_absolute() else (base_dir / path)


def load_asvspoof_protocol(
    protocol_path: str | Path,
    audio_root: str | Path,
    split: str = "",
    dataset: str = "asvspoof2019",
) -> list[AudioRecord]:
    protocol = Path(protocol_path)
    root = Path(audio_root)
    if not protocol.exists() or not root.exists():
        return []
    records: list[AudioRecord] = []
    with protocol.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, utterance_id, _, attack_id, key = parts[:5]
            label = "authentic" if key.lower() == "bonafide" else "synthetic"
            candidates = [
                root / f"{utterance_id}.flac",
                root / f"{utterance_id}.wav",
                root / speaker_id / f"{utterance_id}.flac",
                root / speaker_id / f"{utterance_id}.wav",
            ]
            audio_path = next((p for p in candidates if p.exists()), candidates[0])
            records.append(
                AudioRecord(
                    path=str(audio_path),
                    label=label,
                    dataset=dataset,
                    split=split or _infer_split(protocol),
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    attack_id=attack_id,
                )
            )
    return records


def _find_first_existing(base: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(base.rglob(pattern))
        if matches:
            return matches[0]
    return None


def autodetect_asvspoof_layout(root: str | Path) -> dict[str, str]:
    base = Path(root)
    if not base.exists():
        return {"audio_root": "", "protocol_path": ""}
    audio_root = _find_first_existing(
        base,
        (
            "*ASVspoof2019_LA_*/*/flac",
            "*ASVspoof2019_LA_*/*/*/flac",
            "*/flac",
            "flac",
        ),
    )
    protocol_path = _find_first_existing(
        base,
        (
            "*.LA.cm.dev*.txt",
            "*.LA.cm.train*.txt",
            "*.LA.cm.eval*.txt",
            "*protocol*/*.txt",
        ),
    )
    return {
        "audio_root": str(audio_root) if audio_root else "",
        "protocol_path": str(protocol_path) if protocol_path else "",
    }


def load_librispeech_records(root: str | Path, split: str = "", dataset: str = "librispeech") -> list[AudioRecord]:
    base = Path(root)
    if not base.exists():
        return []
    files = sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in {".flac", ".wav"})
    records: list[AudioRecord] = []
    for path in files:
        parts = path.stem.split("-")
        speaker_id = parts[0] if parts else path.parent.name
        utterance_id = path.stem
        records.append(
            AudioRecord(
                path=str(path),
                label="authentic",
                dataset=dataset,
                split=split or _infer_split(path),
                speaker_id=speaker_id,
                utterance_id=utterance_id,
            )
        )
    return records


def load_fakeavceleb_records(root: str | Path, split: str = "", dataset: str = "fakeavceleb") -> list[AudioRecord]:
    base = Path(root)
    if not base.exists():
        return []
    manifest_candidates = [
        base / "metadata.csv",
        base / "meta.csv",
        base / "labels.csv",
    ]
    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path is None:
        return scan_audio_records(base, dataset=dataset)

    rows: list[AudioRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rel_path = row.get("path") or row.get("audio_path") or row.get("file") or ""
            if not rel_path:
                continue
            audio_path = _resolve_relative_audio(base, rel_path)
            raw_label = (row.get("label") or row.get("class") or row.get("status") or "").lower()
            label = "synthetic" if raw_label in {"fake", "spoof", "synthetic", "generated"} else "authentic"
            rows.append(
                AudioRecord(
                    path=str(audio_path),
                    label=label,
                    dataset=dataset,
                    split=split or row.get("split", _infer_split(audio_path)),
                    speaker_id=row.get("speaker_id", ""),
                    utterance_id=row.get("utterance_id", audio_path.stem),
                    attack_id=row.get("attack_id", row.get("method", "")),
                )
            )
    return rows


def load_dataset_records(dataset_name: str, dataset_root: str | Path = "", protocol_path: str | Path = "") -> list[AudioRecord]:
    dataset_key = dataset_name.lower()
    root = Path(dataset_root) if dataset_root else Path(".")
    if dataset_key == "asvspoof2019":
        if not protocol_path:
            detected = autodetect_asvspoof_layout(root)
            protocol_path = detected["protocol_path"]
            root = Path(detected["audio_root"] or root)
        return load_asvspoof_protocol(protocol_path, root)
    if dataset_key == "librispeech":
        return load_librispeech_records(root)
    if dataset_key == "fakeavceleb":
        return load_fakeavceleb_records(root)
    if protocol_path:
        return load_records_from_manifest(protocol_path, dataset=dataset_name)
    return scan_audio_records(root, dataset=dataset_name)
