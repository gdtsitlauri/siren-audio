from pathlib import Path

from siren.generator import export_generated_manifest, generate_synthetic_corpus
from siren.modules import module_manifest
from siren.presets import preset_manifest
from siren.training import simulate_adversarial_training


def test_module_manifest_exposes_six_modules() -> None:
    manifest = module_manifest()
    assert len(manifest) == 6


def test_presets_manifest_contains_expected_profiles() -> None:
    presets = preset_manifest()
    assert "low_vram_gtx1650" in presets
    assert "realtime_guard" in presets


def test_generate_synthetic_corpus_and_manifest(tmp_path: Path) -> None:
    samples = generate_synthetic_corpus(tmp_path / "generated", count_per_class=2, duration_s=0.2)
    assert samples
    manifest = export_generated_manifest(samples, tmp_path / "generated" / "manifest.csv")
    assert Path(manifest).exists()


def test_simulated_adversarial_training_returns_report() -> None:
    batch = [[0.0] * 256, [0.1] * 256]
    report = simulate_adversarial_training(batch, 16_000)
    assert report.robustness_gain >= 0.0
