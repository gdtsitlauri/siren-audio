import numpy as np

from siren.attacks.core import SirenWaveAttack
from siren.attacks.music import MusicHiddenCommandAttack
from siren.attacks.speaker import SpeakerSpoofSimulator


def _tone(sample_rate: int = 16_000, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return 0.4 * np.sin(2 * np.pi * 220.0 * t)


def test_siren_wave_changes_signal() -> None:
    audio = _tone()
    result = SirenWaveAttack().generate(audio, 16_000)
    assert result.audio.shape == audio.shape
    assert not np.allclose(result.audio, audio)


def test_music_attack_preserves_length() -> None:
    audio = _tone(seconds=3.0)
    result = MusicHiddenCommandAttack().generate(audio, 16_000)
    assert result.audio.shape == audio.shape


def test_speaker_spoof_modes_are_supported() -> None:
    audio = _tone()
    for mode in ("replay", "synthesis", "conversion"):
        result = SpeakerSpoofSimulator().generate(audio, 16_000, mode=mode)
        assert result.metadata["mode"] == mode
