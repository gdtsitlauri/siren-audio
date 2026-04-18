from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    key: str
    title: str
    mode: str
    summary: str


MODULE_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec(
        key="core_adversarial_audio",
        title="Core Adversarial Audio",
        mode="offensive-research",
        summary="Wavelet-inspired perturbation research for offline adversarial audio evaluation.",
    ),
    ModuleSpec(
        key="music_hidden_commands",
        title="Music Adversarial Attacks",
        mode="offensive-research",
        summary="Embedding-like command hiding simulations for music carrier evaluation and detection stress tests.",
    ),
    ModuleSpec(
        key="voice_cloning_detection",
        title="Voice Cloning Detection",
        mode="defensive",
        summary="Training-free audio forensics with explainable signal-level scores.",
    ),
    ModuleSpec(
        key="speaker_spoofing",
        title="Speaker Spoofing",
        mode="offensive-research",
        summary="Replay, synthesis, and conversion-style spoofing simulations for anti-spoof benchmarking.",
    ),
    ModuleSpec(
        key="ultrasonic_attacks",
        title="Ultrasonic Attacks",
        mode="offensive-research",
        summary="Ultrasonic carrier generation and detector benchmarking for high-frequency threat analysis.",
    ),
    ModuleSpec(
        key="defense_module",
        title="Unified Defense Module",
        mode="defensive",
        summary="Explainable multi-signal detection and low-latency screening pipeline.",
    ),
)


def module_manifest() -> list[dict[str, str]]:
    return [
        {"key": item.key, "title": item.title, "mode": item.mode, "summary": item.summary}
        for item in MODULE_SPECS
    ]
