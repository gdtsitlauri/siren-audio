"""SIREN audio research framework."""

from .ablation import export_ablation, run_siren_wave_ablation, summarize_ablation
from .config import RuntimeConfig
from .corpus import resolve_records, score_records, summarize_predictions
from .defenses.detector import DefenseSuite
from .datasets import (
    autodetect_asvspoof_layout,
    load_asvspoof_protocol,
    load_dataset_records,
    load_fakeavceleb_records,
    load_librispeech_records,
)
from .experiments import run_experiments, summarize_experiments
from .forensics.training_free import ForensicsReport, TrainingFreeDetector
from .generator import export_generated_manifest, generate_synthetic_corpus
from .latency import benchmark_latency
from .modules import module_manifest
from .orchestration import run_research_bundle
from .paper_tables import render_ablation_table, render_experiment_table
from .presets import preset_manifest
from .reporting import export_generic_json, export_markdown_table, export_metrics_csv
from .runtime import RuntimeInfo, inspect_runtime
from .streaming import StreamingDefense
from .training import simulate_adversarial_training

__all__ = [
    "DefenseSuite",
    "export_ablation",
    "ForensicsReport",
    "RuntimeInfo",
    "RuntimeConfig",
    "load_asvspoof_protocol",
    "load_dataset_records",
    "load_fakeavceleb_records",
    "load_librispeech_records",
    "resolve_records",
    "score_records",
    "StreamingDefense",
    "TrainingFreeDetector",
    "inspect_runtime",
    "export_generated_manifest",
    "export_generic_json",
    "export_markdown_table",
    "export_metrics_csv",
    "generate_synthetic_corpus",
    "benchmark_latency",
    "autodetect_asvspoof_layout",
    "module_manifest",
    "preset_manifest",
    "render_ablation_table",
    "render_experiment_table",
    "run_experiments",
    "run_research_bundle",
    "run_siren_wave_ablation",
    "summarize_predictions",
    "summarize_ablation",
    "summarize_experiments",
    "simulate_adversarial_training",
]
