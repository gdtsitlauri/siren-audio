from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from siren.attacks.core import SirenWaveAttack
from siren.audio import load_audio, save_audio
from siren.datasets import dataset_manifest
from siren.corpus import export_predictions, resolve_records, score_records, summarize_predictions
from siren.defenses.detector import DefenseSuite
from siren.evaluation import evaluate_suite
from siren.experiments import export_experiments, run_experiments, summarize_experiments
from siren.generator import export_generated_manifest, generate_synthetic_corpus
from siren.latency import benchmark_latency
from siren.modules import module_manifest
from siren.orchestration import run_research_bundle
from siren.ablation import export_ablation, run_siren_wave_ablation, summarize_ablation
from siren.paper_tables import render_ablation_table, render_experiment_table
from siren.presets import preset_manifest
from siren.runtime import inspect_runtime
from siren.streaming import StreamingDefense
from siren.training import simulate_adversarial_training


def _demo_signal(sample_rate: int = 16_000, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (
        0.4 * np.sin(2 * np.pi * 220.0 * t)
        + 0.2 * np.sin(2 * np.pi * 440.0 * t + 0.1)
        + 0.05 * np.sin(2 * np.pi * 880.0 * t)
    ).astype(np.float32)


def cmd_demo(_: argparse.Namespace) -> int:
    audio = _demo_signal()
    rows = evaluate_suite(audio, 16_000)
    print(json.dumps([asdict(row) for row in rows], indent=2))
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    audio, sr = load_audio(args.input)
    report = DefenseSuite().analyze(audio, sr)
    print(
        json.dumps(
            {
                "blocked": report.blocked,
                "risk_score": report.risk_score,
                "alerts": report.alerts,
                "components": report.components,
                "forensics": {
                    "score": report.forensics.score,
                    "label": report.forensics.label,
                    "breakdown": report.forensics.breakdown,
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    audio, sr = load_audio(args.input)
    result = SirenWaveAttack().generate(audio, sr)
    save_audio(args.output, result.audio, sr)
    print(json.dumps(result.metadata, indent=2))
    return 0


def cmd_datasets(_: argparse.Namespace) -> int:
    print(json.dumps(dataset_manifest(), indent=2))
    return 0


def cmd_runtime(_: argparse.Namespace) -> int:
    print(json.dumps(asdict(inspect_runtime()), indent=2))
    return 0


def cmd_stream_demo(args: argparse.Namespace) -> int:
    sample_rate = 16_000
    chunk_ms = args.chunk_ms
    chunk_samples = int(sample_rate * chunk_ms / 1000.0)
    defense = StreamingDefense(sample_rate=sample_rate, window_ms=chunk_ms, history_size=4)
    audio = _demo_signal(sample_rate=sample_rate, seconds=2.0)
    decisions = []
    for start in range(0, len(audio), chunk_samples):
        decision = defense.process_chunk(audio[start : start + chunk_samples])
        decisions.append(
            {
                "chunk_start": start,
                "average_risk": decision.average_risk,
                "blocked": decision.blocked,
                "latest_alerts": decision.latest.alerts,
            }
        )
    print(json.dumps(decisions, indent=2))
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    rows = run_experiments()
    payload = {
        "rows": [asdict(row) for row in rows],
        "summary": summarize_experiments(rows),
    }
    if args.output_dir:
        payload["exported"] = export_experiments(rows, Path(args.output_dir))
    print(json.dumps(payload, indent=2))
    return 0


def cmd_benchmark_corpus(args: argparse.Namespace) -> int:
    records = resolve_records(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        dataset_name=args.dataset_name,
        protocol_path=args.protocol,
    )
    predictions = score_records(records, target_sample_rate=args.sample_rate)
    payload = {
        "records": len(records),
        "summary": summarize_predictions(predictions),
    }
    if args.output_dir:
        payload["exported"] = export_predictions(predictions, args.output_dir)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_modules(_: argparse.Namespace) -> int:
    print(json.dumps(module_manifest(), indent=2))
    return 0


def cmd_presets(_: argparse.Namespace) -> int:
    print(json.dumps(preset_manifest(), indent=2))
    return 0


def cmd_generate_dataset(args: argparse.Namespace) -> int:
    samples = generate_synthetic_corpus(
        output_root=args.output_root,
        count_per_class=args.count_per_class,
        sample_rate=args.sample_rate,
        duration_s=args.duration_s,
    )
    payload = {"generated": len(samples)}
    if args.manifest:
        payload["manifest"] = export_generated_manifest(samples, args.manifest)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_train_demo(_: argparse.Namespace) -> int:
    batch = [_demo_signal(seconds=1.0), _demo_signal(seconds=1.2), _demo_signal(seconds=0.8)]
    report = simulate_adversarial_training(batch, 16_000)
    print(json.dumps(asdict(report), indent=2))
    return 0


def cmd_latency_demo(args: argparse.Namespace) -> int:
    audio = _demo_signal(sample_rate=args.sample_rate, seconds=args.duration_s)
    report = benchmark_latency(audio, sample_rate=args.sample_rate, chunk_ms=args.chunk_ms, runs=args.runs)
    print(json.dumps(asdict(report), indent=2))
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    rows = run_siren_wave_ablation()
    payload = {"rows": [asdict(row) for row in rows], "summary": summarize_ablation(rows)}
    if args.output_dir:
        payload["exported"] = export_ablation(rows, args.output_dir)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_make_tables(args: argparse.Namespace) -> int:
    payload = {}
    if args.experiment_summary and args.experiment_output:
        payload["experiment_table"] = render_experiment_table(args.experiment_summary, args.experiment_output)
    if args.ablation_summary and args.ablation_output:
        payload["ablation_table"] = render_ablation_table(args.ablation_summary, args.ablation_output)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_research_bundle(args: argparse.Namespace) -> int:
    report = run_research_bundle(args.output_dir)
    print(json.dumps(asdict(report), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siren", description="SIREN audio research framework")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the synthetic end-to-end benchmark demo")
    demo.set_defaults(func=cmd_demo)

    detect = sub.add_parser("detect", help="Run unified defensive analysis on an audio file")
    detect.add_argument("input")
    detect.set_defaults(func=cmd_detect)

    attack = sub.add_parser("attack", help="Generate a SIREN-WAVE sample for offline evaluation")
    attack.add_argument("input")
    attack.add_argument("output")
    attack.set_defaults(func=cmd_attack)

    ds = sub.add_parser("datasets", help="Show dataset catalog and local availability")
    ds.set_defaults(func=cmd_datasets)

    runtime = sub.add_parser("runtime", help="Inspect runtime backend and CUDA availability")
    runtime.set_defaults(func=cmd_runtime)

    stream = sub.add_parser("stream-demo", help="Run a sliding-window defense demo")
    stream.add_argument("--chunk-ms", type=int, default=250)
    stream.set_defaults(func=cmd_stream_demo)

    exp = sub.add_parser("experiments", help="Run reproducible synthetic experiment suite")
    exp.add_argument("--output-dir", default="")
    exp.set_defaults(func=cmd_experiments)

    corpus = sub.add_parser("benchmark-corpus", help="Benchmark detectors on a local audio corpus")
    corpus.add_argument("--dataset-root", default="")
    corpus.add_argument("--manifest", default="")
    corpus.add_argument("--dataset-name", default="custom")
    corpus.add_argument("--protocol", default="")
    corpus.add_argument("--sample-rate", type=int, default=16_000)
    corpus.add_argument("--output-dir", default="")
    corpus.set_defaults(func=cmd_benchmark_corpus)

    mods = sub.add_parser("modules", help="Show the six-module SIREN research surface")
    mods.set_defaults(func=cmd_modules)

    presets = sub.add_parser("presets", help="Show runtime and experiment presets")
    presets.set_defaults(func=cmd_presets)

    gen = sub.add_parser("generate-dataset", help="Generate a synthetic local benchmark corpus")
    gen.add_argument("--output-root", required=True)
    gen.add_argument("--manifest", default="")
    gen.add_argument("--count-per-class", type=int, default=4)
    gen.add_argument("--sample-rate", type=int, default=16_000)
    gen.add_argument("--duration-s", type=float, default=2.0)
    gen.set_defaults(func=cmd_generate_dataset)

    train = sub.add_parser("train-demo", help="Run a lightweight adversarial-training simulation summary")
    train.set_defaults(func=cmd_train_demo)

    latency = sub.add_parser("latency-demo", help="Estimate offline and streaming defense latency")
    latency.add_argument("--sample-rate", type=int, default=16_000)
    latency.add_argument("--chunk-ms", type=int, default=20)
    latency.add_argument("--duration-s", type=float, default=1.0)
    latency.add_argument("--runs", type=int, default=5)
    latency.set_defaults(func=cmd_latency_demo)

    ablation = sub.add_parser("ablation", help="Run SIREN-WAVE ablation benchmark")
    ablation.add_argument("--output-dir", default="")
    ablation.set_defaults(func=cmd_ablation)

    tables = sub.add_parser("make-tables", help="Render LaTeX tables from JSON summaries")
    tables.add_argument("--experiment-summary", default="")
    tables.add_argument("--experiment-output", default="")
    tables.add_argument("--ablation-summary", default="")
    tables.add_argument("--ablation-output", default="")
    tables.set_defaults(func=cmd_make_tables)

    bundle = sub.add_parser("research-bundle", help="Run synthetic experiment bundle and generate paper tables")
    bundle.add_argument("--output-dir", required=True)
    bundle.set_defaults(func=cmd_research_bundle)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
