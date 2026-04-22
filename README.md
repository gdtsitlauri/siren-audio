# SIREN


SIREN stands for **Stealth Injection and Recognition of Encoded Neural-adversarial audio**. This repository provides a research-oriented, open-source framework for evaluating adversarial audio attacks, training-free audio forensics, spoofing detection, ultrasonic analysis, and unified defense orchestration.

SIREN is designed as a single workspace for:

- adversarial audio generation research
- audio deepfake and cloned-voice forensics
- spoofing and replay analysis
- ultrasonic threat monitoring
- explainable defensive scoring
- reproducible benchmark and paper-preparation workflows

The repository is intentionally structured as a research system rather than a product. It prioritizes reproducibility, transparency, controlled offline evaluation, and careful result reporting.


## Project Metadata

| Field | Value |
| --- | --- |
| Author | George David Tsitlauri |
| Affiliation | Dept. of Informatics & Telecommunications, University of Thessaly, Greece |
| Contact | gdtsitlauri@gmail.com |
| Year | 2026 |

## Scope

SIREN is designed for:

- offline adversarial audio research
- training-free deepfake and cloned-voice detection
- controlled attack simulation benchmarks
- reproducible experiment pipelines
- explainable defensive scoring

The repository intentionally keeps offensive functionality benchmark-oriented and sandboxed to offline research workflows. It does not include device-targeting playbooks, deployment bypass recipes, or abuse-focused automation.

## What SIREN Gives You

With the current repository state, you can:

- generate research-safe adversarial audio variants for controlled evaluation
- score audio with a training-free deepfake/cloned-voice detector
- benchmark defensive behavior on synthetic and real local corpora
- run ablations for the SIREN-WAVE design choices
- export CSV, JSON, Markdown, and LaTeX artifacts for reporting
- inspect runtime and low-latency screening behavior
- prepare a paper bundle directly from the repo

## Modules

1. Core adversarial audio research via `SIREN-WAVE`
2. Music-hidden command simulation
3. Training-free voice cloning detection
4. Speaker spoofing simulation and analysis
5. Ultrasonic attack analysis and detection
6. Unified defense module with explainable scoring

You can inspect the live module registry with:

```bash
PYTHONPATH=src python -m siren.cli modules
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[audio,dev]"
pytest
python -m siren.cli demo
```

## Public Dataset Setup

Public-download dataset setup can be done in two phases:

```bash
mkdir -p data/downloads
wget -c https://www.openslr.org/resources/12/dev-clean.tar.gz -P data/downloads
wget -c https://www.openslr.org/resources/12/test-clean.tar.gz -P data/downloads
wget -c -O data/downloads/LA.zip 'https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y'
bash scripts/setup_public_datasets.sh
```

Notes:

- `LibriSpeech` subsets above are directly downloadable from the official `OpenSLR` mirror.
- `ASVspoof2019 LA.zip` is downloadable from the official Edinburgh DataShare page.
- `FakeAVCeleb` still requires the official license/form workflow, so SIREN supports its layout but does not bypass the request process.
- after extraction, the large archives under `data/downloads/` can be deleted to save disk space

Without editable install, local development also works with:

```bash
PYTHONPATH=src pytest
PYTHONPATH=src python -m siren.cli demo
```

## Project Layout

- `src/siren/`: core package
- `tests/`: unit tests
- `configs/`: experiment presets
- `results/`: committed result placeholders and benchmark tables
- `paper/`: IEEE-style manuscript scaffold

Important subpaths:

- `src/siren/attacks/`: research-safe attack simulators
- `src/siren/forensics/`: training-free forensic analysis
- `src/siren/defenses/`: unified defense scoring
- `src/siren/experiments.py`: synthetic experiment runner
- `src/siren/corpus.py`: local corpus scoring and exports
- `src/siren/ablation.py`: SIREN-WAVE ablation workflow
- `src/siren/orchestration.py`: bundled research artifact generation
- `data/manifests/`: lightweight manifest-driven real-data smoke tests

## SIREN-WAVE

`SIREN-WAVE` is implemented here as a wavelet-inspired perturbation engine that:

- decomposes the signal into coarse and detail bands
- shapes perturbations by local masking energy
- exposes a stealth-vs-strength tradeoff
- supports research baselines for ablation and defense evaluation

In the current open-source implementation, SIREN-WAVE is used as a reproducible perturbation research mechanism rather than a deployment-focused attack package. The emphasis is on measurable behavior, explainable ablations, and unified benchmarking against defensive analytics.

## Detection Philosophy

The core deepfake detector is training-free. It scores audio using deviations from expected natural-speech behavior:

- MFCC-like cepstral temporal consistency
- pitch and micro-prosody stability
- formant proxy smoothness
- phase coherence
- spectral flux irregularity
- pause and breathing structure proxies
- vocoder artifact indicators

An optional learned head can be added later without changing the explainable feature interface.

This design choice is important for the project:

- it allows immediate use without training data
- it exposes per-signal reasoning rather than opaque logits
- it supports evaluation on unseen synthesis families more naturally than a tightly dataset-fitted classifier

## Current Status

This bootstrap provides:

- a working Python package and CLI
- synthetic-friendly attack and defense APIs
- training-free forensics implementation
- evaluation helpers, runtime probe, streaming defense, and dataset manifest tooling
- paper and results scaffolding
- tests covering core logic

In practice, the repository is no longer an empty scaffold. It already contains:

- complete CLI coverage for the main workflows
- committed synthetic experiment and ablation exports
- public-corpus benchmarking support
- authentic-speech real-data evaluation using extracted `LibriSpeech`
- paper bundle generation for reproducible manuscript assets

## Current Results

The repository now includes committed synthetic and public-data artifacts:

- synthetic research bundle under `results/research_bundle/`
- public `LibriSpeech` benchmark exports under `results/detection_benchmarks/librispeech_public/`
- small real-data smoke benchmark under `results/detection_benchmarks/librispeech_devclean_sample/`

Current measured snapshots:

- streaming defense latency at `16 kHz`, `20 ms` chunks: `0.886 ms` mean, below the `< 10 ms` target
- offline defense latency on the same demo clip: `13.428 ms` mean
- synthetic SIREN-WAVE mean risk: `0.3423`
- synthetic ultrasonic mean risk: `0.3454`
- `LibriSpeech` authentic-only public benchmark count: `3286` clips

Important interpretation note:

- `LibriSpeech` is an authentic-speech corpus, so it is useful for baseline stability and false-positive analysis
- because the current `LibriSpeech` runs do not contain synthetic positives, `AUC` is not informative there and appears as `0.0` in the exported summaries
- real deepfake discrimination claims should be attached later to `FakeAVCeleb` or another labeled synthetic-vs-authentic corpus

## What Is Still External

The remaining gaps are not mostly code gaps. They are external-data and final-evaluation gaps:

- a labeled deepfake corpus such as `FakeAVCeleb` for real synthetic-vs-authentic discrimination claims
- optional full `ASVspoof2019` corpus runs for broader anti-spoof validation
- optional learned baselines such as verified `LCNN` and `RawNet2` integrations
- final manuscript tables based on those external corpora

That means the engineering side of the research system is largely in place. The main remaining work is data-driven validation.

## Clean Repo State

The repository is kept in a small-footprint state by default:

- extracted `LibriSpeech` data is kept because it is already in use
- temporary download archives are not kept in the repo tree
- intermediate caches and generated scratch outputs are removed
- `.codex` is ignored in git because it is local environment state

## CLI

```bash
PYTHONPATH=src python -m siren.cli demo
PYTHONPATH=src python -m siren.cli runtime
PYTHONPATH=src python -m siren.cli datasets
PYTHONPATH=src python -m siren.cli stream-demo --chunk-ms 200
PYTHONPATH=src python -m siren.cli experiments --output-dir /tmp/siren_experiments
PYTHONPATH=src python -m siren.cli benchmark-corpus --dataset-root data/asvspoof2019 --output-dir /tmp/siren_corpus
PYTHONPATH=src python -m siren.cli benchmark-corpus --dataset-name asvspoof2019 --dataset-root data/asvspoof2019/LA/ASVspoof2019_LA_dev/flac --protocol data/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
PYTHONPATH=src python -m siren.cli modules
PYTHONPATH=src python -m siren.cli presets
PYTHONPATH=src python -m siren.cli generate-dataset --output-root data/synthetic_siren --manifest data/synthetic_siren/manifest.csv
PYTHONPATH=src python -m siren.cli train-demo
PYTHONPATH=src python -m siren.cli latency-demo --chunk-ms 20 --runs 5
PYTHONPATH=src python -m siren.cli ablation --output-dir results/ablation/generated
PYTHONPATH=src python -m siren.cli research-bundle --output-dir results/research_bundle
```

Recommended workflow for a quick end-to-end pass:

```bash
PYTHONPATH=src python -m siren.cli demo
PYTHONPATH=src python -m siren.cli latency-demo --chunk-ms 20 --runs 5
PYTHONPATH=src python -m siren.cli ablation --output-dir /tmp/siren_ablation
PYTHONPATH=src python -m siren.cli benchmark-corpus --manifest data/manifests/librispeech_devclean_sample.csv --dataset-name librispeech --output-dir /tmp/siren_real_smoke
PYTHONPATH=src python -m siren.cli research-bundle --output-dir results/research_bundle
```

## Reproducible Experiments

The synthetic experiment runner currently provides:

- seed-controlled evaluation for `42`, `43`, `44`
- SIREN-WAVE and safe proxy baselines
- exported per-run CSV rows and JSON summaries
- confidence interval summaries for quick reporting

It is intended as the framework layer that later plugs into full ASVspoof2019, FakeAVCeleb, and LibriSpeech evaluation jobs.

Outputs produced by the experiment layer include:

- row-wise measurements for each seed and module
- aggregated confidence intervals
- ablation comparisons against the full SIREN-WAVE configuration
- paper-ready LaTeX table generation

## Synthetic Dataset Generator

SIREN now includes a local synthetic corpus generator for rapid offline benchmarking:

- authentic speech-like seed samples
- synthetic variants using music-hidden, spoofing, and ultrasonic simulators
- manifest export for direct use with `benchmark-corpus`
- useful for smoke tests and CI-style validation when full corpora are unavailable

## Presets

Preset profiles are available for:

- `default`
- `low_vram_gtx1650`
- `realtime_guard`

Dataset benchmark preset templates are included for:

- `asvspoof2019_dev`
- `fakeavceleb_eval`
- `librispeech_baseline`

## Research Bundle

The `research-bundle` command creates a compact synthetic paper-prep package:

- experiment summaries
- ablation summaries
- CSV and JSON exports
- Markdown summaries
- LaTeX tables for direct paper inclusion

Committed outputs live under:

- `results/research_bundle/experiments/`
- `results/research_bundle/ablation/`
- `results/research_bundle/paper_tables/`

## Small-Footprint Mode

If you want to keep disk usage low and still progress the research:

- use the existing `LibriSpeech` subsets already present in `data/librispeech/`
- keep temporary benchmark exports in `/tmp` or another untracked directory
- run synthetic experiments and ablations through `research-bundle`
- use `data/manifests/librispeech_devclean_sample.csv` for quick real-data smoke tests
- add larger external corpora such as `FakeAVCeleb` only when you are ready to run final discrimination benchmarks

This mode is the recommended way to continue working when:

- you want to avoid large dataset downloads
- you are iterating on code and infrastructure
- you want reproducible paper artifacts without growing the repo footprint

## Corpus Benchmarking

SIREN now includes a local corpus benchmark path for real audio trees or CSV manifests:

- recursive audio scanning for `wav`, `flac`, `mp3`, `ogg`
- simple label inference from filenames and folders
- manifest-driven evaluation when you want explicit labels
- summary metrics for training-free detection and proxy baselines
- dataset-aware loaders for ASVspoof-style protocols, LibriSpeech trees, and FakeAVCeleb metadata

Expected manifest columns:

```csv
path,label,dataset,split
data/myset/a.wav,authentic,myset,dev
data/myset/b_fake.wav,synthetic,myset,dev
```

ASVspoof-style protocol lines are also supported in the form:

```text
LA_0001 LA_T_1000137 - A01 spoof
LA_0002 LA_T_1000138 - - bonafide
```

## Limitations

The repository currently does not claim all of the following are complete:

- full real-corpus deepfake benchmarking on `FakeAVCeleb`
- full official `ASVspoof2019` benchmark coverage
- production-grade attack deployment tooling
- verified learned baseline integrations with committed checkpoints

These are deliberate boundaries around current scope, available data, and safe release practices.

## Research Positioning

SIREN should be understood as:

- a strong open-source adversarial audio research base
- a complete reproducibility and benchmarking framework
- a training-free forensic analysis platform
- a paper-preparation workspace

It should not be understood as:

- a consumer product
- a stealth deployment toolkit
- a finished claim of exhaustive real-world benchmark superiority


