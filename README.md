# AK Visual Intelligence Platform

A multi-domain visual inspection and anomaly detection system built from scratch.

## Overview

The AK Visual Intelligence Platform provides AI-powered visual inspection for manufacturing quality control. It is designed as a **modular, multi-domain system** — PCB/electronics inspection is the first supported domain, with the architecture ready for additional product types.

### Key Capabilities (Planned)

| Module           | Purpose                                          |
|------------------|--------------------------------------------------|
| Preprocessing    | Image loading, augmentation, domain transforms   |
| Models           | Custom CNN/detection models (no pretrained weights) |
| Anomaly Detection| Detect out-of-distribution and novel defects     |
| Risk & Confidence| Severity scoring, confidence calibration         |
| Explainability   | Visual explanations (Grad-CAM, saliency maps)    |
| API              | REST endpoints for inference and management      |
| Dashboard        | Interactive UI for results and analytics          |

## Project Structure

```
core/          Core AI components (domain-agnostic)
domains/pcb/       PCB-specific inspection logic
api/               FastAPI REST API
dashboard/         Streamlit dashboard
config/            YAML configuration
data/              Datasets (raw/ is gitignored)
tests/             Test suite
docs/              Documentation
scripts/           Utility scripts
```

## Getting Started

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest
```

## Status

**Phase 1: Data Pipeline & Preprocessing** — Project structure created, API health endpoint active, and human-in-the-loop base logic implemented. The DeepPCB dataset has been acquired and bounding box annotation parsing fully verified. Dataloader implementation follows.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the full roadmap.
