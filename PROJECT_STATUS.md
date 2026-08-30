# AK Visual Intelligence Platform — Project Status

## Vision

An AI-powered visual inspection platform for manufacturing quality control.
The system detects defects, scores risk, and provides explainable results
across multiple product domains. **PCB/electronics** is the first domain.

All models are trained from scratch — no pretrained weights.

---

## Phase 0: Project Scaffolding ✅ (Completed 2026-08-30)

### What was created

**Platform core (`platform/`)** — domain-agnostic modules:
- `preprocessing/` — image loading, augmentation, transforms
- `models/` — base model architectures and training utilities
- `anomaly/` — anomaly detection algorithms
- `risk/` — risk scoring and confidence calibration
- `explainability/` — visual explanation utilities (Grad-CAM, saliency)
- `utils/` — shared helpers (logging, metrics, I/O)

**PCB domain (`domains/pcb/`)** — first domain implementation:
- `preprocessing/` — PCB-specific image processing
- `models/` — PCB defect detection models
- `anomaly/` — PCB anomaly detection
- `risk/` — PCB risk profiles
- `explainability/` — PCB-specific explanations
- `config/` — PCB constants and settings

**API (`api/`)** — FastAPI REST endpoint scaffolding

**Dashboard (`dashboard/`)** — Streamlit dashboard scaffolding

**Data (`data/`)** — `raw/` (gitignored), `processed/`, `samples/`

**Tests (`tests/`)** — mirrors source structure with placeholder tests

**Configuration (`config/default.yaml`)** — project-wide settings

**Root files:**
- `requirements.txt` — minimal dependencies
- `setup.py` — package metadata
- `README.md` — project overview and quick start
- `.gitignore` — updated for Python/AI project

### Design decisions
- `platform/` holds reusable code; `domains/` holds product-specific code
- New domains are added as sibling packages under `domains/` (e.g., `domains/textiles/`)
- No pretrained models or weights — everything built from scratch
- Minimal dependencies — only what's needed at each phase

---

## Phase 1: Data Pipeline & Preprocessing (Next)

- [ ] Implement config loader (`platform/utils/`)
- [ ] Build image loading utilities (`platform/preprocessing/`)
- [ ] Add PCB-specific preprocessing (`domains/pcb/preprocessing/`)
- [ ] Create sample data loader for development
- [ ] Add data augmentation transforms
- [ ] Write tests for preprocessing pipeline

## Phase 2: Model Architecture

- [ ] Define base model classes (`platform/models/`)
- [ ] Build PCB defect detection model from scratch (`domains/pcb/models/`)
- [ ] Implement training loop and evaluation
- [ ] Add model checkpointing and versioning

## Phase 3: Anomaly Detection & Risk Scoring

- [ ] Implement anomaly detection algorithms (`platform/anomaly/`)
- [ ] Add PCB-specific anomaly detection (`domains/pcb/anomaly/`)
- [ ] Build risk scoring pipeline (`platform/risk/`)
- [ ] Add confidence calibration

## Phase 4: Explainability

- [ ] Implement Grad-CAM and saliency maps (`platform/explainability/`)
- [ ] Add PCB-specific visual explanations (`domains/pcb/explainability/`)
- [ ] Generate human-readable inspection reports

## Phase 5: API & Dashboard

- [ ] Build FastAPI inference endpoints (`api/`)
- [ ] Create Streamlit dashboard (`dashboard/`)
- [ ] Add real-time inspection views
- [ ] Integrate all pipeline stages end-to-end

## Phase 6: Multi-Domain Expansion

- [ ] Extract domain registration pattern
- [ ] Add second product domain as proof of extensibility
- [ ] Document domain onboarding process
