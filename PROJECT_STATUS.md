# AK Visual Intelligence Platform — Project Status

## Vision

An AI-powered visual inspection platform for manufacturing quality control.
The system detects defects, scores risk, and provides explainable results
across multiple product domains. **PCB/electronics** is the first domain.

All models are trained from scratch — no pretrained weights.

---

## Phase 0: Project Scaffolding ✅ (Completed 2026-08-30)

### What was created

**Platform core (`core/`)** — domain-agnostic modules:
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
- `core/` holds reusable code; `domains/` holds product-specific code
- New domains are added as sibling packages under `domains/` (e.g., `domains/textiles/`)
- No pretrained models or weights — everything built from scratch
- Minimal dependencies — only what's needed at each phase

---

## Phase 0.5: Architectural Fixes ✅ (Completed 2026-08-30)

### What was added/changed

**Human-in-the-loop review (`core/review/`):**
- `base.py` — `VerdictStatus`, `Verdict`, `FeedbackRecord` data classes;
  `BaseReviewStore` and `BaseReviewPolicy` ABCs for verdict persistence
  and review routing

**Base interfaces/contracts added:**
- `core/preprocessing/base.py` — `BasePreprocessor` ABC
- `core/models/base.py` — `BaseModel` and `BaseTrainer` ABCs
- `core/anomaly/base.py` — `BaseAnomalyDetector` ABC, `AnomalyResult` dataclass
- `core/risk/base.py` — `BaseRiskScorer` ABC, `RiskAssessment` dataclass
- `core/explainability/base.py` — `BaseExplainer` ABC, `Explanation` dataclass

**API fixed:**
- `api/main.py` — minimal FastAPI app with `/health` endpoint (runnable)

**Tests replaced:**
- Removed `assert True` placeholder tests
- `tests/core/test_base_interfaces.py` — verifies all ABCs reject direct
  instantiation; tests data class construction for `Verdict`, `AnomalyResult`,
  `RiskAssessment`, `FeedbackRecord`
- `tests/domains/test_pcb.py` — parametrized import test for all PCB submodules
- `tests/api/test_main.py` — tests `/health` endpoint (status, body, version)
- `tests/core/test_utils.py` — skipped placeholder (awaiting Phase 1)
- `tests/conftest.py` — adds project root to sys.path for test imports

**Documentation:**
- `docs/architecture.md` — platform vs domain packs, module responsibilities,
  HITL/feedback flow diagram, how to add a new domain

**Configuration:**
- `config/default.yaml` — added `review` section (auto_review_threshold, random_audit_rate)

**Policy:**
- Explicit "no pretrained models/weights" policy in `core/__init__.py`,
  `core/models/__init__.py`, `requirements.txt`, and `docs/architecture.md`
- `torchvision` retained for transforms/utilities only (noted in requirements.txt)
- Added `httpx` dependency (required by FastAPI TestClient)

---

## Phase 1: Data Pipeline & Preprocessing (Next)

- [ ] Implement config loader (`core/utils/`)
- [ ] Build image loading utilities (`core/preprocessing/`)
- [ ] Add PCB-specific preprocessing (`domains/pcb/preprocessing/`)
- [ ] Create sample data loader for development
- [ ] Add data augmentation transforms
- [ ] Write tests for preprocessing pipeline

## Phase 2: Model Architecture

- [ ] Define concrete model classes (`core/models/`)
- [ ] Build PCB defect detection model from scratch (`domains/pcb/models/`)
- [ ] Implement training loop and evaluation
- [ ] Add model checkpointing and versioning

## Phase 3: Anomaly Detection & Risk Scoring

- [ ] Implement anomaly detection algorithms (`core/anomaly/`)
- [ ] Add PCB-specific anomaly detection (`domains/pcb/anomaly/`)
- [ ] Build risk scoring pipeline (`core/risk/`)
- [ ] Add confidence calibration

## Phase 4: Explainability

- [ ] Implement Grad-CAM and saliency maps (`core/explainability/`)
- [ ] Add PCB-specific visual explanations (`domains/pcb/explainability/`)
- [ ] Generate human-readable inspection reports

## Phase 5: Human-in-the-Loop & Feedback

- [ ] Implement file/DB-backed `ReviewStore` (`core/review/`)
- [ ] Build review routing policy (confidence-based + random audit)
- [ ] Add review UI to dashboard
- [ ] Create feedback export for retraining pipeline
- [ ] Track correction statistics and model drift indicators

## Phase 6: API & Dashboard

- [ ] Build FastAPI inference endpoints (`api/`)
- [ ] Create Streamlit dashboard (`dashboard/`)
- [ ] Add real-time inspection views
- [ ] Integrate HITL review UI
- [ ] Integrate all pipeline stages end-to-end

## Phase 7: Multi-Domain Expansion

- [ ] Extract domain registration pattern
- [ ] Add second product domain as proof of extensibility
- [ ] Document domain onboarding process
