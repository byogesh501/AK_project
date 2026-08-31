# AK Visual Intelligence Platform — Project Status

## Overview & Current State
- **Vision:** Multi-domain AI-powered visual inspection platform for manufacturing quality control (starting with PCB/electronics).
- **Core Policy:** All inspection models trained from scratch — zero pretrained weights.
- **Current Phase:** Phase 1 (Data Pipeline) Planning completed, Implementation pending start.
- **Latest Git Commit:** `[pending commit]` — `docs: Finalize Phase 1 planning & implement annotation parser`
- **Latest Test Suite Result:** [pending tests]
- **Next Step:** Phase 1 — Data Pipeline Implementation (download script, dataset loaders, preprocessing pipeline).

---

## Phase 0: Project Scaffolding ✅ (Completed)

### Implemented & Verified Architecture
- **Platform Core (`core/`)**: Domain-agnostic modular foundation:
  - `preprocessing/` — image transformation and augmentation interfaces
  - `models/` — model architectures and training utilities
  - `anomaly/` — unsupervised anomaly detection algorithms
  - `risk/` — defect risk assessment and scoring
  - `explainability/` — visual inspection explanations (Grad-CAM, saliency)
  - `review/` — human-in-the-loop (HITL) review and feedback collection
  - `utils/` — common utilities (logging, metrics, I/O)
- **Domain Pack (`domains/pcb/`)**: First product domain implementation:
  - `preprocessing/`, `models/`, `anomaly/`, `risk/`, `explainability/`, `config/` submodules with standard interfaces.
- **API (`api/`)**: FastAPI application structure with `/health` endpoint.
- **Dashboard (`dashboard/`)**: Streamlit inspection dashboard scaffolding.
- **Data (`data/`)**: Partitioned into `raw/` (gitignored), `processed/`, and `samples/`.
- **Packaging & Setup**: `setup.py`, `requirements.txt`, `README.md`, `config/default.yaml`.

---

## Phase 0.5: Architectural Fixes & Safeguards ✅ (Completed)

### Implemented & Verified Components

1. **Human-in-the-Loop (HITL) Review Foundation (`core/review/`):**
   - `__init__.py`: Domain-agnostic module exposing `VerdictStatus`, `Verdict`, and `FeedbackRecord`.
   - `base.py`: `VerdictStatus` enum (`ACCEPTED`, `REJECTED`, `CORRECTED`, `NEEDS_REVIEW`), `Verdict` and `FeedbackRecord` dataclasses.
   - `BaseReviewStore` and `BaseReviewPolicy` abstract base classes for verdict persistence and review routing decisions.

2. **Core Abstract Interfaces & Contracts:**
   - `core/preprocessing/base.py`: `BasePreprocessor` ABC (defines `load_image`, `preprocess`, `augment`).
   - `core/models/base.py`: `BaseModel` and `BaseTrainer` ABCs (for architecture and training loop contracts).
   - `core/anomaly/base.py`: `BaseAnomalyDetector` ABC and `AnomalyResult` dataclass (for anomaly scoring/classification).
   - `core/risk/base.py`: `BaseRiskScorer` ABC and `RiskAssessment` dataclass (for risk severity scoring).
   - `core/explainability/base.py`: `BaseExplainer` ABC and `Explanation` dataclass (for Grad-CAM/saliency interfaces).
   - `core/utils/__init__.py`: Shared utilities package (config loader to be implemented in Phase 1).

3. **Runnable FastAPI `/health` Endpoint (`api/main.py`):**
   - Implemented `/health` returning service status, timestamp, and version `0.1.0`.
   - Added `httpx` dependency for test client support.

4. **Test Suite Improvements & Validation:**
   - Removed placeholder `assert True` tests from Phase 0.
   - `tests/core/test_base_interfaces.py`: Validates abstract base classes reject direct instantiation and dataclass contracts construct correctly.
   - `tests/domains/test_pcb.py`: Parametrized import and structure checks across all PCB domain submodules.
   - `tests/api/test_main.py`: HTTP-level tests for `/health` endpoint (response 200, body schema, version field).
   - `tests/core/test_utils.py`: Skipped placeholder with explicit skip reason (awaiting Phase 1 implementation).
   - `tests/test_policy.py`: Automated guard that scans all `.py` files in `core/`, `domains/`, `api/`, and `dashboard/` for forbidden pretrained-weight patterns.
   - `tests/conftest.py`: Ensures project root is available on `sys.path` for import resolution.

5. **Pretrained-Weight Policy & Automated Enforcement:**
   - **Policy:** All inspection models must be trained from scratch. External pretrained weights (e.g., ImageNet, transformers checkpoints) are strictly prohibited.
   - **Enforcement:** `tests/test_policy.py` is an automated guard that scans all project Python source files under `core/`, `domains/`, `api/`, and `dashboard/` for forbidden patterns: `weights=`, `pretrained=`, `.from_pretrained(`, and `torch.hub.load(`.
   - **Bypass mechanism:** Legitimate non-model uses (e.g., describing what is forbidden in a docstring) may opt out on a per-line basis using `# noqa: allow-pretrained`.
   - **Verification:** Guard was confirmed to fail on a temporary test file containing `torch.hub.load()`, then pass after removal. Guard is integrated into CI-equivalent pytest runs.

6. **Repository Hygiene & Safeguards:**
   - `.gitattributes`: Line-ending normalization (LF) across all source, config, and markdown files. Explicit binary tags for model weight formats (`*.pt`, `*.pth`, `*.onnx`) and image formats (`*.png`, `*.jpg`, `*.jpeg`).
   - `.gitignore`: Comprehensive coverage of Python caches (`__pycache__`), virtual environments (`.venv`, `venv`), raw datasets (`data/raw/`), model binaries, IDE configs (`.vscode/`, `.idea/`), and Claude Code local session artifacts (`.claude/settings.local.json`, `.claude/tasks/`, `.claude/output/`).

7. **Architecture Documentation (`docs/architecture.md`):**
   - Documents project guiding principles, layer diagram, and module responsibilities.
   - Describes platform core vs domain-specific pack separation.
   - Outlines the HITL review and feedback flow with a diagram.
   - Provides a step-by-step onboarding guide for future product domains.

---

## Roadmap & Upcoming Phases

### Phase 1: Data Pipeline & Preprocessing (Planning Complete)
- **Status:** Planning complete. Research documented in `docs/phase1_dataset_research.md` (APPROVED).
- **Decisions:**
  - **Dataset:** DeepPCB (1,500 template-paired images, 6 defect classes). Download from GitHub `PCBData/` directly.
  - **Annotation Format:** DeepPCB `x1,y1,x2,y2,type` converted to Normalized YOLO, with `class_id = type - 1`.
  - **Anomaly Strategy:** Three phases: A) Template differencing baseline, B) Small Conv-AE trained from scratch, C) Combined approach. (Deep SVDD removed).
  - **Splits:** 70/15/15 paired split, using golden-template strategy.
- **Next Immediate Actions:**
  - Download and verify DeepPCB dataset.
  - Implement concrete dataset loaders and augmentation pipelines in `core/preprocessing/` and `domains/pcb/preprocessing/`.
  - Build train/val/test splits generation scripts.

### Phase 2: Model Architecture (Planned)
- Custom CNN / defect detection architectures trained from scratch.
- Training loop, loss functions, checkpointing, and evaluation metrics.

### Phase 3: Anomaly Detection & Risk Scoring (Planned)
- Phase A: Template differencing.
- Phase B: Small convolutional autoencoder.
- Phase C: Combined logic.
- Multi-factor risk scoring and confidence calibration.

### Phase 4: Explainability (Planned)
- Saliency maps, Grad-CAM, and inspection report generator.

### Phase 5: Human-in-the-Loop & Feedback Pipeline (Planned)
- Concrete `ReviewStore` and confidence-based routing policies.
- Feedback export for model retraining and drift tracking.

### Phase 6: API & Dashboard Integration (Planned)
- Full REST endpoints in FastAPI and interactive Streamlit UI.

### Phase 7: Multi-Domain Expansion (Planned)
- Multi-domain registration framework and second domain demonstration.
