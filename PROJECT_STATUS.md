# AK Visual Intelligence Platform — Project Status

## Overview & Current State
- **Vision:** Multi-domain AI-powered visual inspection platform for manufacturing quality control (starting with PCB/electronics).
- **Core Policy:** All inspection models trained from scratch — zero pretrained weights.
- **Current Phase:** Phase 1 (Data Pipeline) Implementation started.
- **Latest Action:** Implemented remaining bounding box strict validation and synchronized documentation.
- **Latest Test Suite Result:** 30 passed, 1 skipped (0 failed).
- **Next Step:** Implement concrete dataset loaders and dynamic 70/15/15 generation scripts.

---

## Phase 0 & 0.5: Project Scaffolding & Architecure ✅ (Completed)
- **Core (`core/`) & Domains (`domains/pcb/`)**: Added module stubs for preprocessing, models, anomaly, risk, explainability, safety. Base abstract classes (`BasePreprocessor`, `BaseModel`, `BaseAnomalyDetector`, `BaseRiskScorer`, `BaseExplainer`) rejecting direct instantiation.
- **Review Framework**: Domain-agnostic HITL scaffolding (`core/review/`) with `VerdictStatus`, `Verdict`, and abstract store/policy classes.
- **Quality/Safety**: Added `api/main.py` health endpoint. Strict "no pretrained weights" enforced by `tests/test_policy.py` failing on forbidden patterns.
- **Config & Git Tools**: Comprehensive `.gitignore` and `.gitattributes` avoiding large binaries and syncing LF line endings. Architecture decisions documented in `docs/architecture.md`.

---

## Roadmap & Upcoming Phases

### Phase 1: Data Pipeline & Preprocessing (In Progress)
- **Status:** Planning complete. Research documented in `docs/phase1_dataset_research.md` (APPROVED).
- **Completed Actions:**
  - Acquired a small sample (2 pairs) from the `PCBData/` GitHub repository directly.
  - Verified directory structure, paired mechanism, and format.
  - **CRITICAL DATASET FINDING:** The documentation claims comma-separated values (`x1,y1,x2,y2,type`), but the real annotations are *space-separated* (`x1 y1 x2 y2 type`).
  - Implemented `parse_deeppcb_annotation` (handling spaces & commas) + YOLO converter + regression tests.
  - Updated parser to rigorously validate defect types (1-6) while skipping redundant `verify_sample.py` scripts.
  - **FULL ACQUISITION:** Downloaded the full DeepPCB dataset (cloned from GitHub `tangsanli5201/DeepPCB` and isolated `PCBData/`). No preferred mirrors were accessible without session tokens.
  - **VERIFICATION:** Verified 1,500 perfect image/template pairs with valid annotations. Validated all images are 640x640 cleanly opened via PIL. Verified class distributions (1942 Open, 1506 Short, 1965 Mouse bite, 1625 Spur, 1474 Copper, 1501 Pin hole).
  - Recorded provenance in `data/raw/deeppcb/README.md`.
- **Decisions:**
  - **Dataset:** DeepPCB (1,500 template-paired images, 6 defect classes). Download from GitHub `PCBData/` directly.
  - **Annotation Format:** DeepPCB `x1 y1 x2 y2 type` converted to Normalized YOLO, with `class_id = type - 1`.
  - **Anomaly Strategy:** Three phases: A) Template differencing baseline, B) Small Conv-AE trained from scratch, C) Combined approach. (Deep SVDD removed).
  - **Splits:** 70/15/15 paired split, using golden-template strategy.
- **Next Immediate Actions:**
  - Implement concrete dataset loaders and augmentation pipelines in `core/preprocessing/` and `domains/pcb/preprocessing/`.
  - Build train/val/test 70/15/15 generation scripts using the 1500 available pairs natively.

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
