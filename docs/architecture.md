# Architecture Overview

## Guiding Principles

1. **No pretrained models/weights.** Every model is defined and trained from scratch. (Enforced by automated guard tests).
2. **Platform / Domain separation.** Reusable AI logic lives in `core/`; product-specific logic lives in `domains/<name>/`.
3. **Human-in-the-loop by design.** Every prediction can be reviewed, corrected, and fed back into training.
4. **Minimal dependencies.** Only add what the current phase needs.

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    API  /  Dashboard                    │  ← user-facing
├─────────────────────────────────────────────────────────┤
│                  Human Review (HITL)                    │  ← verdicts, corrections
├──────────────────────┬──────────────────────────────────┤
│   Domain Packs       │      Platform Core               │
│   domains/pcb/       │      core/                       │
│     preprocessing    │        preprocessing  (base)     │
│     models           │        models         (base)     │
│     anomaly          │        anomaly        (base)     │
│     risk             │        risk           (base)     │
│     explainability   │        explainability (base)     │
│     config           │        review         (HITL)     │
│                      │        utils                     │
└──────────────────────┴──────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Location | Purpose |
|---|---|---|
| **Preprocessing** | `core/preprocessing/` | Base interface for image loading, augmentation, and transforms. Domain packs override with product-specific pipelines. |
| **Models** | `core/models/` | Base model and trainer ABCs. Defines the contract for forward/predict/save/load. All models are built from scratch. |
| **Anomaly Detection** | `core/anomaly/` | Base anomaly scoring and detection interface. Domains plug in product-specific detectors. |
| **Risk & Confidence** | `core/risk/` | Base risk scoring and confidence calibration. Domains provide severity maps and thresholds. |
| **Explainability** | `core/explainability/` | Base visual explanation interface (Grad-CAM, saliency). Domains customize overlays and reports. |
| **Review (HITL)** | `core/review/` | Human-in-the-loop: verdicts (accept/reject/correct), feedback storage, review routing policy. |
| **Utils** | `core/utils/` | Shared helpers — config loading, logging, metrics, file I/O. |

---

## Common Platform vs Domain-Specific Packs

**Platform (`core/`)** defines abstract base classes (ABCs) and shared utilities. It contains zero product-specific knowledge.

**Domain packs (`domains/<name>/`)** implement the platform interfaces for a specific product type. Each pack mirrors the platform sub-structure:

```
domains/<name>/
  preprocessing/   → subclasses BasePreprocessor
  models/          → subclasses BaseModel, BaseTrainer
  anomaly/         → subclasses BaseAnomalyDetector
  risk/            → subclasses BaseRiskScorer
  explainability/  → subclasses BaseExplainer
  config/          → domain constants, defect categories, thresholds
```

The first domain pack is `domains/pcb/` (PCB / electronics inspection).

---

## Policy Safeguards

To prevent accidental deviation from project requirements, a strict safeguard runs as part of the test suite (`tests/test_policy.py`). 

The test automatically scans **every `.py` file in the project root** (excluding `tests/`, `__pycache__/`, `.venv/`, `data/`, and other caches) for disallowed keywords that attempt to download or load externally pretrained weights:
- `weights=`
- `pretrained=`
- `.from_pretrained(`
- `torch.hub.load(`

If a legitimate non-model operation requires these strings (for example, string parsing), suffix the line with `# noqa: allow-pretrained` to bypass the guard safely.

---

## Human-in-the-Loop (HITL) & Feedback Flow

```
  Model Prediction
        │
        ▼
  ┌─────────────┐     confidence < threshold?
  │ Review       │◄── or anomaly score high?
  │ Policy       │     or random audit sample?
  └──────┬──────┘
         │ yes
         ▼
  ┌─────────────┐
  │ Human       │  Operator reviews the prediction
  │ Review UI   │  and submits a Verdict:
  └──────┬──────┘    ACCEPTED / REJECTED / CORRECTED
         │
         ▼
  ┌─────────────┐
  │ Review      │  Stores Verdict + optional corrected label
  │ Store       │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │ Feedback    │  Aggregated corrections exported as
  │ Export      │  FeedbackRecord → future retraining data
  └─────────────┘
```

**Key types** (defined in `core/review/base.py`):
- `VerdictStatus` — ACCEPTED, REJECTED, CORRECTED, NEEDS_REVIEW
- `Verdict` — one human judgment on one prediction
- `FeedbackRecord` — batch of verdicts for a domain, ready for retraining
- `BaseReviewStore` — ABC for persisting verdicts
- `BaseReviewPolicy` — ABC for deciding which predictions need human review

---

## Adding a New Product Domain

To add a new domain (e.g., `textiles`):

1. **Create the domain pack:**
   ```
   domains/textiles/
     __init__.py
     preprocessing/__init__.py   # subclass BasePreprocessor
     models/__init__.py          # subclass BaseModel
     anomaly/__init__.py         # subclass BaseAnomalyDetector
     risk/__init__.py            # subclass BaseRiskScorer
     explainability/__init__.py  # subclass BaseExplainer
     config/__init__.py          # defect categories, thresholds
   ```

2. **Implement each interface** by subclassing the corresponding `core/*/base.py` ABC.

3. **Add domain config** to `config/default.yaml` or create `config/textiles.yaml`.

4. **Add tests** under `tests/domains/test_textiles.py`.

5. **Register the domain** in the API/dashboard so it can be selected at runtime.

The platform core does not need to change — it operates on the base interfaces.
