# Phase 1 — PCB Dataset Research & Recommendations

**Date:** 2026-09-01
**Status:** APPROVED
**Author:** Phase 1 planning research

---

## 1. Research Objective

Identify a suitable free/public PCB defect detection dataset for the AK Visual Intelligence Platform MVP, satisfying these hard constraints:

- PCB/electronics domain (primary demonstration)
- Models trained from scratch — no pretrained weights
- Support for **known-defect classification** with localization
- Support for **unknown-anomaly detection** strategy
- Free/open license for research use
- Practical for a student prototype with limited compute

---

## 2. Dataset Comparison

### 2.1 DeepPCB

| Attribute | Details |
|---|---|
| **Source** | [GitHub: tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB) |
| **Paper** | "Online PCB Defect Detector on a New PCB Defect Dataset" (Tang et al., 2019) |
| **Image Count** | 1,500 image pairs (defect image + template image) = 3,000 total images |
| **Resolution** | 640×640 pixels, grayscale-like (thresholded/binarized PCB scans) |
| **Format** | `.jpg` images |
| **Defect Classes** | 6 types: **open circuit**, **short circuit**, **mouse bite**, **spur**, **spurious copper**, **pin hole** |
| **Annotations** | Custom `.txt` format: each line = `x1 y1 x2 y2 type` (1-6). The parser accepts both space separated and comma-separated for robustness. |
| **Localization** | ✅ Bounding box annotations per defect instance |
| **Template/Reference** | ✅ Every defect image has a paired defect-free template image (same board region) |
| **License** | MIT License (explicitly stated in the repository) |
| **Availability** | ✅ Available — hosted directly in the GitHub repo (`PCBData/` directory) as the primary practical download source. Baidu Cloud links are available as a fallback. |
| **Known Limitations** | • Binarized/thresholded images (not natural color photos) — simpler visual domain.<br>• 1,500 pairs is small by deep learning standards.<br>• Class distribution may be uneven. |

**Strengths:**
- Template-paired images are ideal for reference-based anomaly detection
- Clean bounding box annotations ready for object detection
- 6 distinct, well-defined defect classes suitable for classification
- Compact size makes training from scratch feasible on limited hardware
- Well-documented in published paper

**Limitations:**
- Binary/thresholded images — doesn't teach the model to handle real-world color PCB photos
- Relatively small dataset for training deep networks from scratch

---

### 2.2 PKU-Market-PCB (Peking University PCB Dataset)

| Attribute | Details |
|---|---|
| **Source** | [GitHub: Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB](https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB) (also known as PCB_DATASET) |
| **Image Count** | 1,386 base images yielding 10,668 crops |
| **Resolution** | Various; some high-resolution scanned images |
| **Defect Classes** | 6 types: **missing hole**, **mouse bite**, **open circuit**, **short circuit**, **spur**, **spurious copper** |
| **Annotations** | XML format (Pascal VOC style bounding boxes) |
| **Localization** | ✅ Bounding box annotations (Pascal VOC XML) |
| **Template/Reference** | ✅ Template images available for each board type |
| **License** | Not explicitly stated; academic / research use implied |
| **Availability** | ✅ Available on GitHub and via Kaggle mirrors |
| **Known Limitations** | • Heavily augmented to reach the ~10k count.<br>• Original image diversity lower than total count implies. |

**Strengths:**
- Pascal VOC annotation format is well-documented and widely supported
- Similar defect taxonomy to DeepPCB (good for cross-validation of defect categories)
- Higher-resolution images than DeepPCB

**Limitations:**
- Inflated count through augmentation — effective unique image count is lower
- Less established direct template-differencing pipeline compared to DeepPCB pair mechanism

---

### 2.3 MVTec Anomaly Detection Dataset (MVTec AD)

| Attribute | Details |
|---|---|
| **Source** | [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **Relevant Categories** | No dedicated PCB category. Closest electronic-related: none directly PCB. |
| **Image Count** | ~5,354 total across all 15 categories |
| **Annotations** | Pixel-level anomaly segmentation masks + image-level labels (good/defective) |
| **Localization** | ✅ Pixel-level masks for anomalous regions |
| **License** | Free for research/non-commercial use (CC BY-NC-SA 4.0) |
| **Availability** | ✅ Available via official website download |
| **Known Limitations** | • No PCB category — not directly applicable to our domain. • License restricts commercial use. |

**Strengths:**
- Gold standard for anomaly detection research
- Pixel-level anomaly masks are high quality

**Limitations:**
- **Not a PCB dataset** — would only serve as a methodology reference, not as training data
- Non-commercial license

---

### 2.4 HRIPCB (Printed Circuit Board Defect Detection Dataset)

| Attribute | Details |
|---|---|
| **Source** | Associated with paper "TDD-net" / various Kaggle mirrors |
| **Image Count** | 1,386 high-resolution synthetic defect images, often cropped to 10,668 sub-images |
| **Defect Classes** | 6 types (same as PKU-Market-PCB) |
| **Annotations** | Bounding boxes |
| **License** | Academic use |
| **Availability** | Available on Kaggle mirrors with varying quality |

**Assessment:** Synonymous with PKU-Market-PCB structure. Not recommended as a separate independent source.

---

## 3. Comparative Summary

| Criterion | DeepPCB | PKU-Market-PCB | MVTec AD |
|---|---|---|---|
| PCB Domain | ✅ | ✅ | ❌ |
| Defect Classes | 6 | 6 | N/A |
| Image Pairs (with template) | ✅ 1,500 pairs | ✅ (via base images) | ❌ |
| Bounding Box Annotations | ✅ (Custom txt) | ✅ (VOC XML) | ❌ (pixel masks) |
| Train-from-Scratch Feasibility | ✅ Good | ⚠️ Augmented | N/A |
| License | MIT | Research (implicit) | CC BY-NC-SA 4.0 |
| Download Reliability | ✅ GitHub (primary) | ✅ GitHub / Kaggle | ✅ Official site |

---

## 4. Recommendations & Decisions (APPROVED)

### 4.1 Primary Dataset: **DeepPCB**

**Rationale:**
- Best alignment with project requirements: PCB domain, 6 defect classes, bounding box localization, template pairing
- Template-paired images directly enable our reference-board comparison strategy for anomaly detection
- 1,500 pairs is manageable for training from scratch on limited compute
- Well-established benchmark in PCB defect detection literature
- Bounding box annotations can be cleanly parsed and converted to our normalized format

**Download plan:** Primary verified source is directly from the GitHub repository (`PCBData/` directory). Baidu Cloud links provided in the paper/repo serve only as a fallback.

### 4.2 MVP Defect Classes (6 classes)

Derived from DeepPCB's taxonomy, aligning with real PCB manufacturing defect categories:

| Class ID | Defect Name | Description | Category |
|---|---|---|---|
| 0 | `open_circuit` | Break in a copper trace that should be connected | Connectivity |
| 1 | `short_circuit` | Unintended copper connection between traces | Connectivity |
| 2 | `mouse_bite` | Irregular, nibbled-looking edge on a trace | Structural |
| 3 | `spur` | Small, unwanted protrusion from a trace | Structural |
| 4 | `spurious_copper` | Extra copper where there should be none | Excess material |
| 5 | `pin_hole` | Small hole/void in the copper where there shouldn't be one | Void |

### 4.3 Unknown-Anomaly Detection Strategy

**Note:** Complex methods like Deep SVDD are explicitly removed from the MVP and project scope for simplicity and compute efficiency.

**Implementation Sequence:**
- **Phase A — Template differencing baseline:** Direct raw pixel/structural comparison between the input image and its corresponding golden template.
- **Phase B — Small convolutional autoencoder trained from scratch:** Train a lightweight autoencoder strictly on the normal, defect-free template images.
- **Phase C — Combine both approaches:** Merge outputs from the differencing and the autoencoder for robust unknown anomaly detection.

### 4.4 Annotation Format & Parsing

**Standardized internal format: Normalized YOLO-style bounding boxes**

```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates normalized to [0, 1] relative to image dimensions.

**Parsing DeepPCB format:**
DeepPCB annotations are formally documented as comma-separated (`x1,y1,x2,y2,type`), but actually found space-separated (`x1 y1 x2 y2 type`) in the main dataset repository.
Our parser leverages regex (`[\s,]+`) to cleanly support both formats for robustness.

- The `type` is 1-indexed (1 through 6).
- Conversion to 0-indexed: `class_id = type - 1`

Formula for normalized coordinates:
- `x_center = (x1 + x2) / 2 / image_width`
- `y_center = (y1 + y2) / 2 / image_height`
- `width = (x2 - x1) / image_width`
- `height = (y2 - y1) / image_height`

### 4.5 Train / Validation / Test Split

**Split ratio: 70 / 15 / 15**

| Split | Count (of 1,500 pairs) | Purpose |
|---|---|---|
| **Train** | 1,050 pairs | Model training (known-defect detector + autoencoder) |
| **Validation** | 225 pairs | Hyperparameter tuning, early stopping |
| **Test** | 225 pairs | Final evaluation only — never used during training |

**Split strategy:**
- **Stratified by defect class** — ensure each class is proportionally represented
- **No image leaks** — a template and its corresponding defect image go to the same split
- **Deterministic** — use a fixed random seed (42)

### 4.6 Golden / Reference-Board Strategy

DeepPCB's format natively supports a reference-board strategy:
- Each defect image has a **1:1 paired template** (defect-free golden reference)
- Templates are pre-aligned (same region/scale/orientation)
- At inference time, the system leverages the provided matching template. Complex live alignment/registration is deferred beyond the MVP.

---

## 5. Next Steps

1. **Create annotation parsing test** using a real DeepPCB annotation string to verify `x1,y1,x2,y2,type` correctly yields `class_id = type - 1` and precise normalized coordinates.
2. (Deferred) Download script targeting the `PCBData/` GitHub directory.
3. (Deferred) Code implementation for `PCBDataset` loader and splits.
