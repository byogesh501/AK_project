## DeepPCB Dataset

**Provenance & Verification Record**

- **Project:** AK Visual Intelligence Platform (Phase 1 MVP)
- **Source URL:** `https://github.com/tangsanli5201/DeepPCB`
- **Actual Source Used:** Cloned via `git clone https://github.com/tangsanli5201/DeepPCB.git` (GitHub fallback since mirrors were unavailable). `PCBData/` directory isolated.
- **Download/Access Date:** 2026-09-01
- **License/Usage Terms:** MIT License (in repo root) / specifically noted for research purpose in README.

### Full Dataset Verification Status
The full `PCBData/` directory was extracted and verified.

- **Expected directory/file structure:** Verified. 11 group directories (e.g. `group00041`, `group12000`, etc.) containing the defect images and paired annotations.
- **Defect/Template pairing:** Verified. Found **1500 perfectly matched pairs** of `_test.jpg` (defect), `_temp.jpg` (golden template), and `.txt` annotation files across the subdirectories.
- **Image dimensions:** Verified. All 1500 pairs are exactly 640x640 pixels.
- **Corruption check:** Verified. All loaded cleanly via PIL with no apparent file corruption.
- **Annotation format:** Verified. DeepPCB specifies comma-separated formats, but real data is consistently space-separated (`x1 y1 x2 y2 type`). Our `parse_deeppcb_annotation` parser correctly handles this.
- **Annotation classes mapping:** Class distribution over the 1500 pairs matches expectations:
  - 1 (Open circuit): 1942
  - 2 (Short circuit): 1506
  - 3 (Mouse bite): 1965
  - 4 (Spur): 1625
  - 5 (Spurious copper): 1474
  - 6 (Pin hole): 1501
  Total defects mapped: ~10,013 across 1500 images. Distribution is well-balanced.

### Discrepancies Noted
- The original authors provided `test.txt` (499 pairs) and `trainval.txt` (999 pairs). Totaling 1498 pairs.
- Scanning the raw directory yields an actual count of 1500 completely valid pairs.
- **Action:** We will generate our own 70/15/15 deterministic splits directly from the verified 1500 pairs in later stages rather than strictly relying on `trainval.txt` / `test.txt`, keeping data maximal.

*Code artifact: `verify_dataset.py` was executed to rigorously validate these constraints before progressing.*
