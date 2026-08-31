## DeepPCB Dataset (Sample)

**Provenance & Verification Record**

- **Project:** AK Visual Intelligence Platform (Phase 1 MVP)
- **Source URL:** `https://github.com/tangsanli5201/DeepPCB`
- **Actual Source Used:** `https://raw.githubusercontent.com/tangsanli5201/DeepPCB/master/PCBData/` (GitHub Raw)
- **Download/Access Date:** 2026-09-01
- **License/Usage Terms:** MIT License (in repo root) / specifically noted for research purpose in README.

### Source Deviation Log
The Phase 1 protocol specified attempting Roboflow Universe / Ultralytics mirrors first. However, downloading a bulk zip from those platforms requires active session tokens/API keys which are unavailable in the automated environment, and pulling full zipped datasets violates the "small sample only" milestone constraint. Therefore, we explicitly fell back to the documented primary source: the `PCBData/` directory living directly in the DeepPCB GitHub repository. 

### Sample Verification Status
A small sample (2 image pairs from `group00041`) was directly downloaded and verified:

- **Expected directory/file structure:** Verified. Inside `PCBData/` are multiple groups. Specifically, `group00041/` contains `00041/` (for `.jpg` images) and `00041_not/` (for `.txt` annotations).
- **Defect/Template pairing:** Verified. `00041000_temp.jpg` pairs perfectly with `00041000_test.jpg`.
- **Annotation files:** Verified. `.txt` annotations correspond exactly to the paired `.jpg` filenames.
- **Annotation format:** Verified. Annotations are space-separated coordinate lists contrary to the comma-separated format documented by the original author. E.g., `466 441 493 470 3` not `466,441,493,470,3`.
- **Class IDs:** Verified. 1 through 6 mapping sequentially to open, short, mousebite, spur, copper, pin-hole. (The sample annotations show valid types like 1, 2, 3, 4, 5, 6).
- **Actual sample pair count:** 2 pairs downloaded successfully.
- **Corruption check:** Verified. All downloaded `.jpg` images open successfully in PIL with the expected (640x640) resolution and no corruption.

### Notes for Implementation
- **CRITICAL DATASET ANOMALY:** The DeepPCB repo `README.md` explicitly claims the annotation format is `x1,y1,x2,y2,type` (comma-separated with no spaces). However, reading the actual `.txt` files directly from the repository reveals they are heavily formatting with spaces: `x1 y1 x2 y2 type` (space-separated). 
- *Action Required:* The `parse_deeppcb_annotation` function implemented in the earlier planning phase will fail over real data. It must be updated to handle `line.strip().split(' ')` (space separation) instead of strictly comma separation.