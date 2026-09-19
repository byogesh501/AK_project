\# AK Consultants AI Visual Intelligence Platform



\## Project Rules



\- PCB is the first and current domain.

\- Treat actual code and latest test/log results as the source of truth.

\- Inspect existing code before modifying it.

\- Make minimal, practical changes.

\- Do not rebuild working components without evidence.

\- Do not make unnecessary refactors or architecture changes.



\## ML Requirements



\- All inspection models must be trained from scratch.

\- NEVER use pretrained weights.

\- NEVER use pretrained YOLO.

\- NEVER use ImageNet pretrained weights.

\- NEVER add Deep SVDD.



\## Dataset



\- Dataset: DeepPCB.

\- Preserve the existing 1050 / 225 / 225 train/validation/test split.

\- Preserve pair-level split isolation.

\- NEVER leak test data into training or validation.

\- Preserve paired defect-image/template transformations.

\- Preserve annotation masking.



\## Current PCB Detector



\- Custom anchor-free multi-scale CNN detector.

\- Backbone → P3/P4/P5 → FPN/PAN → detection head.

\- Outputs: boxes, objectness, classes.

\- Loss: CIoU + BCE objectness + BCE classification.

\- Do not replace or redesign the detector unless actual evidence requires it.



\## Current Development Priority



1\. Detector evaluation

2\. Unknown anomaly detection

3\. Known + unknown result integration

4\. Confidence/risk scoring

5\. Explainability

6\. Human-in-the-loop

7\. Dashboard/report

8\. API/database when actually required



\## Development Workflow



Before changing code:



1\. Inspect the relevant implementation.

2\. Inspect related tests.

3\. Identify the smallest required change.

4\. Make the change.

5\. Run relevant tests.

6\. Report exactly what changed and what was verified.



\## Important



\- Do not trust stale documentation over actual code and latest test results.

\- Do not claim a feature is implemented without evidence.

\- Do not modify unrelated files.

\- Do not add unnecessary dependencies.

\- Do not jump to dashboard/API/database work while the core PCB inspection pipeline is still being validated.

