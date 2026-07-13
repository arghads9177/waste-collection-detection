# Product Requirements Document — Waste Classification System

**Author:** Argha Dey Sarkar | **Date:** 2026-07-13 | **Status:** Draft

## 1. Problem Statement

Municipal and commercial waste management relies heavily on manual sorting to separate recyclables from general waste, which is slow, inconsistent, and labor-intensive. Automated waste classification at the point of collection or at a materials recovery facility can speed up sorting, reduce contamination of recycling streams, and cut labor costs. Today this sorting is done by hand or not done at all, leading to recyclable material being landfilled.

## 2. Objective & Success Criteria

Build an image classification system that assigns a waste image to one of 12 predefined categories with a confidence score, exposed via a REST API and a simple web UI, so it can be integrated into a sorting workflow or used as a standalone classification tool.

Success criteria:
- Top-1 validation accuracy ≥ 85% on the held-out test split.
- Per-class F1 ≥ 0.75 for every one of the 12 classes (no class silently ignored).
- Inference latency ≤ 300ms per image on CPU (typical local/API deployment target).
- API returns a prediction + confidence scores for all 12 classes on a single upload.
- Every prediction is logged (input reference, predicted class, confidence, timestamp).

## 3. CV Task Mapping

| Business ask | Canonical CV task | Output |
|---|---|---|
| "What kind of waste is this?" | Multi-class image classification (12 classes) | Predicted class label + per-class confidence scores |

## 4. Users & Usage Context

- **Primary user (v1):** a single operator or developer uploading one image at a time via the web UI, or a client system calling the REST API programmatically.
- **Environment:** static images (photos of individual waste items), not live video streams. Lighting/background vary since the source dataset is a mix of web-scraped images.
- **Scale (v1):** low-throughput, single-image requests — not a batch/streaming pipeline. Designed to be extensible to batch later, not required now.

## 5. Data Position

- **Available data today:** [Kaggle — Garbage Classification (mostafaabla/garbage-classification)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification), ~15,000 images across 12 classes (paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, trash). Downloaded manually by the user and placed under `data/raw/`.
- **Data to collect:** none planned for v1 — training is entirely on the Kaggle dataset.
- **Labeling plan:** none needed — dataset ships pre-labeled by directory structure (one folder per class).
- **Privacy/compliance:** dataset is public and license-permitting for research/educational use; no PII/faces expected in waste imagery. No on-prem constraint.
- **Data quality risk:** dataset is known to contain some corrupted and duplicate images — the preprocessing pipeline must detect and remove both before splitting.

## 6. Constraints

- **Hardware / deployment target:** data preprocessing runs on the local laptop (CPU). Training and evaluation run on Google Colab (T4 GPU). Inference (API) is deployed/served on CPU-class hardware — no GPU assumed at serving time.
- **Latency / throughput budget:** ≤ 300ms per image on CPU inference — acceptable for a single-image upload workflow, not real-time video.
- **Licensing:** stick to permissively licensed libraries (PyTorch, torchvision, FastAPI, Streamlit — all open source, no AGPL).
- **Budget:** free-tier Google Colab (T4), no paid cloud infra required.

## 7. Cost of Errors

- **False positive (e.g., misclassifying trash as recyclable):** contaminates the recycling stream — a real-world materials-recovery-facility concern, but for this v1 educational/demo system the direct cost is a wrong label shown to the user.
- **False negative (recyclable misclassified as trash):** recyclable material lost to landfill — arguably the worse outcome since it defeats the system's purpose, but v1 optimizes for overall accuracy/macro-F1 rather than an asymmetric cost function. Threshold/cost-sensitive tuning is called out as a future improvement, not a v1 requirement.
- Both error types are visible to the user via the confidence scores shown in the UI/API response, so low-confidence predictions are self-flagging rather than silently wrong.

## 8. Scope

**In scope (v1):**
- Data preprocessing pipeline: load, detect/remove corrupted files, detect/remove duplicates, train/val/test split.
- Dataset loading + augmentation (torchvision transforms).
- Transfer learning + fine-tuning on both EfficientNet-B0 and ConvNeXt-Tiny; pick the winner by val accuracy + latency.
- Training and validation pipeline (runs on Colab T4).
- Evaluation report (per-class metrics, confusion matrix, latency benchmark).
- Model export to PyTorch (`.pt`) and ONNX.
- FastAPI inference service: upload image → predicted class + confidence scores, prediction logging to SQLite.
- Streamlit web UI: upload image, display prediction + confidence scores, calls the FastAPI service.
- Unit, integration, and smoke tests.
- README + performance report.

**Out of scope (v1):**
- Real-time video / multi-object detection (this is single-image classification, not detection despite the repo's name).
- Active learning / continuous retraining loop.
- Multi-user auth, batch upload, or a production-grade database (SQLite is sufficient for v1).
- Edge/mobile deployment (TensorRT/CoreML) — ONNX export is the deployment ceiling for v1.
- Cost-sensitive/asymmetric threshold tuning.

## 9. Milestones

| Milestone | Deliverable | Target |
|---|---|---|
| M1 | Data preprocessing pipeline (clean, dedupe, split) validated on downloaded dataset | Local, CPU |
| M2 | Trained EfficientNet-B0 and ConvNeXt-Tiny models meeting the 85% accuracy bar | Colab T4 |
| M3 | Evaluation report + exported ONNX model + FastAPI service + Streamlit UI, tested end-to-end | Local |

## 10. Open Questions

- None outstanding — resolved during requirements clarification: backbone (train both, pick winner), compute split (local preprocessing / Colab T4 training), UI stack (Streamlit), prediction logging (SQLite), dataset acquisition (manual download), package naming (`waste_classification`).
