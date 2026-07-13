# Implementation Plan — Waste Classification System

## Context

`docs/PRD.md` and `docs/TDD.md` are approved. This plan translates the TDD's module design into an ordered, phase-by-phase build sequence, landing each phase with its own tests before moving to the next, per the `cv-project-engineering` skill's "implement incrementally" convention.

Two execution environments drive the phase split: **local/CPU** (scaffolding, preprocessing, module code + tests, export, API, UI, Docker) and **Google Colab T4** (Phase 3 only — augmentation/training/evaluation notebook).

---

## Phase 0 — Repository Scaffolding

- Run the skill's `scripts/init_cv_project.py waste-collection-detection` (or apply its output manually since the folder already exists) to generate: `pyproject.toml` (package `waste_classification`, Python 3.11), `.gitignore`, `configs/.env.example`, `src/waste_classification/config.py` + `logging_config.py`, test skeletons, `Dockerfile`, `CLAUDE.md`.
- Reconcile with the existing `.gitignore`/README rather than overwriting blindly.
- Populate `configs/.env.example` with all variables from TDD §7 (DATA_*, MODEL_*, TRAIN_*, API_*, STREAMLIT_API_URL, PREDICTION_LOG_DB, OUTPUT_DIR, LOG_LEVEL).
- `uv sync && uv run pytest` should pass (empty/stub tests) before moving on.
- **Exit criteria:** clonable-and-runnable skeleton, `uv run pytest` green.

## Phase 1 — Data Preprocessing Pipeline (local, CPU)

- User manually downloads the Kaggle dataset into `data/raw/garbage_classification/<class>/*.jpg`.
- Implement `src/waste_classification/data/preprocessing.py`:
  - `find_corrupted_images()` — PIL open/verify, log + exclude failures.
  - `find_duplicate_images()` — hash-based (MD5 exact pass, optional perceptual hash), log counts removed per class.
  - `create_splits()` — stratified 70/15/15 train/val/test, writes `data/processed/splits/{train,val,test}.csv`, seeded via `DATA_SPLIT_SEED`.
  - **Portability (TDD §3):** `create_splits()` writes image paths **relative to the dataset root** (e.g. `plastic/img.jpg`), never absolute — this is what lets the same CSVs work on Colab.
- Implement `src/waste_classification/data/dataset.py`:
  - `WasteDataset` (torch `Dataset` reading from split CSVs + a **`data_root` argument** defaulting to `DATA_RAW_DIR`, so Colab can repoint it).
  - `build_train_transforms()` / `build_eval_transforms()` per TDD §3 augmentation policy.
- Thin `scripts/preprocess.py` CLI wired to the above.
- Tests: unit tests for corrupted/duplicate detection and split stratification (`tests/unit/`) using `tests/fixtures/` sample images (2-3 per a few classes, including one deliberately corrupted/duplicate fixture); a **portability guard test asserting CSVs contain only relative paths**; integration test loading `WasteDataset` end-to-end over fixtures.
- **Exit criteria:** running `scripts/preprocess.py` against the downloaded dataset produces clean split CSVs with logged corrupt/duplicate counts; `uv run pytest tests/unit tests/integration` green.

## Phase 2 — Model & Training Library Code (local, CPU-testable)

- Implement `src/waste_classification/models/factory.py`: `build_model(backbone, num_classes, pretrained)` for `efficientnet_b0` and `convnext_tiny` (torchvision pretrained, head replaced with 12-way linear), `load_checkpoint()`. **Checkpoints are self-describing dicts** (`backbone`, `num_classes`, `class_names`, `state_dict`, `hyperparameters`, `val_metrics`) so `load_checkpoint()`/export rebuild the right architecture from the file itself, not from `.env` (TDD §4).
- Implement `src/waste_classification/training/trainer.py`: `Trainer` class encapsulating warmup (frozen backbone) + fine-tune (full network) phases, AdamW, CosineAnnealingLR, early stopping, per-epoch checkpointing (self-describing dict format above), optional **class-weighted loss** (for the imbalance remediation path) — driven entirely by function calls, no CLI, since it will be *imported* by the Colab notebook per TDD §4.
- Implement `src/waste_classification/training/metrics.py`: `compute_classification_metrics()` (accuracy, macro-F1, per-class P/R/F1, confusion matrix).
- Tests: unit tests for `compute_classification_metrics()` against known confusion matrices; integration test running `Trainer.fit()` for one epoch on a tiny 2-class/4-image fixture set on CPU (seconds, not minutes) — this is what proves the training loop works *before* burning Colab time.
- **Exit criteria:** one-epoch CPU smoke run completes and checkpoints without error; all Phase 2 tests green.

## Phase 3 — Colab Training & Evaluation Notebook (Google Colab, T4 GPU)

- Author `notebooks/colab_train_eval.ipynb` per TDD §4/§5 structure:
  1. Setup — clone repo, **`pip install -e .`** (so `import waste_classification` resolves — cloning alone doesn't) + `pip install -r requirements/requirements-train.txt`, mount Google Drive and point `data_root` at the uploaded raw images, pull committed split CSVs from the clone, confirm GPU.
  2. Data & augmentation (import `WasteDataset`, transform builders; build `DataLoader`s).
  3. Training — both backbones via `build_model()` + `Trainer.fit()`; checkpoint every epoch + best-by-val-accuracy.
  4. Evaluation — `compute_classification_metrics()` per backbone on the test split; inline confusion matrix + metric tables.
  5. **Per-class-bar check + iterate (not one-shot):** explicitly verify per-class F1 ≥ 0.75 for all 12 classes; if any class misses (likely given dataset imbalance), re-run with the remediation ladder — class-weighted loss → more fine-tune epochs/lower LR → targeted augmentation — before declaring a winner.
  6. Model comparison & selection — side-by-side table (accuracy, macro-F1, param count, rough latency estimate); markdown cell documents the winning backbone and why.
  7. Download — zip `models/checkpoints/`, `outputs/training_runs/`, `outputs/eval_reports/`.
- **Dataset-to-Colab mechanism:** user uploads raw images to Google Drive once; notebook mounts + points `data_root` at it (Kaggle re-download documented as fallback). Avoids re-uploading ~2GB per session.
- Regenerate/maintain `requirements/requirements-train.txt` (GPU-relevant extras) before this phase, since the notebook installs from it.
- **This phase is executed manually on Colab by the user**, not by Claude Code directly — Claude Code's job is authoring the notebook and the library code it calls; running it and downloading artifacts is a user action.
- **Exit criteria:** downloaded `models/checkpoints/` contains both backbones' best checkpoints; `outputs/eval_reports/` contains per-class metrics for both; a clear winner is recorded in the notebook.

## Phase 4 — Export & Local Evaluation (local, CPU)

- Copy the winning checkpoint to `models/exported/waste-classifier-v1.pt`.
- Implement `src/waste_classification/inference/export.py`: `export_to_onnx()` (opset 17, dynamic batch axis) + `verify_onnx_output()` (PyTorch vs ONNX max-abs-diff < 1e-3 on a fixture batch). Wire up thin `scripts/export.py` — it **reads the backbone from the checkpoint dict** and rebuilds via `build_model()`, so it works regardless of `.env` (TDD §4).
- Implement `scripts/evaluate.py --benchmark-latency`: CPU-only latency benchmark (N=100 after 10 warmup, mean/p50/p95) against the exported model — the one measurement deliberately kept off the GPU notebook per TDD §5.
- Write `models/metadata/waste-classifier-v1.json` (backbone, dataset split version, hyperparameters, eval metrics, latency, training git commit) per the skill's model-metadata convention — this file gets committed.
- Tests: integration test for ONNX export + `verify_onnx_output()` round-trip using a fixture image/model.
- **Exit criteria:** `models/onnx/waste-classifier-v1.onnx` exists and passes the verification check; metadata JSON committed; latency benchmark meets the ≤300ms PRD target (or is flagged if not).

## Phase 5 — Inference Module & REST API (local, CPU)

> **Not blocked on Colab:** implement `scripts/make_dummy_model.py` (reuses `build_model()` + `export_to_onnx()`) to emit a randomly-initialized `.pt` + `.onnx` of the correct architecture, and build/test everything in Phases 5–6 against it. Real weights (Phases 3–4) only change the numbers, not the plumbing; the smoke tests re-run against the genuine model once it exists.

- Implement `src/waste_classification/inference/predictor.py`: `Predictor.predict(image) -> PredictionResult` — preprocessing (eval transforms) → ONNX Runtime CPU forward pass → softmax → postprocessing.
- Implement `src/waste_classification/api/`: FastAPI `create_app()`, `/health`, `/predict` (multipart upload → `PredictionResult` + latency), `/predictions/recent`.
- Implement `src/waste_classification/api/db.py`: SQLite (`PREDICTION_LOG_DB`) — `log_prediction()`, `get_recent_predictions()`; every `/predict` call logs timestamp, predicted class, confidence, full score vector, latency.
- Error handling per TDD §8 (400 invalid upload, 413 oversized, fail-fast on model load error).
- Tests: smoke test loading an exported model (PyTorch and ONNX — dummy during dev, real once Phase 4 lands) through `Predictor.predict()` on a fixture image, asserting a valid 12-class probability distribution; integration test hitting `/predict` via FastAPI `TestClient` against the dummy model with a fixture image, asserting the SQLite log row is written.
- **Exit criteria:** `uvicorn` serves `/health` and `/predict` locally; a manual curl/upload round-trip returns predicted class + confidence scores; smoke + integration tests green.

## Phase 6 — Web UI (Streamlit)

- Implement a Streamlit app (`scripts/run_ui.py` or `src/waste_classification/ui/app.py` run via `streamlit run`): file upload widget → POST to `STREAMLIT_API_URL/predict` → display predicted class, confidence bar chart across all 12 classes.
- Manual verification: start FastAPI service + Streamlit app together, upload a real image through the browser, confirm the round trip and that a row lands in `outputs/predictions.db`.
- **Exit criteria:** end-to-end manual test in a browser succeeds for at least one image per a sample of classes.

## Phase 7 — Deployment, Docs, and Wrap-up

- `Dockerfile` for the FastAPI service (CPU base, `requirements/requirements.txt`, copies `models/exported/` + `models/onnx/`, runs uvicorn); `docker/docker-compose.yml` adding the Streamlit service pointed at the API container via `STREAMLIT_API_URL`.
- Regenerate `requirements/requirements.txt` and `requirements/requirements-dev.txt` via `uv export`.
- Write the **performance report** (`docs/performance_report.md` or similar): final accuracy/macro-F1/per-class metrics, confusion matrix, latency numbers, backbone comparison table, from Phase 3/4 artifacts.
- Complete `README.md`: overview, install (uv + Docker), usage (preprocess → Colab train → export → serve → UI), architecture diagram (Mermaid, from TDD §1), API docs, known limitations, future improvements.
- Full test suite pass (`uv run pytest`) across unit/integration/smoke tiers.
- **Exit criteria:** `docker compose up` serves both API and UI; README allows a fresh clone to reproduce inference (given a downloaded model) without tribal knowledge.

---

## Verification Approach (applies throughout)

- After each phase, run `uv run pytest` for the tiers that phase introduced/touches before moving to the next phase — no phase starts with red tests from the previous one.
- Phase 3 (Colab) is the one phase not directly executed by Claude Code; Phase 2's CPU smoke test is what de-risks it beforehand, and Phase 4's local re-verification (ONNX round-trip, smoke test) is what confirms the downloaded artifacts are actually usable.
- Before Phase 7 sign-off, do one full manual walkthrough: preprocess → (assume Colab already run) → export → start API → start UI → upload an image in the browser → confirm prediction, confidence display, and a new row in `outputs/predictions.db`.
