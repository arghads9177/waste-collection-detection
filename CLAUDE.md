# CLAUDE.md — Waste Classification System

This repo follows the **cv-project-engineering** conventions. Key rules:

- **Environment**: uv only. Add deps with `uv add`, never bare pip. After dependency
  changes run `uv export -o requirements/requirements-dev.txt` and `uv export --no-dev -o requirements/requirements.txt`.
- **Config**: all runtime configuration comes from `.env` via `src/waste_classification/config.py`
  (pydantic-settings). Never hardcode paths, thresholds, URLs, or hyperparameters.
  Never read os.environ directly — import `settings`.
- **Logging**: no print() outside notebooks. `logger = logging.getLogger(__name__)`;
  entry points call `setup_logging()` from `src/waste_classification/logging_config.py`.
- **Layout**: production logic lives in `src/waste_classification/` (data/ models/ inference/
  training/ utils/). `scripts/` are thin CLI wrappers. Notebooks are exploration only.
- **Data**: `data/raw` is immutable; everything in `data/processed` must be
  regenerable by a script. Datasets and weights are gitignored; model metadata JSONs
  in `models/metadata/` ARE committed and must accompany every trained model.
- **Tests**: every feature lands with tests. unit = pure logic, integration =
  pipeline on fixtures, smoke = one real image through the real model. Run: `uv run pytest`.
- **Docs**: keep README, docs/PRD.md, docs/TDD.md, docs/IMPLEMENTATION_PLAN.md current.
- **Phases**: Implement incrementally per docs/IMPLEMENTATION_PLAN.md (Phase 0-7), landing tests before moving on.
- **Model exports**: use self-describing dicts (backbone, num_classes, class_names, state_dict, hyperparameters, metrics) so checkpoints can rebuild from themselves without `.env`.
