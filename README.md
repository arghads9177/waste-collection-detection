# waste-collection-detection

## Overview
_One paragraph: the problem, the CV task, the approach. See docs/PRD.md and docs/TDD.md._

## Installation
```bash
uv sync                      # create environment from pyproject.toml + uv.lock
cp configs/.env.example .env # then edit values
uv run pytest                # verify setup
```

## Usage
```bash
uv run python scripts/train.py       # train
uv run python scripts/evaluate.py    # evaluate on val set
uv run python scripts/export.py      # export weights (ONNX/TensorRT)
uv run python scripts/infer.py       # run inference / serve
```

## Architecture
See `docs/architecture.md` (Mermaid diagram) and `docs/TDD.md`.

## Model & Data
- Weights are build artifacts — see `models/metadata/` for the versioned record.
- Datasets are never committed; see `data/README.md` for provenance and how to pull them.

## Known Limitations
_List honestly._

## Future Improvements
_Roadmap items._
