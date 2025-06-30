# Project Structure: C60.ai Molecular AutoML

## Main Package Layout

- `c60/` - Core logic, evolutionary engine, GNNs, utils, introspection
  - `core/` - Pipeline, automl, optimizer, evaluator, generator
  - `engine/` - Evolutionary search, graph schema/validation, search loop
  - `gnn/` - GNN models and encoders
  - `introspect/` - Explainability, RL/NAS agents, pipeline story
  - `utils/` - Utility modules
- `api/`, `cli/`, `cloud/` - (Scaffolded, only `__init__.py` present)
- `examples/` - Example scripts and pipelines
- `notebooks/` - Jupyter notebooks
- `docs/` - Documentation, Sphinx config
- `tests/` - Unit/integration tests

## Introspect Subpackage
- `introspector.py` - PipelineIntrospector (logging, explainability)
- `agents.py` - RLSearchAgent, NASearchAgent
- `story.py` - PipelineStory (stories, visualization)
- `__init__.py` - Imports all above

## Notes
- All core features are modularized for research/extensibility.
- Sphinx config and requirements should list all required doc extensions.
- Update README(s) with this structure.
