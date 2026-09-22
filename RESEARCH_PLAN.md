# HandCalib-WM research proposal

A calibration-aware, causal hand state-space model using RGB-D and an IMU glove, with MoCap supervision during training only.

- [Complete formulation, architecture, losses and training plan](docs/research/HAND_WORLD_MODEL_PLAN.md)
- [Related work: 22 references with reading notes and limitations](docs/research/RELATED_WORK.md)
- [Data contract, baselines, splits, metrics and implementation tests](docs/research/EXPERIMENT_PROTOCOL.md)
- [Proposed configuration](docs/research/handcalib.proposed.yaml)
- [Project-page source](docs/index.html)
- [GitHub Pages deployment instructions](docs/research/PAGES_SETUP.md)

**Status:** research design, not trained results or a runnable training package. The existing calibration/delivery code, reviewed media, and README remain unchanged. Original data must be audited before training, particularly raw-IMU availability, metric depth, marker-versus-joint semantics, and MoCap-conditioned visualizations.
