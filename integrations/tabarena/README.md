# TabFM ↔ TabArena / AutoGluon integration

The TabFM model contribution for the [TabArena](https://github.com/autogluon/tabarena)
leaderboard. These files follow TabArena's per-model contribution layout and
belong at `packages/tabarena/src/tabarena/models/tabfm/` in a TabArena
checkout:

- `model.py` — `TabFMModel`, an AutoGluon `AbstractTorchModel` wrapper around
  the PyTorch backend (`tabfm[pytorch]`). AutoGluon-side preprocessing is a
  no-op (TabFM handles mixed types and missing values natively); the wrapper
  manages device placement, activation-chunk sizes for large tasks, and
  pickling (the network is reloaded from the cached checkpoint on unpickle).
- `hpo.py` — the config set. `TabFM.ensemble()` combines 32 data-view members
  with an *internal* NNLS/greedy weighting fit on TabFM's own
  cross-validation; TabArena forbids that inner CV, so each member is exposed
  as its own config via `member_index` (bit-identical to the member inside a
  full `.ensemble()` run) and AutoGluon's OOF-based weighted ensemble replaces
  the internal weighting. 33 configs total: `c1` = plain default TabFM
  (n_estimators=32, uniform mean), `c2..c33` = the 32 `.ensemble()` members.
- `info.py` — TabArena `MethodMetadata` / `ModelInfo` registry entry.
- `__init__.py` — package exports.

Requires the `member_index` / `ensemble_method='greedy'` /
`feature_allocation` support in this repository's
`tabfm/src/classifier_and_regressor.py`.

Leaderboard variants produced by this config set:

- **TabFM (default)**: config `c1` — a plain `TabFMClassifier()` /
  `TabFMRegressor()` (itself a 32-view uniform-mean ensemble).
- **TabFM (tuned)**: the single best config by validation score.
- **TabFM (tuned + ensembled)**: AutoGluon's greedy weighted ensemble over
  the 33 configs' out-of-fold predictions — the TabArena-legal reproduction
  of `TabFM.ensemble()`.
