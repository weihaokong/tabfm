# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sharded regression with the JAX backend via sequence-parallel inference.

The JAX twin of ``seqpar_regression_example.py`` (which uses the PyTorch
backend). ``seqpar.predict`` shards the in-context rows of each ensemble
member across all visible devices, so training folds that exceed a single
device's memory can be used as context. Run as a single process driving all
local devices:

    python examples/seqpar_jax_regression_example.py

On a multi-host TPU slice, run the same script on every worker after
``jax.distributed.initialize()`` (see ``tpu_validate_airfoil.py``).

Optional knobs shown below:
  * ``mesh=seqpar.make_mesh_2d(D)`` -- a 2-D (data x seqpar) grid: ``D``
    ensemble members run concurrently, each sequence-sharded over the
    remaining ``device_count // D`` devices. ``D`` must divide the device
    count. Omit for the 1-D default (one member at a time over all devices).
  * ``splash=True`` -- use the fused Pallas splash-attention kernel for the
    sharded ICL attention (TPU only; the default memory-efficient attention
    also runs on CPU/GPU).
"""

import numpy as np
import pandas as pd

import tabfm
from tabfm.src.jax import seqpar


def run_example(model=None) -> np.ndarray:
  """Generates dummy data and runs sharded regression."""
  if model is None:
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="regression")

  # Standard scikit-learn style estimator; ``fit`` is cheap (no device work).
  reg = tabfm.TabFMRegressor(model=model, n_estimators=4, random_state=0)

  X_train = pd.DataFrame({
      "num_feat_1": [1.5, 2.5, 3.5, 4.5, 5.5],
      "cat_feat_1": ["A", "B", "A", "B", "C"],
  })
  y_train = np.array([10.5, 20.0, 11.0, 29.5, 21.0])

  X_test = pd.DataFrame({
      "num_feat_1": [2.0, 4.0],
      "cat_feat_1": ["B", "A"],
  })

  reg.fit(X_train, y_train)

  # Sharded predict: same fitted estimator, same ensemble combination as
  # ``reg.predict`` -- only the forward pass is sharded across devices.
  # With no ``mesh`` argument, seqpar builds its default 1-D mesh over ALL
  # visible devices (axis "seqpar_rows"), i.e. these two calls are equivalent:
  #   seqpar.predict(reg, X_test)
  #   seqpar.predict(reg, X_test, mesh=jax.sharding.Mesh(devs, ("seqpar_rows",)))
  # where ``devs`` is jax.devices() ordered by (process_index, id).
  preds = seqpar.predict(reg, X_test)

  # 2-D grid + splash kernel (uncomment on a TPU with >= 4 devices):
  #   mesh = seqpar.make_mesh_2d(2)  # 2 members at a time, seq over the rest
  #   preds = seqpar.predict(reg, X_test, mesh=mesh, splash=True)
  return preds


if __name__ == "__main__":
  print(
      "Running TabFM sharded regression... (Note: compilation and model"
      " execution may take a few minutes on first run)"
  )
  predictions = run_example()
  print("Sharded regression predictions:\n", predictions)
