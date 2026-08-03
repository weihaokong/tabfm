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

"""Sharded classification with the JAX backend via sequence-parallel inference.

The classification twin of ``seqpar_jax_regression_example.py``.
``seqpar.predict_proba`` shards the in-context rows of each ensemble member
across all visible devices, then applies the estimator's own class-shift
correction and probability averaging. Run as a single process driving all
local devices:

    python examples/seqpar_jax_classification_example.py

See ``seqpar_jax_regression_example.py`` for the 2-D mesh
(``seqpar.make_mesh_2d``) and splash-attention (``splash=True``) knobs, which
work identically here.
"""

import numpy as np
import pandas as pd

import tabfm
from tabfm.src.jax import seqpar


def run_example(model=None) -> np.ndarray:
  """Generates dummy data and runs sharded classification."""
  if model is None:
    model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")

  clf = tabfm.TabFMClassifier(model=model, n_estimators=4, random_state=0)

  # Mixed column types, as in classification_example.py.
  X_train = pd.DataFrame({
      "num_feat_1": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
      "cat_feat_1": ["A", "B", "A", "B", "C", "A"],
  })
  y_train = np.array(["yes", "no", "yes", "no", "no", "yes"])

  X_test = pd.DataFrame({
      "num_feat_1": [2.0, 4.0],
      "cat_feat_1": ["B", "A"],
  })

  clf.fit(X_train, y_train)

  # Sharded predict_proba; columns follow ``clf.classes_``. As in the
  # regression example, omitting ``mesh`` uses seqpar's default 1-D mesh over
  # all visible devices.
  probs = seqpar.predict_proba(clf, X_test)
  return probs


if __name__ == "__main__":
  print(
      "Running TabFM sharded classification... (Note: compilation and model"
      " execution may take a few minutes on first run)"
  )
  probabilities = run_example()
  print("Sharded classification probabilities:\n", probabilities)
