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

"""Regression tests for predict on a host with more than one visible device.

These run on simulated CPU devices, so they exercise the multi-device forward
path in CI without needing GPUs. The device count must be requested before JAX
initializes, which is why this lives in its own module.
"""

import os

# Request two host devices before JAX initializes so jax.devices() returns more
# than one. Kept in its own test module so it does not affect other tests.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

from absl.testing import absltest
import numpy as np

try:
  from flax import nnx
  from tabfm.src.jax import model as tabfm_model

  HAS_JAX = True
except ImportError:
  HAS_JAX = False

from tabfm.src.classifier_and_regressor import TabFMClassifier
from tabfm.src.classifier_and_regressor import TabFMRegressor

# pylint: disable=invalid-name


def _tiny_model(loss):
  return tabfm_model.TabFM(
      loss=loss,
      max_classes=2,
      embed_dim=8,
      col_num_blocks=1,
      col_nhead=2,
      col_num_inds=8,
      row_num_blocks=1,
      row_nhead=2,
      row_num_cls=1,
      icl_num_blocks=1,
      icl_nhead=2,
      rngs=nnx.Rngs(0),
  )


class MultiDevicePredictTest(absltest.TestCase):
  """predict must work when more than one device is visible and no mesh is set.

  Without a user-configured mesh the default batch is 1, but the JAX forward
  path built a data sharding over every visible device. That forced a batch of
  size 1 into an N-way shard (IndivisibleError) and placed the inputs on all
  devices while the weights stayed on device 0 (device mismatch). The forward
  path now keeps the sharding it derives from the active mesh (none by default,
  so the call is replicated), which is correct for one or many devices.
  """

  def setUp(self):
    super().setUp()
    if not HAS_JAX:
      self.skipTest("JAX is required for this test.")
    import jax  # pylint: disable=g-import-not-at-top

    if len(jax.devices()) < 2:
      self.skipTest("Requires at least two (simulated) devices.")

  def test_classifier_predict_multi_device(self):
    clf = TabFMClassifier(model=_tiny_model("cross_entropy"), n_estimators=2)
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    y = rng.randint(0, 2, size=20)
    clf.fit(X, y)
    preds = clf.predict(X[:5])
    self.assertEqual(np.asarray(preds).shape, (5,))

  def test_regressor_predict_multi_device(self):
    reg = TabFMRegressor(model=_tiny_model("mse"), n_estimators=2)
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    y = rng.rand(20)
    reg.fit(X, y)
    preds = reg.predict(X[:5])
    self.assertEqual(np.asarray(preds).shape, (5,))


if __name__ == "__main__":
  absltest.main()
