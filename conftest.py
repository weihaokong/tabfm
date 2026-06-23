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

"""Pytest configuration.

The test suite uses ``absltest.TestCase`` helpers (e.g. ``create_tempdir``),
which read absl flags such as ``--test_tmpdir``. Those flags are only parsed by
``absltest.main()``; under the pytest runner they are never parsed, raising
``UnparsedFlagAccessError``. Parse them here so absltest-based tests work under
pytest.
"""

import sys

from absl import flags


def pytest_configure(config):  # noqa: D401  (pytest hook)
  del config  # Unused.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv[:1])
