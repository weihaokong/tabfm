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

"""Regression tests for query-varying mask/bias handling in FLASH attention."""

import jax
import jax.numpy as jnp
import numpy as np
from tabfm.src.jax import memory_efficient_attention

from absl.testing import absltest

_BATCH = 2
_HEADS = 2
_DIM = 8


def _reference_attention(query, key, value, bias):
  """Exact attention: the model's JAX backend."""
  return jax.nn.dot_product_attention(
      query=query, key=key, value=value, bias=bias, scale=1.0
  )


def _flash_attention(query, key, value, bias, chunk_size):
  """FLASH attention, called the way MultiheadAttention calls it."""
  return memory_efficient_attention.dot_product_attention_multihead(
      query=query,
      key=key,
      value=value,
      bias=bias,
      dtype=query.dtype,
      query_chunk_size=chunk_size,
      key_chunk_size=chunk_size,
  )


def _mask_to_bias(mask):
  """Keep-mask to additive bias, as MultiheadAttention builds it."""
  return jnp.where(mask, 0.0, -1e30)


class MemoryEfficientAttentionTest(absltest.TestCase):

  def _random_qkv(self, q_len, kv_len):
    rng = np.random.default_rng(0)
    query = jnp.asarray(
        rng.standard_normal((_BATCH, q_len, _HEADS, _DIM)), jnp.float32
    )
    key = jnp.asarray(
        rng.standard_normal((_BATCH, kv_len, _HEADS, _DIM)), jnp.float32
    )
    value = jnp.asarray(
        rng.standard_normal((_BATCH, kv_len, _HEADS, _DIM)), jnp.float32
    )
    return query, key, value

  def _assert_flash_matches_reference(self, q_len, kv_len, bias, chunk_size):
    query, key, value = self._random_qkv(q_len, kv_len)
    actual = _flash_attention(query, key, value, bias, chunk_size)
    expected = _reference_attention(query, key, value, bias)
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-5
    )

  def test_query_varying_mask_multiple_chunks(self):
    # A mask that varies per query row must not collapse to the chunk's first row.
    q_len = kv_len = 256
    query_idx = jnp.arange(q_len)[:, None]
    key_idx = jnp.arange(kv_len)[None, :]
    mask = jnp.broadcast_to(
        (key_idx <= query_idx + 50)[None, None], (_BATCH, 1, q_len, kv_len)
    )
    self._assert_flash_matches_reference(
        q_len, kv_len, _mask_to_bias(mask), chunk_size=128
    )

  def test_query_varying_mask_single_chunk(self):
    # Same guarantee when the whole sequence is a single chunk (chunk_size == q_len).
    q_len = kv_len = 64
    query_idx = jnp.arange(q_len)[:, None]
    key_idx = jnp.arange(kv_len)[None, :]
    mask = jnp.broadcast_to(
        (key_idx <= query_idx)[None, None], (_BATCH, 1, q_len, kv_len)
    )
    self._assert_flash_matches_reference(
        q_len, kv_len, _mask_to_bias(mask), chunk_size=q_len
    )

  def test_broadcast_key_mask(self):
    # The [B, 1, 1, S] key-padding mask the shipped model builds must stay exact.
    q_len = kv_len = 256
    key_idx = jnp.arange(kv_len)[None, :]
    mask = jnp.broadcast_to(
        (key_idx < 100)[None, None], (_BATCH, 1, 1, kv_len)
    )
    self._assert_flash_matches_reference(
        q_len, kv_len, _mask_to_bias(mask), chunk_size=128
    )

  def test_kv_broadcastable_bias(self):
    # A bias with kv dim 1 (broadcast over keys) must not crash dynamic_slice.
    q_len = kv_len = 256
    rng = np.random.default_rng(1)
    bias = jnp.asarray(
        rng.standard_normal((_BATCH, 1, q_len, 1)), jnp.float32
    )
    self._assert_flash_matches_reference(q_len, kv_len, bias, chunk_size=128)


if __name__ == "__main__":
  absltest.main()
