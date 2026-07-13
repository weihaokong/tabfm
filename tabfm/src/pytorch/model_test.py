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

import unittest

from absl import logging
import numpy as np
import torch
from flax import nnx
import jax.numpy as jnp

from tabfm.src.jax.model import TabFM as JaxTabFM, YEmbeddingScheme
from tabfm.src.pytorch import model as PyTorchTabFM
from tabfm.src.hugging_face.torch_convert import convert, jax_params


class PyTorchModelTest(unittest.TestCase):

  def test_pytorch_model_instantiation(self):
    """Verifies that the PyTorch model instantiates with config and runs a forward pass."""
    model = PyTorchTabFM.TabFM(
        embed_dim=16,
        max_classes=10,
        col_num_blocks=2,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=2,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        is_classifier=True
    )
    self.assertIsNotNone(model)
    
    # Run dummy forward pass
    x = torch.randn(2, 4, 6) # [B, T, H]
    y = torch.randint(0, 3, (2, 4)) # [B, T]
    train_size = torch.tensor([2, 3]) # [B]
    out = model(x, y, train_size)
    self.assertEqual(out.shape, (2, 4, 10))

  def test_jax_pytorch_parity(self):
    """Verifies JAX vs PyTorch model outputs are numerically equal up to 1e-4."""
    for is_classifier in [True, False]:
      with self.subTest(is_classifier=is_classifier):
        # 1. Config definitions
        cfg = dict(
            embed_dim=32,
            max_classes=4,
            col_num_blocks=2,
            col_nhead=4,
            col_num_inds=16,
            row_num_blocks=2,
            row_nhead=4,
            row_num_cls=4,
            icl_num_blocks=3,
            icl_nhead=4,
            ff_factor=4,
            feature_group_size=3,
            use_bias=False
        )

        # 2. Instantiate JAX model deterministically (random init)
        jax_model = JaxTabFM(
            loss="cross_entropy" if is_classifier else "rmse",
            activation="swiglu",
            feature_group=True,
            **cfg,
            y_embedding_scheme=YEmbeddingScheme.ADD_Y_TO_X_POST_EMBEDDING,
            rngs=nnx.Rngs(42),
            dtype=jnp.float32
        )

        # 3. Instantiate PyTorch model with matching config
        torch_model = PyTorchTabFM.TabFM(
            embed_dim=cfg["embed_dim"],
            max_classes=cfg["max_classes"],
            col_num_blocks=cfg["col_num_blocks"],
            col_nhead=cfg["col_nhead"],
            col_num_inds=cfg["col_num_inds"],
            row_num_blocks=cfg["row_num_blocks"],
            row_nhead=cfg["row_nhead"],
            row_num_cls=cfg["row_num_cls"],
            icl_num_blocks=cfg["icl_num_blocks"],
            icl_nhead=cfg["icl_nhead"],
            ff_factor=cfg["ff_factor"],
            feature_group_size=cfg["feature_group_size"],
            is_classifier=is_classifier
        )

        state_dict, missing = convert(jax_params(jax_model), torch_model)
        self.assertEqual(len(missing), 0)
        torch_model.load_state_dict(state_dict, strict=True)
        torch_model.eval()

        # 5. Prepare random input data
        b, t, h = 3, 5, 8
        np.random.seed(123)
        x_np = np.random.normal(size=(b, t, h)).astype(np.float32)
        
        if is_classifier:
          y_np = np.random.randint(0, cfg["max_classes"], size=(b, t)).astype(np.float32)
        else:
          y_np = np.random.normal(size=(b, t)).astype(np.float32)
          
        train_size_np = np.array([2, 3, 4], dtype=np.int32)
        d_np = np.array([5, 6, 7], dtype=np.int32) # active feature counts (d < h)
        cat_mask_np = np.zeros((b, h), dtype=bool)
        cat_mask_np[0, :3] = True
        cat_mask_np[1, :4] = True

        # JAX Inputs
        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        train_size_jax = jnp.array(train_size_np)
        d_jax = jnp.array(d_np)
        cat_mask_jax = jnp.array(cat_mask_np)

        # PyTorch Inputs
        x_torch = torch.from_numpy(x_np)
        y_torch = torch.from_numpy(y_np)
        train_size_torch = torch.from_numpy(train_size_np)
        d_torch = torch.from_numpy(d_np)
        cat_mask_torch = torch.from_numpy(cat_mask_np)

        # 6. Forward passes
        # JAX
        jax_out = jax_model(x_jax, y_jax, train_size_jax, cat_mask=cat_mask_jax, d=d_jax)
        jax_out_np = np.asarray(jax_out)

        # PyTorch
        with torch.no_grad():
          torch_out = torch_model(x_torch, y_torch, train_size_torch, cat_mask=cat_mask_torch, d=d_torch)
          torch_out_np = torch_out.numpy()

        # 7. Compare JAX vs PyTorch outputs
        diff = np.abs(jax_out_np - torch_out_np)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Assert max difference is less than 1e-4
        self.assertLess(
            max_diff,
            1e-4,
            f"Fidelity discrepancy found: max diff = {max_diff}, mean diff = {mean_diff} for is_classifier={is_classifier}"
        )

  def _make_prefill_decode_model(self):
    model = PyTorchTabFM.TabFM(
        embed_dim=32,
        max_classes=5,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=4,
        icl_num_blocks=2,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=3,
        is_classifier=True,
    )
    model.eval()
    with torch.no_grad():
      # fourier_frequencies{,_cat} are checkpoint-loaded buffers that default
      # to all-zero at random init, which would make the Fourier cell features
      # (and hence every row's embedding) identically constant regardless of
      # x -- a vacuous test. Seed them so cell embeddings genuinely depend on
      # per-row input, matching what a real (checkpoint-loaded) model does.
      model.cell_embedder.fourier_frequencies.normal_()
      model.cell_embedder.fourier_frequencies_cat.normal_()
      # per_dim_scale also defaults to all-zero; give it a small nonzero init
      # so attention scores are not degenerate.
      for name, p in model.named_parameters():
        if "per_dim_scale" in name:
          p.normal_(std=0.1)
    return model

  def test_prefill_decode_consistency(self):
    """decode(test | prefill(context)) should match forward(concat) on test rows."""
    torch.manual_seed(0)
    b, t, h = 1, 20, 4
    train_len = 10
    max_classes = 5

    model = self._make_prefill_decode_model()

    x = torch.randn(b, t, h)
    y = torch.randint(0, max_classes, (b, t)).float()
    train_size = torch.tensor([train_len])

    with torch.no_grad():
      out_full = model(x, y, train_size)
    out_full_test = out_full[:, train_len:, :]

    x_train, y_train = x[:, :train_len, :], y[:, :train_len]
    x_test = x[:, train_len:, :]

    with torch.no_grad():
      _, cache = model.prefill(x_train, y_train)
      self.assertIn("col1", cache)
      self.assertIn("col2", cache)
      self.assertIn("icl", cache)
      out_decode = model.decode(x_test, cache)

    self.assertEqual(out_decode.shape, out_full_test.shape)
    # Sanity check that the test data isn't in the degenerate "all rows
    # identical" regime (which would make this parity check vacuous).
    self.assertGreater(
        (out_full_test[:, 0] - out_full_test[:, 1]).abs().max().item(), 1e-4
    )
    diff_fp32 = (out_full_test - out_decode).abs()
    max_diff_fp32 = diff_fp32.max().item()
    logging.info(
        "[prefill/decode parity] fp32 max-abs-diff = %.6e", max_diff_fp32
    )
    # fp32 prefill/decode is algebraically exact; gate near machine precision.
    self.assertLess(
        max_diff_fp32, 1e-5,
        f"fp32 prefill/decode vs full-forward diff too large: {max_diff_fp32}"
    )

    # Separately report the bf16 max-abs-diff (not gated -- bf16 has ~1e-2/1e-3
    # relative precision, so this is diagnostic rather than a strict gate).
    model_bf16 = self._make_prefill_decode_model()
    model_bf16.load_state_dict(model.state_dict())
    model_bf16 = model_bf16.to(torch.bfloat16).eval()

    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = y.to(torch.bfloat16)
    x_train_bf16, y_train_bf16 = x_bf16[:, :train_len, :], y_bf16[:, :train_len]
    x_test_bf16 = x_bf16[:, train_len:, :]

    with torch.no_grad():
      out_full_bf16 = model_bf16(x_bf16, y_bf16, train_size)
      out_full_test_bf16 = out_full_bf16[:, train_len:, :]
      _, cache_bf16 = model_bf16.prefill(x_train_bf16, y_train_bf16)
      out_decode_bf16 = model_bf16.decode(x_test_bf16, cache_bf16)

    max_diff_bf16 = (
        out_full_test_bf16.float() - out_decode_bf16.float()
    ).abs().max().item()
    logging.info(
        "[prefill/decode parity] bf16 max-abs-diff = %.6e", max_diff_bf16
    )

  def test_quantized_cache_decode_consistency(self):
    """decode() with an int8-quantized ICLearningCache stays close to fp32."""
    torch.manual_seed(0)
    b, t, h = 1, 20, 4
    train_len = 10
    max_classes = 5

    model = self._make_prefill_decode_model()

    x = torch.randn(b, t, h)
    y = torch.randint(0, max_classes, (b, t)).float()
    x_train, y_train = x[:, :train_len, :], y[:, :train_len]
    x_test = x[:, train_len:, :]

    with torch.no_grad():
      _, cache_fp32 = model.prefill(x_train, y_train)
      out_decode_fp32 = model.decode(x_test, cache_fp32)

      icl_cache = cache_fp32["icl"]
      quantized_icl_cache = icl_cache.quantize()
      for (k_q, v_q) in quantized_icl_cache.layer_caches:
        self.assertIsInstance(k_q, PyTorchTabFM.QuantizedTensor)
        self.assertEqual(k_q.data.dtype, torch.int8)
        self.assertIsInstance(v_q, PyTorchTabFM.QuantizedTensor)
        self.assertEqual(v_q.data.dtype, torch.int8)
      self.assertEqual(
          quantized_icl_cache.prefill_seq_len, icl_cache.prefill_seq_len
      )
      cache_quantized = dict(cache_fp32, icl=quantized_icl_cache)
      out_decode_quantized = model.decode(x_test, cache_quantized)

    self.assertEqual(out_decode_quantized.shape, out_decode_fp32.shape)
    max_diff = (out_decode_fp32 - out_decode_quantized).abs().max().item()
    fp32_scale = out_decode_fp32.abs().max().item()
    rel_diff = max_diff / (fp32_scale + 1e-12)
    logging.info(
        "[quantized cache] fp32-vs-int8 max-abs-diff = %.6e "
        "(rel = %.6e of output scale %.4f)",
        max_diff,
        rel_diff,
        fp32_scale,
    )
    # int8 is lossy; gate on relative deviation (2% >> int8's ~0.8% resolution).
    self.assertLess(
        rel_diff, 0.02,
        f"int8-quantized cache decode diverged too much from fp32: "
        f"rel={rel_diff:.3e} (abs={max_diff:.3e}, scale={fp32_scale:.3e})"
    )

  def test_quantize_unsupported_dtype_raises(self):
    """quantize() rejects a dtype not in the registered quantization ranges."""
    t = torch.randn(2, 3)
    with self.assertRaises(ValueError):
      PyTorchTabFM._quantize_tensor(t, dtype=torch.int16)

  def test_move_cache_to_device_roundtrip(self):
    """move_cache_to_device preserves values and structure (CPU roundtrip)."""
    torch.manual_seed(0)
    b, t, h = 1, 12, 4
    train_len = 8
    max_classes = 5
    model = self._make_prefill_decode_model()
    x = torch.randn(b, t, h)
    y = torch.randint(0, max_classes, (b, t)).float()
    x_train, y_train = x[:, :train_len, :], y[:, :train_len]
    x_test = x[:, train_len:, :]

    with torch.no_grad():
      _, cache = model.prefill(x_train, y_train)
      cache["icl"] = cache["icl"].quantize()
      moved = PyTorchTabFM.move_cache_to_device(cache, "cpu")
      out_direct = model.decode(x_test, cache)
      out_moved = model.decode(x_test, moved)

    self.assertTrue(torch.equal(out_direct, out_moved))
    for (k, v), (k_m, v_m) in zip(
        cache["icl"].layer_caches, moved["icl"].layer_caches
    ):
      self.assertTrue(torch.equal(k.data, k_m.data))
      self.assertTrue(torch.equal(k.scale, k_m.scale))
      self.assertEqual(str(k_m.data.device), "cpu")
      self.assertTrue(torch.equal(v.data, v_m.data))
      self.assertTrue(torch.equal(v.scale, v_m.scale))


if __name__ == "__main__":
  unittest.main()
