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

"""Reproduction script: TabFM v1.1.0 on TabArena kddcup09_appetency, fold 0.

Self-contained cross-machine comparison. Prints the full environment
(jax/jaxlib version, device, dtype, attention impls, chunk sizes) alongside
the metrics so two machines' results can be diffed unambiguously.

    pip install -e .[jax,examples] scikit-learn
    python repro_kddcup_jax.py --n-estimators 1              # 1-member, bf16
    python repro_kddcup_jax.py --n-estimators 1 --dtype f32  # the control
    python repro_kddcup_jax.py                               # full 32-member
    python repro_kddcup_jax.py --backend pytorch
    python repro_kddcup_jax.py --quick                       # fast smoke test

Reference numbers, 1x H100 (jax 0.10.1, cuda12 wheels), kddcup09 fold 0.
Single ensemble member (``--n-estimators 1``), so timings are per member:

    backend  dtype  matmul     AUC      predict
    jax      bf16   -          0.7697   81 s
    jax      f32    default    0.8311   83 s     <- TF32 tensor cores
    jax      f32    highest    0.8312   108 s    <- true fp32
    pytorch  bf16   -          0.7287   8 s
    pytorch  f32    -          0.8312   -

bf16 costs ~0.10 AUC per member on BOTH backends here; fp32 recovers it and
the two backends then agree to 1e-4 (0.8311 vs 0.8312).

Full 32-member presets (``--n-estimators 32``, the default):

    jax      bf16   default    0.7192   36 min
    jax      bf16   ensemble   0.7801   55 min
    pytorch  bf16   default    0.8262   ~2 min

Note the asymmetry: averaging 32 bf16 members *helps* PyTorch (0.7287 ->
0.8262) but *hurts* JAX (0.7697 -> 0.7192). Per-member accuracy is similar,
so the divergence is in how the ensemble combines bf16 members, not in the
forward pass.

Two things this dataset exposes:

* Precision. bf16 (7-bit mantissa) loses ~0.06 AUC here; TF32 (10-bit,
  what ``dtype=float32`` uses by default on Ampere+) recovers all of it,
  and true fp32 (23-bit) adds nothing beyond TF32. The dataset is 1.8%
  positive with 212 features and a wide dynamic range.
* Kernel choice, not arithmetic, dominates JAX runtime: the same bf16
  forward takes 81 s with the chunked ``flash`` scan and 13.7 s with fused
  cuDNN (``--attn cudnn``), which is why halving precision barely changes
  the total.

A machine reporting materially better *bf16* numbers than the above is the
interesting case; the f32 rows should agree everywhere.
"""

import argparse
import time

import numpy as np

TASK_ID = 363683  # TabArena kddcup09_appetency


def load_fold0(quick=False):
  """Returns (X_train, y_train, X_test, y_test) for repeat 0 / fold 0."""
  import openml

  task = openml.tasks.get_task(TASK_ID)
  ds = task.get_dataset()
  x, y, _, _ = ds.get_data(target=ds.default_target_attribute)
  split = task.download_split().split[0][0][0]
  x_train, y_train = x.iloc[split.train], y.iloc[split.train]
  x_test, y_test = x.iloc[split.test], y.iloc[split.test]
  if quick:
    rng = np.random.default_rng(0)
    itr = rng.permutation(len(x_train))[:3000]
    ite = rng.permutation(len(x_test))[:968]
    x_train, y_train = x_train.iloc[itr], y_train.iloc[itr]
    x_test, y_test = x_test.iloc[ite], y_test.iloc[ite]
  return x_train, y_train, x_test, y_test


def describe_env(backend, dtype_flag):  # pylint: disable=missing-function-docstring
  """Prints everything that could make two machines disagree."""
  import platform

  print(f"python      : {platform.python_version()}")
  if backend == "jax":
    import jax
    import jaxlib

    from tabfm.src.jax import model as tm

    dev = jax.devices()[0]
    print(f"jax/jaxlib  : {jax.__version__} / {jaxlib.__version__}")
    print(f"device      : {dev.device_kind} (platform={dev.platform}, "
          f"n={len(jax.devices())})")
    print(f"chunk sizes : row={tm.ROW_CHUNK_SIZE} col={tm.COL_CHUNK_SIZE} "
          f"ffn={tm.FFN_CHUNK_SIZE}")
    print(f"matmul prec : {jax.config.jax_default_matmul_precision}")
  else:
    import torch

    print(f"torch       : {torch.__version__}")
    print(f"device      : {torch.cuda.get_device_name(0)}"
          if torch.cuda.is_available() else "device      : cpu")
  print(f"dtype       : {dtype_flag}")


def build_model(backend, dtype_flag, attn):
  """Loads the v1.1.0 checkpoint for the requested backend/dtype."""
  if backend == "jax":
    import jax.numpy as jnp

    from tabfm.src.jax import tabfm_v1_1_0 as loader

    kwargs = {}
    if attn != "default":
      kwargs.update(col_attention_impl=attn, icl_attention_impl=attn)
    if dtype_flag == "f32":
      kwargs["dtype"] = jnp.float32
    print(f"attention   : {attn} (col/icl)"
          f"{'' if attn == 'default' else ', row=jax'}")
    return loader.load(model_type="classification", **kwargs)

  import torch

  from tabfm.src.pytorch import tabfm_v1_1_0 as loader

  return loader.load(
      model_type="classification",
      device="cuda" if torch.cuda.is_available() else None,
      dtype=None if dtype_flag == "f32" else torch.bfloat16,
  )


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--backend", choices=["jax", "pytorch"], default="jax")
  p.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
  p.add_argument("--attn", default="default",
                 help="jax only: 'default', 'flash', 'cudnn', 'jax'")
  p.add_argument("--presets", default="default,ensemble")
  p.add_argument("--n-estimators", type=int, default=None,
                 help="override the preset's member count (1 = fastest "
                      "apples-to-apples comparison)")
  p.add_argument("--matmul", choices=["default", "highest"], default="default",
                 help="jax float32 matmul precision: 'default' uses TF32 "
                      "tensor cores on Ampere+, 'highest' forces true fp32")
  p.add_argument("--quick", action="store_true",
                 help="3k-row context instead of the full 33k fold")
  args = p.parse_args()

  if args.backend == "jax" and args.matmul == "highest":
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")

  from sklearn.metrics import log_loss, roc_auc_score

  import tabfm

  print("=" * 68)
  describe_env(args.backend, args.dtype)
  x_train, y_train, x_test, y_test = load_fold0(args.quick)
  print(f"data        : train={x_train.shape} test={x_test.shape}"
        f"{' [QUICK subsample]' if args.quick else ''}")
  print("=" * 68)

  model = build_model(args.backend, args.dtype, args.attn)
  for preset in args.presets.split(","):
    kw = {"random_state": 0}
    if args.n_estimators is not None:
      kw["n_estimators"] = args.n_estimators
    est = (
        tabfm.TabFMClassifier.ensemble(model=model, **kw)
        if preset == "ensemble"
        else tabfm.TabFMClassifier(model=model, **kw)
    )
    t0 = time.time()
    est.fit(x_train, y_train)
    probs = est.predict_proba(x_test)
    elapsed = time.time() - t0
    y_idx = (np.asarray(y_test) == est.classes_[1]).astype(int)
    auc = roc_auc_score(y_idx, probs[:, 1])
    n_members = kw.get("n_estimators", 32)
    print(f"{args.backend:<8} {args.dtype:<5} matmul={args.matmul:<8} "
          f"{preset:<9} n_est={n_members:<3} "
          f"AUC={auc:.4f}  log-loss={log_loss(y_idx, probs[:, 1]):.5f}  "
          f"({elapsed:.0f}s)")


if __name__ == "__main__":
  main()
