"""Validation runner for TabFM JAX seqpar on a single-host TPU VM.

Setup on the TPU VM (e.g. v5litepod-8):

  pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  git clone -b pr/pytorch-seqpar https://github.com/weihaokong/tabfm && cd tabfm
  pip install -e .[jax] absl-py pytest openml scikit-learn
  python -m pytest tabfm/src/jax/seqpar_test.py -q     # unit tests on real chips
  python tpu_validate_seqpar.py                        # this script
  python tpu_validate_seqpar.py --fold0                # + real-data check (openml)

What it does:
  1. Prints the device topology it sees.
  2. Synthetic regression: sharded seqpar.predict over all chips vs the stock
     single-device predict -- reports RMSE of both + max prediction diff.
  3. (--fold0) TabArena GiveMeSomeCredit fold 0 (135k-row context):
     sharded predict_proba vs stock, with timings.
"""

import argparse
import time

import numpy as np


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--fold0", action="store_true")
  p.add_argument("--distributed", action="store_true",
                 help="multi-host slice: join the distributed runtime "
                      "(run this script on every worker simultaneously)")
  args = p.parse_args()

  import jax

  if args.distributed:
    jax.distributed.initialize()  # Cloud TPU: coordinator auto-detected
  import tabfm
  from tabfm.src.jax import seqpar

  print(f"devices: {len(jax.devices())} x {jax.devices()[0].device_kind} "
        f"(process_count={jax.process_count()})")

  # --- 2. synthetic regression, sharded vs stock ---
  rng = np.random.default_rng(0)
  n_tr, n_te, f = 20_000, 1_000, 20
  X = rng.standard_normal((n_tr + n_te, f)).astype(np.float32)
  w = np.random.default_rng(1).standard_normal(f)
  y = X @ w + 0.1 * rng.standard_normal(n_tr + n_te)

  model = tabfm.tabfm_v1_0_0_jax.load(model_type="regression")
  reg = tabfm.TabFMRegressor(model=model, n_estimators=2, random_state=0)
  reg.fit(X[:n_tr], y[:n_tr])

  t0 = time.time(); ref = reg.predict(X[n_tr:]); t_ref = time.time() - t0
  t0 = time.time(); out = seqpar.predict(reg, X[n_tr:]); t_warm = time.time() - t0
  t0 = time.time(); out = seqpar.predict(reg, X[n_tr:]); t_out = time.time() - t0

  def rmse(p_):
    return float(np.sqrt(np.mean((y[n_tr:] - p_) ** 2)))

  d = np.abs(out - ref)
  print(f"[synthetic] stock 1-dev: rmse={rmse(ref):.4f} t={t_ref:.1f}s")
  print(f"[synthetic] seqpar {len(jax.devices())}-dev: rmse={rmse(out):.4f} "
        f"t={t_out:.1f}s (first call {t_warm:.1f}s incl. compile)")
  print(f"[synthetic] pred diff: max={d.max():.5f} mean={d.mean():.6f} "
        f"corr={np.corrcoef(out, ref)[0, 1]:.6f}")

  if not args.fold0:
    return

  # --- 3. real data: GiveMeSomeCredit fold 0 ---
  import openml
  from sklearn.metrics import roc_auc_score

  task = openml.tasks.get_task(363636)
  ds = task.get_dataset()
  Xd, yd, _, _ = ds.get_data(target=ds.default_target_attribute)
  split = task.download_split().split[0][0][0]
  Xtr, ytr = Xd.iloc[split.train], yd.iloc[split.train]
  Xte, yte = Xd.iloc[split.test], yd.iloc[split.test]

  model_c = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
  clf = tabfm.TabFMClassifier(model=model_c, n_estimators=1, random_state=42)
  clf.fit(Xtr, ytr)
  y_bin = (np.asarray(yte) == clf.classes_[1]).astype(int)

  t0 = time.time(); refp = clf.predict_proba(Xte); t_ref = time.time() - t0
  t0 = time.time(); outp = seqpar.predict_proba(clf, Xte); t_out = time.time() - t0
  d = np.abs(outp - refp)
  print(f"[fold0] stock 1-dev: AUC={roc_auc_score(y_bin, refp[:, 1]):.5f} t={t_ref:.1f}s")
  print(f"[fold0] seqpar: AUC={roc_auc_score(y_bin, outp[:, 1]):.5f} t={t_out:.1f}s")
  print(f"[fold0] diff: max|dP|={d.max():.4f} mean={d.mean():.5f} "
        f"corr={np.corrcoef(outp[:, 1], refp[:, 1])[0, 1]:.5f}")


if __name__ == "__main__":
  main()


def bench_splash():
  """mea vs splash on an ICL-like shape, plus stock fold-0-scale timing.

  Run as:  python -c "import tpu_validate_seqpar as t; t.bench_splash()"
  """
  import time
  import jax, jax.numpy as jnp
  from tabfm.src.jax import memory_efficient_attention as mea
  from tabfm.src.jax import model as tabfm_model
  from flax import nnx

  key = jax.random.PRNGKey(0)
  nh, hd, tq, tkv = 8, 256, 16384, 65536   # ICL-like, sized for 16GB HBM
  q = jax.random.normal(key, (1, tq, nh, hd), jnp.bfloat16) * 0.02
  k = jax.random.normal(key, (1, tkv, nh, hd), jnp.bfloat16) * 0.02

  def bench(fn, n=3):
    out = jax.block_until_ready(fn())
    t0 = time.time()
    for _ in range(n):
      out = jax.block_until_ready(fn())
    return (time.time() - t0) / n, out

  f_mea = jax.jit(lambda: mea.dot_product_attention_multihead(
      query=q, key=k, value=k, dtype=__import__("numpy").dtype("bfloat16"),
      enable_dropout=False, query_chunk_size=128, key_chunk_size=128))
  t_mea, ref = bench(f_mea)
  print(f"mea (128-chunks):  {t_mea*1000:8.1f} ms")

  mha = tabfm_model.MultiheadAttention(
      embed_dim=nh * hd, num_heads=nh,
      attention_impl=tabfm_model.AttentionImplementation.SPLASH,
      dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
  # Compare kernels only: bypass projections by calling the splash branch via
  # a raw kernel invocation on the same q/k/v.
  from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sak
  from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sam
  kernel = sak.make_splash_mha(
      sam.MultiHeadMask([sam.FullMask((tq, tkv))] * nh),
      block_sizes=sak.BlockSizes(block_q=128, block_kv=128),
      head_shards=1, q_seq_shards=1)
  f_spl = jax.jit(lambda: jax.vmap(kernel)(
      q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
      k.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3))
  t_spl, out = bench(f_spl)
  import numpy as np
  d = float(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32)).max())
  print(f"splash:            {t_spl*1000:8.1f} ms   ({t_mea/t_spl:.1f}x, max diff {d:.4f})")
