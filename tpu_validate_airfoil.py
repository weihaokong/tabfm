"""Validate the TabFM JAX backend on TPU against the GPU reference number.

Task: TabArena ``airfoil_self_noise`` (OpenML task 363612), repeat 0 / fold 0.
Reference (A100, ``TabFMRegressor.ensemble()``, random_state=0):

    RMSE ~= 0.797    R^2 ~= 0.987    MAE ~= 0.546

The checkpoint (``google/tabfm-1.0.0-jax``) is public -- no HF token required.

Multi-host note: on a pod slice (v5litepod-16 = 4 hosts x 4 chips) this MUST be
started on every worker at the same time -- a lone process blocks forever in
pod initialisation waiting for its peers. Launch with:

  gcloud compute tpus tpu-vm ssh tabfm-mh --zone=us-central1-a --worker=all \
    --command="~/tabfm-venv/bin/python ~/tabfm/tpu_validate_airfoil.py"

Multi-host works as of the _batch_forward allgather fix (see CHANGELOG below).
For a single-host run on one worker of a multi-host slice, use::

    TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
    TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost \
    python tpu_validate_airfoil.py --no-distributed --preset both

Without those overrides a lone process hangs forever: the VM metadata declares
TPU_PROCESS_BOUNDS=2,2,1 / TOPOLOGY=4x4, so libtpu waits to rendezvous all four
hosts over ICI.

Multi-host status (measured 2026-07-24, 4 processes / 16 chips): topology comes
up correctly (process_index 0..3, local_device_count=4, device_count=16) but
scoring dies in ``_combine_predictions``::

    ValueError: shapes (32,) and (128,501) not aligned

32 is n_estimators; 128 is 32 x process_count. Under a multi-process runtime the
jitted forward returns an ensemble-member axis inflated by process_count. The
jaxtyping TypeCheckError on ``_batch_forward``'s "B T_test L_out" return is the
same bug caught one frame earlier.

DO NOT "fix" this with JAXTYPING_DISABLE=1. That annotation is the only thing
preventing a silently wrong answer. Measured on the default preset:

    single host (correct)          RMSE=0.93245
    4 hosts, JAXTYPING_DISABLE=1   RMSE=1.0229   <-- wrong, no error raised

All four workers agreed exactly on the wrong number, so cross-worker agreement
is NOT a correctness signal. The inflated rows are not duplicates of the local
members -- if they were, the mean over 4E rows would equal the mean over E and
the numbers would match. They do not.

``--seqpar`` (the explicitly-sharded path) also fails multi-host, with
"CopyArrays only supports destination device list of the same size as the array
device lists" -- it places state on the 16-device mesh while the arrays live on
4 local devices. Both presets fail, both paths. Use single host.

Verified result (single host, 4x TPU v5 lite, 2026-07-24)::

    ensemble  RMSE=0.80531  R2=0.98656  MAE=0.55157   (ref 0.797 / 0.987 / 0.546)
    default   RMSE=0.93245  R2=0.98198  MAE=0.64099

``--mesh`` is EXPERIMENTAL and currently fails. Setting a global mesh makes
``classifier_and_regressor`` shard the data while the params stay on device 0;
replicating the params first (as ``seqpar.predict`` does) clears that error but
then OOMs on v5e -- "Attempting to allocate 384.00M, 367.94M free". The
supported multi-device entry point in this repo is ``seqpar.predict``, which
``tpu_validate_seqpar.py`` already exercises; prefer that over ``--mesh``.
"""

import argparse
import importlib.util
import pathlib
import sys
import time

import numpy as np

TASK_ID = 363612
SEED = 0

# A100 reference for TabFMRegressor.ensemble() on this fold.
REFERENCE = {"rmse": 0.797, "r2": 0.987, "mae": 0.546}

_HERE = pathlib.Path(__file__).resolve().parent


def _load_example_module():
  """Imports examples/tabarena_regression_example.py by path.

  Loading the shipped example rather than re-implementing the split keeps the
  data handling bit-identical to what the GPU reference number was produced
  with. ``examples/`` is not a package, hence the by-path import.
  """
  path = _HERE / "examples" / "tabarena_regression_example.py"
  spec = importlib.util.spec_from_file_location("_tabarena_reg_example", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      "--distributed",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="join the JAX distributed runtime (required on a multi-host slice; "
           "pass --no-distributed for a single-host TPU or a CPU smoke test)",
  )
  ap.add_argument(
      "--preset",
      choices=("ensemble", "default", "both"),
      default="ensemble",
      help="which TabFMRegressor preset to score (default: ensemble, the one "
           "the reference number is for)",
  )
  ap.add_argument(
      "--mesh",
      action="store_true",
      help="EXPERIMENTAL / known-failing: install a data-parallel mesh over "
           "every chip and use the sharded batch path. OOMs on v5e (see the "
           "module docstring); the supported multi-device path is "
           "seqpar.predict, covered by tpu_validate_seqpar.py",
  )
  ap.add_argument(
      "--tol",
      type=float,
      default=0.02,
      help="absolute RMSE tolerance vs the GPU reference (default: 0.02)",
  )
  ap.add_argument(
      "--n-estimators",
      type=int,
      default=None,
      help="override n_estimators (diagnostic: with 4 processes the member "
           "axis comes back n_estimators * process_count)",
  )
  ap.add_argument(
      "--batch-size",
      type=int,
      default=None,
      help="ensemble members per forward pass. The library default is 1, so "
           "the ensemble preset runs 32 sequential single-member passes; this "
           "is also why the 'data' mesh axis has nothing to shard.",
  )
  ap.add_argument(
      "--data-shards",
      type=int,
      default=None,
      help="with --seqpar, use a 2-D (data x seqpar) mesh: this many members "
           "run concurrently across the data axis, each sequence-sharded over "
           "the remaining devices. Must divide the device count.",
  )
  ap.add_argument(
      "--seqpar",
      action="store_true",
      help="predict via tabfm.src.jax.seqpar (explicitly sharded over the "
           "mesh) instead of the stock single-process predict; this is the "
           "supported multi-host path",
  )
  ap.add_argument(
      "--topology-only",
      action="store_true",
      help="print the device topology and exit; no checkpoint, no HF token "
           "needed (use this as the pod sanity check)",
  )
  args = ap.parse_args()

  import jax  # imported after argparse so --help works without a TPU

  if args.distributed:
    # Cloud TPU: coordinator address / process count are auto-detected.
    jax.distributed.initialize()

  idx = jax.process_index()

  def emit(msg):
    """Prints with a worker tag; --worker=all interleaves the four streams."""
    print(f"[w{idx}] {msg}", flush=True)

  devs = jax.devices()
  emit(
      f"process_index={idx} process_count={jax.process_count()} "
      f"local_device_count={jax.local_device_count()} "
      f"device_count={len(devs)} kind={devs[0].device_kind}"
  )

  if args.topology_only:
    return 0

  import tabfm

  t0 = time.time()
  model = tabfm.tabfm_v1_0_0_jax.load(model_type="regression")
  emit(f"checkpoint loaded in {time.time() - t0:.1f}s")

  example = _load_example_module()
  x_train, y_train, x_test, y_test = example._load_fold_0(TASK_ID)  # pylint: disable=protected-access
  emit(f"fold 0: train={x_train.shape} test={x_test.shape}")

  presets = ("ensemble", "default") if args.preset == "both" else (args.preset,)
  builders = {
      "ensemble": tabfm.TabFMRegressor.ensemble,
      "default": tabfm.TabFMRegressor,
  }

  def _score_seqpar(reg):
    """Fits, then predicts via the explicitly-sharded seqpar path."""
    from sklearn.metrics import (  # pylint: disable=g-import-not-at-top, g-multiple-import
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from tabfm.src.jax import seqpar  # pylint: disable=g-import-not-at-top

    reg.fit(x_train, y_train)
    if args.data_shards is not None:
      # 2-D mesh: members batched over the data axis, sequence over seqpar.
      mesh_2d = seqpar.make_mesh_2d(args.data_shards)
      emit(f"seqpar 2-D mesh: data={args.data_shards} "
           f"seqpar={len(jax.devices()) // args.data_shards}")
      pred = np.asarray(seqpar.predict(reg, x_test, mesh=mesh_2d),
                        dtype=float).ravel()
    else:
      pred = np.asarray(seqpar.predict(reg, x_test), dtype=float).ravel()
    return (
        mean_squared_error(y_test, pred) ** 0.5,
        r2_score(y_test, pred),
        mean_absolute_error(y_test, pred),
    )

  def score_all():
    out = {}
    for name in presets:
      kwargs = {"model": model, "random_state": SEED}
      if args.n_estimators is not None:
        kwargs["n_estimators"] = args.n_estimators
      if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
      reg = builders[name](**kwargs)
      t = time.time()
      if args.seqpar:
        rmse, r2, mae = _score_seqpar(reg)
      else:
        rmse, r2, mae = example._evaluate(  # pylint: disable=protected-access
            reg, x_train, y_train, x_test, y_test
        )
      out[name] = (rmse, r2, mae, time.time() - t)
    return out

  if args.mesh:
    # Auto axis types keep sharding inference implicit; the Explicit default in
    # jax 0.10 would push sharding-in-types through tabfm's jitted predict step.
    # Order devices so each host's local chips are contiguous in the 1-D mesh.
    # jax.make_mesh's default flat order interleaves hosts, and pjit rejects
    # host-local inputs unless one host's devices form a contiguous subcube.
    ordered = sorted(devs, key=lambda d: (d.process_index, d.id))
    mesh = jax.sharding.Mesh(np.array(ordered, dtype=object), ("data",))
    # Setting a mesh only makes classifier_and_regressor shard the *data*; the
    # params stay on device 0 and jit then rejects the mismatch. Replicate the
    # model state across the mesh first, exactly as seqpar.predict does
    # (tabfm/src/jax/seqpar.py: device_put(state, NamedSharding(mesh, P()))).
    from flax import nnx  # pylint: disable=g-import-not-at-top
    from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top
    from jax.sharding import NamedSharding, PartitionSpec  # pylint: disable=g-import-not-at-top, g-multiple-import

    graphdef, state = nnx.split(model)
    if jax.process_count() > 1:
      # device_put cannot move a process-local array onto a sharding spanning
      # all global devices ("CopyArrays only supports destination device list
      # of the same size as the array device lists"). Every process holds an
      # identical full copy of the params, so lift them to a globally
      # replicated array instead.
      state = multihost_utils.host_local_array_to_global_array(
          state, mesh, PartitionSpec()
      )
    else:
      state = jax.device_put(state, NamedSharding(mesh, PartitionSpec()))
    model = nnx.merge(graphdef, state)  # score_all() reads this via closure
    emit(f"mesh: data={len(devs)} (sharded batch path, params replicated)")
    with jax.sharding.set_mesh(mesh):
      results = score_all()
  else:
    emit("no mesh: unsharded single-device path (per process)")
    results = score_all()

  ok = True
  for name, (rmse, r2, mae, dt) in results.items():
    emit(f"{name:<9} RMSE={rmse:.5g} R2={r2:.5f} MAE={mae:.5g} ({dt:.1f}s)")
    if name == "ensemble":
      delta = abs(rmse - REFERENCE["rmse"])
      verdict = "PASS" if delta <= args.tol else "FAIL"
      ok = ok and delta <= args.tol
      emit(
          f"{verdict}: ensemble RMSE {rmse:.5g} vs GPU reference "
          f"{REFERENCE['rmse']} (delta={delta:.4g}, tol={args.tol})"
      )

  return 0 if ok else 1


if __name__ == "__main__":
  sys.exit(main())
