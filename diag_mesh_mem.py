"""Where does the ~15.4G go in the sharded (--mesh) path?

The mesh path OOMs with an identical signature regardless of batch_size or chip
count (384.00M requested, ~367M free), so the culprit is a fixed allocation, not
batch padding. This snapshots HBM after each stage to find it.

Run single-host on one worker (fast, reproduces the same OOM):

  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost \
  python diag_mesh_mem.py
"""

import sys

import numpy as np


def gb(n):
  return f"{n / 1e9:6.2f}G"


def main():
  import jax

  d0 = jax.local_devices()[0]

  def snap(label):
    s = d0.memory_stats() or {}
    used = s.get("bytes_in_use", 0)
    peak = s.get("peak_bytes_in_use", 0)
    limit = s.get("bytes_limit", 0)
    print(
        f"{label:<34} used={gb(used)} peak={gb(peak)} limit={gb(limit)}",
        flush=True,
    )
    return used

  print(f"devices={len(jax.devices())} kind={jax.devices()[0].device_kind}",
        flush=True)
  snap("00 baseline")

  import tabfm
  from tabfm.src.jax import seqpar  # pylint: disable=unused-import

  model = tabfm.tabfm_v1_0_0_jax.load(model_type="regression")
  snap("01 after checkpoint load")

  # Size of the parameter state itself.
  from flax import nnx
  graphdef, state = nnx.split(model)
  leaves = jax.tree_util.tree_leaves(state)
  total = sum(int(np.prod(x.shape)) * x.dtype.itemsize for x in leaves)
  print(f"param leaves={len(leaves)} total={gb(total)}", flush=True)
  snap("02 after nnx.split")

  devs = jax.devices()
  ordered = sorted(devs, key=lambda d: (d.process_index, d.id))
  mesh = jax.sharding.Mesh(np.array(ordered, dtype=object), ("data",))
  from jax.sharding import NamedSharding, PartitionSpec

  if jax.process_count() > 1:
    from jax.experimental import multihost_utils
    state2 = multihost_utils.host_local_array_to_global_array(
        state, mesh, PartitionSpec()
    )
  else:
    state2 = jax.device_put(state, NamedSharding(mesh, PartitionSpec()))
  snap("03 after param replication")

  model2 = nnx.merge(graphdef, state2)
  snap("04 after nnx.merge")

  # Now the actual predict, under the mesh, on a tiny input.
  import importlib.util
  import pathlib
  p = pathlib.Path(__file__).resolve().parent / "examples" / "tabarena_regression_example.py"
  spec = importlib.util.spec_from_file_location("_ex", p)
  ex = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(ex)
  x_train, y_train, x_test, y_test = ex._load_fold_0(363612)  # pylint: disable=protected-access
  snap("05 after data load")

  # Match the real failing config: ensemble preset (n_estimators=32 + NNLS).
  n_est = int(sys.argv[1]) if len(sys.argv) > 1 else 32
  bs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
  print(f"CONFIG ensemble n_estimators={n_est} batch_size={bs}", flush=True)
  reg = tabfm.TabFMRegressor.ensemble(
      model=model2, random_state=0, n_estimators=n_est, batch_size=bs
  )

  import traceback
  try:
    with jax.sharding.set_mesh(mesh):
      snap("06 inside set_mesh, before fit")
      reg.fit(x_train, y_train)
      snap("07 after fit UNDER MESH")
      _ = reg.predict(x_test)
      snap("08 after predict UNDER MESH")
  except Exception as exc:  # pylint: disable=broad-except
    snap("XX at failure")
    print(f"FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    tb = traceback.format_exc().splitlines()
    for line in [l for l in tb if "classifier_and_regressor" in l or "diag_mesh" in l][-6:]:
      print("  " + line.strip(), flush=True)

  return 0


if __name__ == "__main__":
  sys.exit(main())
