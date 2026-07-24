"""Does sequence sharding actually win? TabArena GiveMeSomeCredit, fold 0.

Companion to tpu_validate_airfoil.py. airfoil has a 1002-row context, which is
~32x too small for seqpar to pay off: c_blk is rounded up to _MAB1_KEY_CHUNK
(2048), so each device pads its 63-row shard to a full chunk and the sharded
path runs SLOWER. GiveMeSomeCredit has a ~135k-row context -- comfortably past
the n_train >= 2048 * world break-even -- so this is where seqpar should win.

This also exercises TabFMClassifier._batch_forward, whose allgather fix is
otherwise untested (the airfoil runner only covers the regressor).

Reports ROC AUC and wall clock for the stock (single-device) and seqpar
(sequence-sharded) paths. Each path is guarded: the stock path may OOM on a
135k-row context, which is precisely the motivation for seqpar, so a failure
there is a result, not a crash.

Multi-host: launch on every worker at once.
  gcloud compute tpus tpu-vm ssh tabfm-mh --zone=us-central1-a --worker=all \
    --quiet --internal-ip --command="~/tabfm-venv/bin/python ~/tabfm/tpu_validate_gmsc.py"

Single host on one worker of a slice:
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost \
  python tpu_validate_gmsc.py --no-distributed
"""

import argparse
import sys
import time
import traceback

import numpy as np

TASK_ID = 363636  # TabArena GiveMeSomeCredit
SEED = 42


def _load_fold_0(task_id):
  """Returns (X_train, y_train, X_test, y_test) for repeat-0 / fold-0."""
  import openml  # pylint: disable=g-import-not-at-top

  task = openml.tasks.get_task(task_id)
  dataset = task.get_dataset()
  x, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
  split = task.download_split().split[0][0][0]
  return (
      x.iloc[split.train],
      y.iloc[split.train],
      x.iloc[split.test],
      y.iloc[split.test],
  )


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      "--distributed",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="join the JAX distributed runtime (required on a multi-host slice)",
  )
  ap.add_argument(
      "--n-estimators", type=int, default=1,
      help="ensemble members (default 1: this measures sharding, not accuracy)",
  )
  ap.add_argument(
      "--mode", choices=("both", "stock", "seqpar"), default="both",
      help="which prediction path(s) to time",
  )
  ap.add_argument(
      "--data-shards", type=int, default=None,
      help="use a 2-D (data x seqpar) mesh for the seqpar path: this many "
           "members run concurrently across the data axis, each "
           "sequence-sharded over the remaining devices",
  )
  args = ap.parse_args()

  import jax  # pylint: disable=g-import-not-at-top

  if args.distributed:
    jax.distributed.initialize()

  idx = jax.process_index()

  def emit(msg):
    print(f"[w{idx}] {msg}", flush=True)

  devs = jax.devices()
  emit(
      f"process_index={idx} process_count={jax.process_count()} "
      f"local_device_count={jax.local_device_count()} "
      f"device_count={len(devs)} kind={devs[0].device_kind}"
  )

  import tabfm  # pylint: disable=g-import-not-at-top
  from sklearn.metrics import roc_auc_score  # pylint: disable=g-import-not-at-top

  t0 = time.time()
  model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
  emit(f"checkpoint loaded in {time.time() - t0:.1f}s")

  t0 = time.time()
  x_train, y_train, x_test, y_test = _load_fold_0(TASK_ID)
  emit(
      f"fold 0: train={x_train.shape} test={x_test.shape} "
      f"(loaded in {time.time() - t0:.1f}s)"
  )

  # Break-even for sequence sharding: each device's shard must fill one
  # _MAB1_KEY_CHUNK, else it pads and the sharded path does extra work.
  from tabfm.src.jax import seqpar  # pylint: disable=g-import-not-at-top

  world = len(devs)
  n_train = x_train.shape[0]
  per_dev = n_train // world
  emit(
      f"context={n_train} world={world} rows/device={per_dev} "
      f"chunk={seqpar._MAB1_KEY_CHUNK} "  # pylint: disable=protected-access
      f"-> {'ABOVE' if per_dev >= seqpar._MAB1_KEY_CHUNK else 'BELOW'} "  # pylint: disable=protected-access
      "break-even"
  )

  clf = tabfm.TabFMClassifier(
      model=model, n_estimators=args.n_estimators, random_state=SEED
  )
  t0 = time.time()
  clf.fit(x_train, y_train)
  emit(f"fit in {time.time() - t0:.1f}s")

  y_bin = (np.asarray(y_test) == clf.classes_[1]).astype(int)
  results = {}

  def run(name, fn):
    """Times fn; an exception (e.g. OOM on the stock path) is a result."""
    try:
      t = time.time()
      proba = fn()
      dt = time.time() - t
      auc = roc_auc_score(y_bin, np.asarray(proba)[:, 1])
      results[name] = (auc, dt)
      emit(f"{name:<7} AUC={auc:.5f}  {dt:.1f}s")
    except Exception as exc:  # pylint: disable=broad-except
      results[name] = (None, None)
      emit(f"{name:<7} FAILED: {type(exc).__name__}: {str(exc)[:160]}")
      traceback.print_exc()

  if args.mode in ("both", "stock"):
    run("stock", lambda: clf.predict_proba(x_test))
  if args.mode in ("both", "seqpar"):
    if args.data_shards is not None:
      mesh_2d = seqpar.make_mesh_2d(args.data_shards)
      emit(f"seqpar 2-D mesh: data={args.data_shards} "
           f"seqpar={len(devs) // args.data_shards}")
      run("seqpar", lambda: seqpar.predict_proba(clf, x_test, mesh=mesh_2d))
    else:
      run("seqpar", lambda: seqpar.predict_proba(clf, x_test))

  if results.get("stock", (None,))[0] is not None and (
      results.get("seqpar", (None,))[0] is not None
  ):
    a_auc, a_t = results["stock"]
    b_auc, b_t = results["seqpar"]
    emit(
        f"VERDICT: seqpar {a_t / b_t:.2f}x vs stock "
        f"({a_t:.1f}s -> {b_t:.1f}s), dAUC={abs(a_auc - b_auc):.5f}"
    )
  return 0


if __name__ == "__main__":
  sys.exit(main())
