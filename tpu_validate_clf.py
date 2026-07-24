"""Validate TabFMClassifier's STOCK multi-host predict path.

Closes a gap the other runners leave open. The allgather fix in _batch_forward
(the data_sharding guard) was applied to both TabFMRegressor and
TabFMClassifier, but only the regressor's copy was measured multi-host (airfoil,
stock path). The classifier's copy was only ever exercised via seqpar (which
bypasses _batch_forward) or via the stock path on GiveMeSomeCredit (which OOMs
before completing). This runs the classifier's STOCK path -- predict_proba ->
_batch_forward -- on a small-context classification task so it fits unsharded on
all 16 chips.

Task: TabArena maternal_health_risk (OpenML 363685), 676-row context, 3 classes.
The check is exact: multi-host predictions must equal single-host predictions.
If the allgather still inflated the member axis, they would differ or crash.

Multi-host (all workers at once):
  gcloud compute tpus tpu-vm ssh tabfm-mh --zone=us-central1-a --worker=all \
    --quiet --internal-ip --command="~/tabfm-venv/bin/python ~/tabfm/tpu_validate_clf.py"

Single host on one worker:
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
  TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost \
  python tpu_validate_clf.py --no-distributed
"""

import argparse
import hashlib
import sys

import numpy as np

TASK_ID = 363685
SEED = 0


def _load_fold_0(task_id):
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
      f"local_device_count={jax.local_device_count()} device_count={len(devs)}"
  )

  import tabfm  # pylint: disable=g-import-not-at-top
  from sklearn.metrics import accuracy_score, roc_auc_score  # pylint: disable=g-import-not-at-top, g-multiple-import

  model = tabfm.tabfm_v1_0_0_jax.load(model_type="classification")
  x_train, y_train, x_test, y_test = _load_fold_0(TASK_ID)
  emit(f"fold 0: train={x_train.shape} test={x_test.shape} "
       f"classes={y_train.nunique()}")

  clf = tabfm.TabFMClassifier(model=model, random_state=SEED)
  clf.fit(x_train, y_train)

  # Stock path: predict_proba -> _batch_forward (the fixed code).
  proba = np.asarray(clf.predict_proba(x_test), dtype=float)
  pred = clf.classes_[np.argmax(proba, axis=1)]
  acc = accuracy_score(y_test, pred)
  try:
    auc = roc_auc_score(y_test, proba, multi_class="ovr")
  except Exception:  # pylint: disable=broad-except
    auc = float("nan")

  # Exact fingerprint of the probability matrix, so a multi-host run can be
  # compared bit-for-bit against a single-host run. If the member axis were
  # still inflated by process_count, this would change (or predict_proba would
  # have raised).
  digest = hashlib.md5(
      np.round(proba, 6).tobytes()
  ).hexdigest()[:12]
  emit(f"stock predict_proba: shape={proba.shape} acc={acc:.5f} "
       f"auc_ovr={auc:.5f} proba_md5={digest}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
