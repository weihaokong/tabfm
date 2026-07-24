"""Verify the 2-D mesh puts the seqpar axis intra-host.

Prints, for make_mesh_2d(data_shards), the process_index (host) at each
(data, seqpar) mesh coordinate, then checks that process_index is CONSTANT
along the seqpar axis (intra-host) and VARIES along the data axis (across
hosts). Run on all workers of the slice.
"""

import sys

import numpy as np


def main():
  import jax

  jax.distributed.initialize()
  if jax.process_index() != 0:
    return 0  # one reporter is enough; the mesh is global

  from tabfm.src.jax import seqpar

  for data_shards in (int(sys.argv[1]) if len(sys.argv) > 1 else 4, 2):
    mesh = seqpar.make_mesh_2d(data_shards)
    # mesh.devices is a (D, S) object array of Device; map to process_index.
    grid = np.vectorize(lambda d: d.process_index)(mesh.devices)
    print(f"\n=== make_mesh_2d({data_shards})  axes={mesh.axis_names} "
          f"shape={tuple(mesh.devices.shape)} ===", flush=True)
    print("process_index (host) at each [data, seqpar] cell:", flush=True)
    for d in range(grid.shape[0]):
      print(f"  data={d}: seqpar-> {list(grid[d])}", flush=True)
    # intra-host along seqpar axis (axis=1): every row constant?
    seqpar_intrahost = all(len(set(grid[d])) == 1 for d in range(grid.shape[0]))
    # across-host along data axis (axis=0): every column all-distinct?
    data_crosshost = all(
        len(set(grid[:, s])) == grid.shape[0] for s in range(grid.shape[1])
    )
    print(f"  seqpar axis intra-host (all-gather stays on one host): "
          f"{seqpar_intrahost}", flush=True)
    print(f"  data axis spans distinct hosts: {data_crosshost}", flush=True)

  return 0


if __name__ == "__main__":
  sys.exit(main())
