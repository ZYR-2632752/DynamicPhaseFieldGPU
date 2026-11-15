# Outputs and timing

`history.csv` has one row per accepted step. The P1 solvers retain a device-side
ring and transfer it in batches of at most 100 steps. An overflow is an error.
Q1 and H8 write their scalar diagnostics after each accepted step.

| Column | Meaning |
|---|---|
| `time_s`, `step` | Accepted time in seconds and step number |
| `reaction` | Axial support force for the bar; driven-boundary force for SENT, KW and H8 |
| `kinetic`, `elastic`, `fracture` | Current energies |
| `work` | Accumulated external work |
| `viscous` | Accumulated `eta/dt * volume * (d_new-d_old)^2` |
| `physical_balance` | Kinetic + stored + viscous energy - external work - initial energy |
| `accounting_error` | Balance after including the discrete curvature, constraint and mechanical terms |
| `phase_identity` | Residual of the fixed-displacement incremental phase-energy identity |
| `damage_max` | Largest nodal damage |
| `min_delta_step` | Smallest accepted nodal damage increment |
| `KKT` | Infinity norm of the diagonally scaled projected gradient |
| `CG`, `AS` | PCG and projected-Newton iteration counts for this step |

The branching plate has balanced prescribed tractions and no displacement support;
its `reaction` column is `NaN`, not a measured reaction. Its applied traction is
specified by `load_scale` in the example.

Time is always in seconds. Forces are in N. Bar and SENT use m and J; KW, branching
and H8 use mm and N mm. The 2D examples use unit out-of-plane thickness in their
respective length units. These are total modeled-domain energies; symmetry factors
are not applied automatically.

`final.vtk` is a legacy binary unstructured-grid file. Connectivity is zero-based
in the file; coordinates, damage and displacement use Float64. VTK binary blocks
use big-endian byte order. No displacement scale factor is applied.

`loop_wall_seconds` starts after initialization and device synchronization, and
ends after the last accepted step, history write and device synchronization. It
includes first-step compilation and diagnostic transfers, and excludes final VTK
export. Use it to inspect a run, not to reproduce the manuscript's reported GPU/CPU
speedups without the corresponding warm-up and sampling protocol.

`termination = "step_limit"` means the requested short run completed, not that the
physical end time was reached. `termination = "end_time"` identifies a complete
configured trajectory.
