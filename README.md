# DynamicPhaseFieldGPU

GPU finite-element solvers for viscous dynamic phase-field fracture. Displacements
are advanced by central differences. A backward Euler phase-field update enforces
the nodal bounds `d_old <= d <= 1` through projected Newton and Jacobi-PCG iterations.

There are five example entry points. Geometry or mesh paths, material constants,
phase-field parameters, loading, end time and output settings are grouped at the
top of each file; separate scripts are not needed for each
parameter value. Numerical time stepping requires an NVIDIA GPU. Mesh preparation
and file output run on the host; CPU comparison solvers are not included.

This is a release candidate assembled from the research archive. See
[validation and manuscript correspondence](docs/VALIDATION.md) before using it to
reproduce a published figure or quote a runtime.

## Setup

Use Julia 1.12 and an NVIDIA GPU on Linux or Windows. CUDA.jl selects its CUDA
runtime; follow the [CUDA.jl installation instructions](https://cuda.juliagpu.org/stable/installation/overview/)
if the driver is not detected. Persistent cooperative kernels require cooperative
launch support. Their block counts are bounded by the measured device residency.

From this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using CUDA; CUDA.versioninfo()'
julia --project=. examples/bar.jl --steps 10 --output outputs/bar_check
```

`Project.toml` and `Manifest.toml` specify the preparation environment. The original
paper's package lock was not retained in the recovered archive.

## Examples

| Entry point | Default input | Units |
|---|---|---|
| `examples/bar.jl` | 8,421 nodes; Q1 tensile bar | m, N, s |
| `examples/sent.jl` | 23,005 nodes; P1 SENT half-plate | m, N, s |
| `examples/kalthoff_winkler.jl` | 260,361 nodes; archived P1 impact mesh | mm, N, s |
| `examples/branching.jl` | 180,301 nodes; mirrored P1 plate and initial damage | mm, N, s |
| `examples/impact3d.jl` | 218-node H8 fixture for installation checks | mm, N, s |

Each entry runs the `steps` value in its parameter block (100 by default).
Use `--steps N` to override it or `--full`
to reach its configured end time. Output directories must be new.
Set `output_dir` in the example or pass `--output`; an empty value uses a timestamped
directory. The mesh files contain fixed geometry and boundary data. Changing the
branching example's `gc_N_mm` or `ell_mm` requires a matching initial damage seed.

```sh
julia --project=. examples/sent.jl --full --output outputs/sent
julia --project=. examples/kalthoff_winkler.jl --steps 10 --output outputs/kw_check
julia --project=. examples/branching.jl --full --output outputs/branching
julia --project=. examples/impact3d.jl --steps 10 --output outputs/h8_check
```

The small H8 fixture is not a resolved fracture benchmark. Download the
[2,318,269-node input](https://github.com/ZYR-2632752/DynamicPhaseFieldGPU/releases/download/v0.1.0-rc1/DynamicPhaseFieldGPU_Impact3D_mesh.tar.gz)
as `DynamicPhaseFieldGPU_Impact3D_mesh.tar.gz` and place it next to this repository:

```sh
tar -xzf ../DynamicPhaseFieldGPU_Impact3D_mesh.tar.gz -C .
julia --project=. examples/impact3d.jl --mesh meshes/impact3d --steps 10 --output outputs/h8_large_check
# After checking memory use and the short run:
julia --project=. examples/impact3d.jl --mesh meshes/impact3d --full --output outputs/h8_large
```

## Output

Each run writes:

- `history.csv`: time, force, energies, damage bounds, KKT residual and iteration counts.
- `final.vtk`: damage and displacement in binary VTK format, readable in ParaView.
- `run.toml`: parameters, mesh size, device/software versions, timestep, completion
  status and timing scope.

The CSV columns and units are documented in [OUTPUT.md](docs/OUTPUT.md). Scalar
histories are written during the loop; the final field is exported after timing.
The reported loop time includes first-step compilation. It is not the warmed-up
timing protocol used for the paper's speedup comparison.

To plot a history, install Matplotlib in your Python environment and run:

```sh
python -m pip install matplotlib
python scripts/plot_history.py outputs/bar_check
```

## Checks

The input and interface checks do not run a CPU fracture solver and do not require
CUDA:

```sh
julia --threads=4 test/runtests.jl
```

On the NVIDIA machine, run the five short GPU checks before a full simulation:

```sh
julia --project=. test/gpu_smoke.jl
```

KKT, box constraints, damage irreversibility and the incremental energy identity
are checked by the time integrators. A failed check aborts the run and records the
error in `run.toml`.

## Paper

Yongran Zhang, Ruizheng Zhang, Peng Zhang, Pengfei Cui and Keke Tang,
*Phase-field viscosity effects in dynamic fracture: A fully GPU-resident
bound-constrained solver*, manuscript.

No publication DOI is assigned to the manuscript.
Repository: [github.com/ZYR-2632752/DynamicPhaseFieldGPU](https://github.com/ZYR-2632752/DynamicPhaseFieldGPU).
