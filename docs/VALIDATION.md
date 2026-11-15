# Validation and manuscript correspondence

This package separates recovered code, newly connected example drivers and
executed checks. The runtime metadata records the input actually used.

| Example | Preparation status | Remaining comparison |
|---|---|---|
| Bar | Recovered Q1 quadrature and GPU mechanics/phase operators; Q1 matrix action connected to the shared persistent box solver | GPU trajectory and force/energy comparison after the connection |
| SENT | Recovered P1 spectral operators and GPU timestep driver; default 23,005-node grid matches the manuscript count | Full trajectory with the resolved package environment |
| KW | Recovered 260,361-node mesh, spectral split and 0.195 mm length scale | The manuscript lists 201,836 nodes, an orthogonal split and 0.39 mm. This archived configuration is not that final benchmark |
| Branching | Recovered 180,301-node mirrored mesh and seeded damage; balanced tractions connected to the shared P1 GPU driver | New GPU adapter has not been run; the archived input was labelled a candidate, so final-run identity remains unverified |
| 3D impact | Recovered B-bar H8 operators; 218-node fixture and separate 2,318,269-node mesh | Fixture is for execution checks only; validate the large-mesh trajectory and memory requirements on the target GPU |

The large H8 mesh's archived note describes a local refinement of a reconstructed
Rudshaug geometry. It is not the original authors' mesh. The saved metadata is
retained in its input manifest.

## Local checks

`test/runtests.jl` checks input hashes and dimensions, Q1 assembly/action agreement,
loading, nodal boundary masks, branching seed symmetry, H8 connectivity, output
encoding and rejection of existing output directories. It does not integrate a
CPU reference solution.

`test/gpu_smoke.jl` runs two accepted steps of each example and checks the recorded
KKT and irreversibility diagnostics and exported state. Passing it is necessary
for a release, but it does not establish mesh convergence, the final crack path or
agreement with every figure in the manuscript.

The preparation machine is Apple Silicon, without an NVIDIA device. GPU trajectories
and performance have not yet been checked for this package.

## Source changes

- Files were extracted from the research backup; CPU comparison solvers, cloud
  launchers, credentials and old run outputs were not included.
- Shared P1 and H8 kernels retain their mathematical update. The P1 cooperative
  launch now selects a block count within measured device residency.
- The Q1 bar uses its recovered four-point quadrature and operators. Its assembled
  element action was added as a descriptor of the existing persistent box solver;
  no projected-Newton or PCG convergence rule was changed.
- Branching reuses the constant-traction P1 GPU integrator. The initial acceleration
  includes the traction applied at `t = 0+`; the seeded damage is the lower bound.
- The five drivers add parameter intake, scoped timing, scalar history and VTK
  export. They do not change the manuscript or regenerate its figures.

This public release candidate still requires GPU checks and a trajectory comparison
for the new Q1/branching connections, resolution of the KW manuscript configuration,
and confirmation of the license and paper citation with the authors.
