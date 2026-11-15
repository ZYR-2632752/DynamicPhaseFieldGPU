using DynamicPhaseFieldGPU

const PARAMETERS = (
    # Mirrored P1 mesh and initial damage field
    mesh = normpath(joinpath(@__DIR__, "..", "meshes", "branching")),

    # Material [MPa, tonne/mm^3, mm/s]
    youngs_modulus_MPa = 32000.0,
    poisson_ratio = 0.2,
    density_tonne_mm3 = 2.45e-9,
    c_R_mm_s = 2125.21e3,

    # Phase field [N/mm, mm, Pa s]; Gc and ell must match the seed.
    gc_N_mm = 0.003,
    ell_mm = 0.5,
    eta_Pa_s = 0.0,
    kappa = 1e-6,

    # Balanced edge tractions; scale 1 applies 1 MPa.
    load_scale = 1.0,

    # Time [s] and output
    end_time_s = 77.8e-6,
    steps = 100,
    output_dir = "",
)

if abspath(PROGRAM_FILE) == @__FILE__
    run_example(:branching, PARAMETERS, ARGS)
end
