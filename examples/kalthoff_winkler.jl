using DynamicPhaseFieldGPU

const PARAMETERS = (
    # Geometry and boundary nodes are stored in this mesh.
    mesh = normpath(joinpath(@__DIR__, "..", "meshes", "kw")),
    mesh_mass_density_tonne_mm3 = 8e-9,

    # Material [MPa, tonne/mm^3, mm/s]
    youngs_modulus_MPa = 190000.0,
    poisson_ratio = 0.3,
    density_tonne_mm3 = 8e-9,
    c_R_mm_s = 2802970.0,

    # Phase field [N/mm, mm, Pa s]; archived configuration
    gc_N_mm = 22.13,
    ell_mm = 0.195,
    eta_Pa_s = 19.2,
    kappa = 1e-6,

    # Impact [mm/s, s]
    impact_speed_mm_s = 16500.0,
    ramp_time_s = 1e-6,

    # Time [s] and output
    bound_reference_kappa = 1e-7,  # Preserve the archived bound normalization.
    end_time_s = 100e-6,
    steps = 100,
    output_dir = "",
)

if abspath(PROGRAM_FILE) == @__FILE__
    run_example(:kw, PARAMETERS, ARGS)
end
