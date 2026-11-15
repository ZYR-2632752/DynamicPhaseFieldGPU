using DynamicPhaseFieldGPU

const PARAMETERS = (
    # Use meshes/impact3d for the full mesh after extracting its archive.
    mesh = normpath(joinpath(@__DIR__, "..", "meshes", "h8_fixture")),
    outer_y_mm = 100.0,
    outer_z_mm = 100.0,
    impact_radius_mm = 25.0,

    # Material [MPa, tonne/mm^3]
    youngs_modulus_MPa = 190000.0,
    poisson_ratio = 0.3,
    density_tonne_mm3 = 8e-9,

    # Phase field [N/mm, mm, Pa s]
    gc_N_mm = 22.13,
    ell_mm = 1.0,
    eta_Pa_s = 0.001,
    kappa = 1e-6,

    # Impact [mm/s, s]
    impact_speed_mm_s = 16500.0,
    ramp_time_s = 0.5e-6,

    # Time [s] and output
    end_time_s = 141.4e-6,
    steps = 100,
    output_dir = "",
)

if abspath(PROGRAM_FILE) == @__FILE__
    run_example(:impact3d, PARAMETERS, ARGS)
end
