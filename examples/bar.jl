using DynamicPhaseFieldGPU

const PARAMETERS = (
    # Geometry and mesh [m]
    length_m = 1.0,
    height_m = 0.05,
    nx = 400,
    ny = 20,

    # Material [Pa, kg/m^3]; zero Poisson ratio
    youngs_modulus_Pa = 190e9,
    density_kg_m3 = 8000.0,

    # Phase field [N/m, m, Pa s]
    gc_N_m = 22130.0,
    ell_m = 0.0075,
    eta_Pa_s = 21.5,
    kappa = 1e-6,

    # Step traction at t = 0+ [Pa]
    stress_Pa = 150e6,

    # Time [s] and output
    wave_step_factor = 0.8,
    end_time_s = 600e-6,
    steps = 100,
    output_dir = "",  # Empty: timestamped folder under outputs/.
)

if abspath(PROGRAM_FILE) == @__FILE__
    run_example(:bar, PARAMETERS, ARGS)
end
