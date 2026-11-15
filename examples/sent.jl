using DynamicPhaseFieldGPU

const PARAMETERS = (
    # Half-plate geometry and P1 mesh [m]
    length_m = 0.1,
    height_m = 0.05,
    crack_length_m = 0.05,
    nx = 214,
    ny = 106,

    # Material [Pa, kg/m^3, m/s]
    lambda_Pa = 120e9,
    shear_modulus_Pa = 80e9,
    density_kg_m3 = 7000.0,
    c_R_m_s = 3135.23,

    # Phase field [N/m, m, Pa s]
    gc_N_m = 2700.0,
    ell_m = 0.001,
    eta_Pa_s = 1000.0,
    kappa = 1e-6,

    # Prescribed top-edge velocity [m/s]
    loading_rate_m_s = 0.01,

    # Time [s] and output
    end_time_s = 7e-3,
    steps = 100,
    output_dir = "",
)

if abspath(PROGRAM_FILE) == @__FILE__
    run_example(:sent, PARAMETERS, ARGS)
end
