using LinearAlgebra

function mersel_q1_grid(cfg, nx, ny)
    nx > 1 && ny > 0 && iseven(ny) || error("Use a positive even transverse cell count")
    hx = cfg["L"] / nx
    hy = cfg["area"] / ny
    n = (nx + 1) * (ny + 1)
    cells = Matrix{Int}(undef, 4, nx * ny)
    volume = zeros(n)
    for j in 0:(ny - 1), i in 0:(nx - 1)
        e = i + 1 + nx * j
        a = i + 1 + (nx + 1) * j
        cells[:, e] = [a, a + 1, a + nx + 2, a + nx + 1]
        for k in cells[:, e]
            volume[k] += hx * hy / 4
        end
    end
    quad = [
        (;
            N = ((1 - x) * (1 - y), (1 + x) * (1 - y), (1 + x) * (1 + y), (1 - x) * (1 + y)) ./ 4,
            gx = (-(1 - y), (1 - y), (1 + y), -(1 + y)) ./ (2hx),
            gy = (-(1 - x), -(1 + x), (1 + x), (1 - x)) ./ (2hy),
        ) for x in (-1 / sqrt(3), 1 / sqrt(3)) for y in (-1 / sqrt(3), 1 / sqrt(3))
    ]
    free = ones(2, n)
    free[1, 1:(nx + 1):end] .= 0
    free[2, 1:(nx + 1)] .= 0
    upper = ones(n)
    upper[(nx + 1):(nx + 1):end] .= 0
    loads = zeros(2, n)
    for j in 0:ny
        loads[1, nx + 1 + (nx + 1) * j] = cfg["stress"] * hy * (j in (0, ny) ? 0.5 : 1.0)
    end
    (;
        nx,
        ny,
        n,
        hx,
        hy,
        cells,
        quad,
        volume,
        weight = hx * hy / 4,
        free,
        upper,
        loads,
        mass = cfg["rho"] * volume,
        C = cfg["eta_d"] * volume,
        E = cfg["E"],
        gc = cfg["Gc"],
        ell = cfg["ell"],
        kappa = cfg["kappa"],
    )
end

@inline function mersel_q1_split(exx, eyy, exy, E, degradation)
    mean = (exx + eyy) / 2
    dev = (exx - eyy) / 2
    radius = hypot(dev, exy)
    a = mean + radius
    b = mean - radius
    pa = max(a, 0.0)
    pb = max(b, 0.0)
    factor = radius == 0 ? 0.0 : (pa - pb) / (2radius)
    pxx = (pa + pb) / 2 + factor * dev
    pyy = (pa + pb) / 2 - factor * dev
    pxy = factor * exy
    (
        E * (exx + (degradation - 1) * pxx),
        E * (eyy + (degradation - 1) * pyy),
        E * (exy + (degradation - 1) * pxy),
        0.5E * (pa^2 + pb^2),
        0.5E * (min(a, 0.0)^2 + min(b, 0.0)^2),
    )
end

function make_grid(p)
    nx, ny = p.nx, p.ny
    eta, stress, kappa = p.eta_Pa_s, p.stress_Pa, p.kappa
    all(isfinite, (eta, stress, kappa)) && eta >= 0 && stress > 0 && 0 <= kappa <= 1 ||
        error("Invalid bar parameters")
    cfg = Dict(
        "L" => p.length_m,
        "area" => p.height_m,
        "E" => p.youngs_modulus_Pa,
        "rho" => p.density_kg_m3,
        "Gc" => p.gc_N_m,
        "ell" => p.ell_m,
        "eta_d" => eta,
        "kappa" => kappa,
        "stress" => stress,
    )
    all(k -> isfinite(cfg[k]) && cfg[k] > 0, ("L", "area", "E", "rho", "Gc", "ell")) ||
        error("Invalid bar material or geometry")
    isfinite(p.wave_step_factor) && 0 < p.wave_step_factor <= 1 || error("Invalid wave-step factor")
    g = mersel_q1_grid(cfg, nx, ny)
    g.free[2, (end - nx):end] .= 0.0
    xy = [k == 1 ? i * g.hx : j * g.hy for k in 1:2, (i, j) in [(i, j) for j in 0:ny for i in 0:nx]]
    D = diagm([g.E, g.E, g.E / 2])
    K = zeros(8, 8)
    for q in g.quad
        B = zeros(3, 8)
        for a in 1:4
            B[1, 2a - 1] = q.gx[a]
            B[2, 2a] = q.gy[a]
            B[3, 2a - 1] = q.gy[a]
            B[3, 2a] = q.gx[a]
        end
        K .+= g.weight .* (B' * D * B)
    end
    omega2 =
        (1 + kappa) * maximum(vec(sum(abs.(K); dims = 2))) / (p.density_kg_m3 * g.hx * g.hy / 4)
    dt = min(p.wave_step_factor * min(g.hx, g.hy) / sqrt(g.E / p.density_kg_m3), 2 / sqrt(omega2))
    # Coordinates are output metadata, not part of the device kernel argument.
    (; grid = g, xy, dt)
end
