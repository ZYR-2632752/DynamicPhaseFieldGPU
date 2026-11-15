function kw_grid(p, parameters)
    n = length(p.mass)
    ne = length(p.volume)
    xy = permutedims(p.nodes)
    tri = permutedims(p.tri)
    G = permutedims(p.G, (3, 2, 1))
    area = copy(p.volume)
    rho = parameters.density_tonne_mm3
    eta = parameters.eta_Pa_s * 1e-6
    gc = parameters.gc_N_mm
    ell = parameters.ell_mm
    kappa = parameters.bound_reference_kappa
    E = parameters.youngs_modulus_MPa
    nu = parameters.poisson_ratio
    @assert all(
        x -> isfinite(x) && x > 0, (rho, E, gc, ell, parameters.mesh_mass_density_tonne_mm3)
    )
    @assert isfinite(nu) && -1 < nu < 0.5
    lam = E * nu / ((1 + nu) * (1 - 2nu))
    mu = E / (2 * (1 + nu))
    volume = p.mass / parameters.mesh_mass_density_tonne_mm3
    counts = zeros(Int, n)
    for i in tri
        counts[i] += 1
    end
    offsets = Int32.(vcat(1, 1 .+ cumsum(counts)))
    adj = zeros(Int32, 3ne)
    slots = copy(offsets[1:(end - 1)])
    for e in 1:ne, a in 1:3
        i = tri[a, e]
        adj[slots[i]] = 3(e - 1) + a
        slots[i] += 1
    end
    free = ones(2, n)
    free[1, p.fixed_x] .= 0.0
    free[2, p.fixed_y] .= 0.0
    rate = zeros(2, n)
    rate[1, p.impact] .= 1.0 # unit prescribed displacement shape
    D = [lam + 2mu lam 0.0; lam lam + 2mu 0.0; 0.0 0.0 mu]
    bounds = zeros(ne)
    workers = zeros(Int, ne)
    Threads.@threads :static for e in 1:ne
        workers[e] = Threads.threadid()
        B = zeros(3, 6)
        for a in 1:3
            B[1, 2a - 1] = G[1, a, e]
            B[2, 2a] = G[2, a, e]
            B[3, 2a - 1] = G[2, a, e]
            B[3, 2a] = G[1, a, e]
        end
        bounds[e] = maximum(vec(sum(abs.(area[e] * (B' * D * B)); dims = 2))) / (rho * area[e] / 3)
    end
    dtmechanical = 2 / sqrt((1 + kappa) * maximum(bounds))
    @assert minimum(volume) > 0 && isfinite(dtmechanical) && dtmechanical > 0
    (;
        n,
        ne,
        xy,
        tri,
        G,
        area,
        volume,
        mass = p.mass .* (rho / parameters.mesh_mass_density_tonne_mm3),
        C = eta * volume,
        free,
        rate,
        offsets,
        adj,
        rho,
        eta,
        gc,
        ell,
        kappa,
        lam,
        mu,
        dtmechanical,
        preprocess_workers = Tuple(sort(unique(workers))),
    )
end

function kw_loading(t; speed_mm_s, ramp_s)
    @assert isfinite(t) && t >= 0
    v0 = speed_mm_s
    ramp = ramp_s
    (
        t < ramp ? 0.5v0 * t * t / ramp : v0 * (t - 0.5ramp),
        v0 * min(t / ramp, 1.0),
        t < ramp ? v0 / ramp : 0.0,
    )
end
