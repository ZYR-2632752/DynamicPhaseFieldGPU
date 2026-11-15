using LinearAlgebra

function grid(p)
    nx, ny = p.nx, p.ny
    @assert iseven(nx) && nx >= 4 && ny >= 2
    @assert all(
        x -> isfinite(x) && x > 0,
        (
            p.length_m,
            p.height_m,
            p.lambda_Pa,
            p.shear_modulus_Pa,
            p.density_kg_m3,
            p.gc_N_m,
            p.ell_m,
            p.loading_rate_m_s,
        ),
    )
    @assert 0 < p.crack_length_m < p.length_m
    hx = p.length_m / nx
    hy = p.height_m / ny
    n = (nx + 1) * (ny + 1)
    ne = 2nx * ny
    xy = zeros(2, n)
    tri = zeros(Int32, 3, ne)
    G = zeros(2, 3, ne)
    area = zeros(ne)
    for j in 0:ny, i in 0:nx
        k = 1 + i + (nx + 1) * j
        xy[:, k] = [i * hx, j * hy]
    end
    e = 0
    for j in 0:(ny - 1), i in 0:(nx - 1)
        a = 1 + i + (nx + 1) * j
        b = a + 1
        c = b + nx + 1
        d = a + nx + 1
        for ids in (iseven(i + j) ? ((a, b, c), (a, c, d)) : ((a, b, d), (b, c, d)))
            e += 1
            tri[:, e] .= ids
            x = xy[:, collect(ids)]
            twice =
                (x[1, 2] - x[1, 1]) * (x[2, 3] - x[2, 1]) -
                (x[1, 3] - x[1, 1]) * (x[2, 2] - x[2, 1])
            @assert twice > 0
            area[e] = twice / 2
            G[:, :, e] =
                [
                    x[2, 2] - x[2, 3] x[2, 3] - x[2, 1] x[2, 1] - x[2, 2];
                    x[1, 3] - x[1, 2] x[1, 1] - x[1, 3] x[1, 2] - x[1, 1]
                ] / twice
        end
    end
    volume = zeros(n)
    counts = zeros(Int, n)
    for e in 1:ne, k in 1:3
        i = tri[k, e]
        volume[i] += area[e] / 3
        counts[i] += 1
    end
    offsets = Int32.(vcat(1, 1 .+ cumsum(counts)))
    adj = zeros(Int32, 3ne)
    slots = copy(offsets[1:(end - 1)])
    for e in 1:ne, k in 1:3
        i = tri[k, e]
        adj[slots[i]] = 3(e - 1) + k
        slots[i] += 1
    end
    free = ones(2, n)
    rate = zeros(2, n)
    top = (n - nx):n
    free[:, top] .= 0.0
    rate[2, top] .= p.loading_rate_m_s
    free[2, findall(i -> abs(xy[2, i]) < 1e-14 && xy[1, i] >= p.crack_length_m - 1e-14, 1:n)] .= 0.0
    lam = p.lambda_Pa
    mu = p.shear_modulus_Pa
    rho = p.density_kg_m3
    eta = p.eta_Pa_s
    gc = p.gc_N_m
    ell = p.ell_m
    # Element Gershgorin bound with lumped mechanical mass.
    D = [lam + 2mu lam 0.0; lam lam + 2mu 0.0; 0.0 0.0 mu]
    bounds = zeros(ne)
    for e in 1:ne
        B = zeros(3, 6)
        for k in 1:3
            B[1, 2k - 1] = G[1, k, e]
            B[2, 2k] = G[2, k, e]
            B[3, 2k - 1] = G[2, k, e]
            B[3, 2k] = G[1, k, e]
        end
        Ke = area[e] * B' * D * B
        bounds[e] = maximum(vec(sum(abs.(Ke); dims = 2))) / (rho * area[e] / 3)
    end
    dtmechanical = 2 / sqrt(maximum(bounds))
    @assert isapprox(sum(area), p.length_m * p.height_m; rtol = 1e-12) && minimum(volume) > 0
    (;
        nx,
        ny,
        n,
        ne,
        hx,
        hy,
        xy,
        tri,
        G,
        area,
        volume,
        mass = rho * volume,
        C = eta * volume,
        free,
        rate,
        offsets,
        adj,
        lam,
        mu,
        rho,
        eta,
        gc,
        ell,
        dtmechanical,
        loading_rate = p.loading_rate_m_s,
    )
end
