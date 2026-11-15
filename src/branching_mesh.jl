module BranchingMesh
using LinearAlgebra, SHA, TOML
file_sha(path) = open(io -> bytes2hex(sha256(io)), path, "r")
function load_grid(folder, p)
    eta_Pa_s = p.eta_Pa_s
    @assert isfinite(eta_Pa_s) && eta_Pa_s >= 0
    m = TOML.parsefile(joinpath(folder, "manifest.toml"))
    n = Int(m["nodes"])
    ne = Int(m["elements"])
    for (name, digest) in m["sha256"]
        @assert file_sha(joinpath(folder, name)) == digest "Mesh identity differs: $name"
    end
    function arr(name, T, dims)
        @assert filesize(joinpath(folder, name)) == sizeof(T) * prod(dims)
        x = Array{T}(undef, dims)
        read!(joinpath(folder, name), x)
        x
    end
    xy = arr("xy.f64", Float64, (2, n))
    tri = arr("tri.i32", Int32, (3, ne))
    fext = arr("f100.f64", Float64, (2, n))
    @assert minimum(tri) >= 1 && maximum(tri) <= n && all(isfinite, xy)
    rho = p.density_tonne_mm3
    eta = Float64(eta_Pa_s) * 1e-6
    gc = p.gc_N_mm
    ell = p.ell_mm
    kappa = p.kappa
    E = p.youngs_modulus_MPa
    nu = p.poisson_ratio
    c_R = p.c_R_mm_s
    @assert all(x -> isfinite(x) && x > 0, (rho, E, gc, ell, c_R))
    @assert isfinite(nu) && -1 < nu < 0.5 && isfinite(kappa) && kappa >= 0
    lam = E * nu / ((1 + nu) * (1 - 2nu))
    mu = E / (2 * (1 + nu))
    G = zeros(2, 3, ne)
    area = zeros(ne)
    volume = zeros(n)
    counts = zeros(Int, n)
    for e in 1:ne
        a, b, c = tri[:, e]
        x1, y1 = xy[:, a]
        x2, y2 = xy[:, b]
        x3, y3 = xy[:, c]
        twice = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        @assert twice > 0
        area[e] = twice / 2
        G[:, :, e] = [y2 - y3 y3 - y1 y1 - y2; x3 - x2 x1 - x3 x2 - x1] / twice
        for i in (a, b, c)
            volume[i] += area[e] / 3
            counts[i] += 1
        end
    end
    offsets = Int32.(vcat(1, 1 .+ cumsum(counts)))
    adj = zeros(Int32, 3ne)
    slots = copy(offsets[1:(end - 1)])
    for e in 1:ne, a in 1:3
        i = tri[a, e]
        adj[slots[i]] = 3(e - 1) + a
        slots[i] += 1
    end
    D = [lam + 2mu lam 0.0; lam lam + 2mu 0.0; 0.0 0.0 mu]
    bounds = zeros(ne)
    Threads.@threads :static for e in 1:ne
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
    @assert minimum(volume) > 0 && isapprox(sum(area), 4000.0; rtol = 1e-11)
    @assert all(isfinite, fext) && all(iszero, fext[1, :])
    @assert isapprox(sum(max.(fext[2, :], 0.0)), 100.0; atol = 1e-10)
    @assert isapprox(sum(min.(fext[2, :], 0.0)), -100.0; atol = 1e-10)
    @assert maximum(abs.(vec(sum(fext; dims = 2)))) < 1e-10
    @assert abs(sum(xy[1, :] .* fext[2, :])) < 1e-8
    # No mechanical anchors: all nodes obey the free dynamic traction problem.
    g = (;
        n,
        ne,
        xy,
        tri,
        G,
        area,
        volume,
        mass = rho * volume,
        C = eta * volume,
        free = ones(2, n),
        rate = zeros(2, n),
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
        c_R,
        Pi_eta = eta * c_R / gc,
    )
    g, fext
end

function read_seed(g, fext, folder; load_scale = 1.0, mesh_manifest_sha256)
    meta = TOML.parsefile(joinpath(folder, "seed.toml"))
    @assert g.gc == meta["gc_N_mm"] && g.ell == meta["ell_mm"] "Gc and ell require a matching initial damage seed"
    @assert meta["status"] == "passed" && meta["nodes"] == g.n && meta["elements"] == g.ne
    @assert meta["mesh_manifest_sha256"] == mesh_manifest_sha256 "Initial field belongs to a different mesh"
    @assert file_sha(joinpath(folder, "d0.f64")) == meta["d0_sha256"]
    @assert file_sha(joinpath(folder, "q0.f64")) == meta["q0_sha256"]
    @assert filesize(joinpath(folder, "d0.f64")) == 8g.n
    @assert filesize(joinpath(folder, "q0.f64")) == 8g.ne
    d = Vector{Float64}(undef, g.n)
    read!(joinpath(folder, "d0.f64"), d)
    q = Vector{Float64}(undef, g.ne)
    read!(joinpath(folder, "q0.f64"), q)
    @assert all(isfinite, d) && all(0 .<= d .<= 1) && all(isfinite, q)
    # Constant traction is present at t=0; the initial acceleration must not be zero.
    seed = (;
        u = zeros(2, g.n),
        v = zeros(2, g.n),
        a = load_scale * fext ./ reshape(g.mass, 1, :),
        d,
        H = zeros(g.ne),
        cached = zeros(g.ne),
        t = 0.0,
        step = 0,
        initialK = 0.0,
        initialE = meta["initial_fracture_N_mm"],
        work = 0.0,
        visc = 0.0,
        curv = 0.0,
        normal = 0.0,
        supply = 0.0,
        mech = 0.0,
    )
    seed, q
end

end
