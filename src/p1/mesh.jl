using CUDA, LinearAlgebra, SparseArrays, TOML, SHA

@inline function miehe_split(exx, eyy, exy, lam, mu, g)
    tr = exx + eyy
    mean = tr / 2
    dev = (exx - eyy) / 2
    r = hypot(dev, exy)
    a = max(mean + r, 0.0)
    b = max(mean - r, 0.0)
    m = (a + b) / 2
    factor = r == 0 ? 0.0 : (a - b) / (2r)
    px = m + factor * dev
    py = m - factor * dev
    pxy = factor * exy
    tp = max(tr, 0.0)
    tn = min(tr, 0.0)
    pp = lam / 2 * tp^2 + mu * (a^2 + b^2)
    pm = lam / 2 * tn^2 + mu * ((exx - px)^2 + (eyy - py)^2 + 2 * (exy - pxy)^2)
    (
        g * (lam * tp + 2mu * px) + lam * tn + 2mu * (exx - px),
        g * (lam * tp + 2mu * py) + lam * tn + 2mu * (eyy - py),
        2mu * (g * pxy + exy - pxy),
        pp,
        pm,
    )
end

include("grid.jl")

function buffers(g; gpu = true)
    gpu || error("This distribution contains the GPU solver only")
    dev = CuArray
    h = merge(
        g,
        (;
            xy = dev(g.xy),
            tri = dev(g.tri),
            G = dev(g.G),
            area = dev(g.area),
            volume = dev(g.volume),
            mass = dev(g.mass),
            C = dev(g.C),
            free = dev(g.free),
            rate = dev(g.rate),
            offsets = dev(g.offsets),
            adj = dev(g.adj),
        ),
    )
    (;
        g = h,
        localf = dev(zeros(2, 3, g.ne)),
        localp = dev(zeros(3, g.ne)),
        qp = dev(zeros(g.ne)),
        elastic = dev(zeros(g.ne)),
        fracture = dev(zeros(g.ne)),
        bounds = dev(zeros(g.ne)),
        f = dev(zeros(2, g.n)),
        phase = dev(zeros(g.n)),
    )
end

@inline function gather!(i, g, lf, lp, f, phase)
    fx = fy = fd = 0.0
    @inbounds for j in g.offsets[i]:(g.offsets[i + 1] - 1)
        k = g.adj[j]
        e = div(k - 1, 3) + 1
        a = (k - 1) % 3 + 1
        fx += lf[1, a, e]
        fy += lf[2, a, e]
        fd += lp[a, e]
    end
    @inbounds f[1, i] = fx
    @inbounds f[2, i] = fy
    @inbounds phase[i] = fd
    nothing
end

function gather_kernel!(g, lf, lp, f, phase)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= g.n && gather!(i, g, lf, lp, f, phase)
    return nothing
end
