module H8Operators

using LinearAlgebra
include("spectral.jl")
using .MieheSpectral3D: miehe3d
export workspace, mechanics!, mechanical_bound

# Keep only device-compatible scalar/array fields; quadrature wrappers stay compact.
function operator_grid(g)
    names = (
        :n,
        :ne,
        :nq,
        :m,
        :eptr,
        :conn,
        :support_element,
        :N,
        :G,
        :weight,
        :offsets,
        :adj,
        :mass,
        :free,
        :gc,
        :ell,
        :eta,
        :rho,
        :kappa,
        :lam,
        :mu,
    )
    NamedTuple{names}(Tuple(getproperty(g, k) for k in names))
end

function gpu_work! end
function eachwork!(f, n, gpu, args...)
    gpu || error("This distribution contains the GPU solver only")
    gpu_work!(f, n, args...)
    nothing
end
zeros_like(g, dims...) = fill!(similar(g.mass, Float64, dims...), 0.0)

@inline function average_gradient!(a, avgG, g)
    e = g.support_element[a]
    volume = x = y = z = 0.0
    @inbounds for q in 1:g.nq
        v = g.weight[q, e]
        volume += v
        x += v * g.G[1, q, a]
        y += v * g.G[2, q, a]
        z += v * g.G[3, q, a]
    end
    @inbounds avgG[1, a] = x / volume
    @inbounds avgG[2, a] = y / volume
    @inbounds avgG[3, a] = z / volume
    nothing
end

"""FP64 B-bar H8 workspace; g is already on the selected backend.

Stress storage is optional: force, quadrature psi+ and element energies are always
available. No quadrature wrapper is expanded or copied. One workspace is sequential.
"""
function workspace(g; gpu = true, record_stress = false)
    g.nq == 8 && g.m == 8g.ne && g.n > 0 || error("Expected full-integration H8 grid")
    gg = operator_grid(g)
    avgG = zeros_like(g, 3, g.m)
    eachwork!(average_gradient!, g.m, gpu, avgG, gg)
    (;
        g = gg,
        gpu,
        f = zeros_like(g, 3, g.n),
        q = zeros_like(g, g.nq, g.ne),
        elastic = zeros_like(g, g.ne),
        fracture = zeros_like(g, g.ne),
        avgG,
        stress = record_stress ? zeros_like(g, 6, g.nq, g.ne) : nothing,
        localforce = zeros_like(g, 3, g.m),
        bounds = zeros_like(g, g.n),
    )
end

@inline function mechanics_element!(
    e, u, d, g, avgG, localforce, stress, qp, energies, fractures, bbar
)
    first = g.eptr[e]
    last = g.eptr[e + 1] - 1
    meantrace = elastic = fracture = 0.0
    @inbounds for a in first:last
        i = g.conn[a]
        meantrace += avgG[1, a] * u[1, i] + avgG[2, a] * u[2, i] + avgG[3, a] * u[3, i]
        localforce[1, a] = 0.0
        localforce[2, a] = 0.0
        localforce[3, a] = 0.0
    end
    @inbounds for q in 1:g.nq
        exx = eyy = ezz = exy = eyz = exz = dq = dx = dy = dz = 0.0
        for a in first:last
            i = g.conn[a]
            gx = g.G[1, q, a]
            gy = g.G[2, q, a]
            gz = g.G[3, q, a]
            ux = u[1, i]
            uy = u[2, i]
            uz = u[3, i]
            exx += gx * ux
            eyy += gy * uy
            ezz += gz * uz
            exy += (gy * ux + gx * uy) / 2
            eyz += (gz * uy + gy * uz) / 2
            exz += (gz * ux + gx * uz) / 2
            dq += g.N[q, a] * d[i]
            dx += gx * d[i]
            dy += gy * d[i]
            dz += gz * d[i]
        end
        correction = bbar ? (meantrace - exx - eyy - ezz) / 3 : 0.0
        exx += correction
        eyy += correction
        ezz += correction
        deg = (1 - dq)^2 + g.kappa
        sxx, syy, szz, sxy, syz, sxz, plus, minus = miehe3d(
            exx, eyy, ezz, exy, eyz, exz, g.lam, g.mu, deg
        )
        if stress !== nothing
            stress[1, q, e] = sxx
            stress[2, q, e] = syy
            stress[3, q, e] = szz
            stress[4, q, e] = sxy
            stress[5, q, e] = syz
            stress[6, q, e] = sxz
        end
        qp[q, e] = plus
        weight = g.weight[q, e]
        elastic += weight * (deg * plus + minus)
        fracture += weight * g.gc / 2 * (dq * dq / g.ell + g.ell * (dx * dx + dy * dy + dz * dz))
        meanstress = bbar ? (sxx + syy + szz) / 3 : 0.0
        for a in first:last
            gx = g.G[1, q, a]
            gy = g.G[2, q, a]
            gz = g.G[3, q, a]
            localforce[1, a] +=
                weight * (gx * sxx + gy * sxy + gz * sxz + (avgG[1, a] - gx) * meanstress)
            localforce[2, a] +=
                weight * (gx * sxy + gy * syy + gz * syz + (avgG[2, a] - gy) * meanstress)
            localforce[3, a] +=
                weight * (gx * sxz + gy * syz + gz * szz + (avgG[3, a] - gz) * meanstress)
        end
    end
    @inbounds energies[e] = elastic
    @inbounds fractures[e] = fracture
    nothing
end

@inline function gather_force!(i, f, localforce, g)
    fx = fy = fz = 0.0
    @inbounds for j in g.offsets[i]:(g.offsets[i + 1] - 1)
        a = g.adj[j]
        fx += localforce[1, a]
        fy += localforce[2, a]
        fz += localforce[3, a]
    end
    @inbounds f[1, i] = fx
    @inbounds f[2, i] = fy
    @inbounds f[3, i] = fz
    nothing
end
function mechanics!(w, u, d; bbar = true)
    g = w.g
    size(u) == (3, g.n) && length(d) == g.n || throw(DimensionMismatch("H8 state shape"))
    eachwork!(
        mechanics_element!,
        g.ne,
        w.gpu,
        u,
        d,
        g,
        w.avgG,
        w.localforce,
        w.stress,
        w.q,
        w.elastic,
        w.fracture,
        bbar,
    )
    eachwork!(gather_force!, g.n, w.gpu, w.f, w.localforce, g)
    w
end

# Integrate ordinary undegraded H8 element entries before their absolute row sums.
# The B-bar volumetric projection cannot increase isotropic linear elastic energy.
@inline function bound_support!(a, out, g)
    rx = ry = rz = 0.0
    @inbounds begin
        e = g.support_element[a]
        ia = g.conn[a]
        for b in g.eptr[e]:(g.eptr[e + 1] - 1)
            ib = g.conn[b]
            xx = xy = xz = yx = yy = yz = zx = zy = zz = 0.0
            for q in 1:g.nq
                ax = g.G[1, q, a]
                ay = g.G[2, q, a]
                az = g.G[3, q, a]
                bx = g.G[1, q, b]
                by = g.G[2, q, b]
                bz = g.G[3, q, b]
                v = g.weight[q, e]
                prod = ax * bx + ay * by + az * bz
                xx += v * ((g.lam + g.mu) * ax * bx + g.mu * prod)
                yy += v * ((g.lam + g.mu) * ay * by + g.mu * prod)
                zz += v * ((g.lam + g.mu) * az * bz + g.mu * prod)
                xy += v * (g.lam * ax * by + g.mu * ay * bx)
                xz += v * (g.lam * ax * bz + g.mu * az * bx)
                yx += v * (g.lam * ay * bx + g.mu * ax * by)
                yz += v * (g.lam * ay * bz + g.mu * az * by)
                zx += v * (g.lam * az * bx + g.mu * ax * bz)
                zy += v * (g.lam * az * by + g.mu * ay * bz)
            end
            if !iszero(g.free[1, ib])
                rx += abs(xx)
                ry += abs(yx)
                rz += abs(zx)
            end
            if !iszero(g.free[2, ib])
                rx += abs(xy)
                ry += abs(yy)
                rz += abs(zy)
            end
            if !iszero(g.free[3, ib])
                rx += abs(xz)
                ry += abs(yz)
                rz += abs(zz)
            end
        end
        out[1, a] = iszero(g.free[1, ia]) ? 0.0 : rx
        out[2, a] = iszero(g.free[2, ia]) ? 0.0 : ry
        out[3, a] = iszero(g.free[3, ia]) ? 0.0 : rz
    end
    nothing
end
@inline function bound_node!(i, bounds, localbound, g)
    rx = ry = rz = 0.0
    @inbounds for j in g.offsets[i]:(g.offsets[i + 1] - 1)
        a = g.adj[j]
        rx += localbound[1, a]
        ry += localbound[2, a]
        rz += localbound[3, a]
    end
    @inbounds bounds[i] = max(rx, ry, rz) / g.mass[i]
    nothing
end
"""Return (dt, omega2); dt=2/sqrt(omega2), with no additional safety multiplier.

Reuses localforce scratch. Call mechanics! before subsequently consuming its force.
"""
function mechanical_bound(w)
    g = w.g
    0 <= g.kappa <= 1 || error("Invalid residual stiffness")
    eachwork!(bound_support!, g.m, w.gpu, w.localforce, g)
    eachwork!(bound_node!, g.n, w.gpu, w.bounds, w.localforce, g)
    omega2 = (1 + g.kappa) * maximum(w.bounds)
    isfinite(omega2) && omega2 > 0 || error("Invalid mechanical bound")
    (; dt = 2 / sqrt(omega2), omega2)
end

# The tensile Hessian is bounded by the full elastic Hessian; multiply the
# undegraded ordinary-H8 bound by 1+kappa. B-bar projection is energy nonincreasing.

end
