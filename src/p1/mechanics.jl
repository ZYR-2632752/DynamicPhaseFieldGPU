@inline function main_element!(e, u, d, g, lf, lp, qp, el, fr)
    # Integrate the quadratic degradation exactly on each linear triangle.
    @inbounds begin
        exx = eyy = exy = dx = dy = sd = sd2 = 0.0
        for a in 1:3
            i = g.tri[a, e]
            da = d[i]
            gx = g.G[1, a, e]
            gy = g.G[2, a, e]
            exx += gx * u[1, i]
            eyy += gy * u[2, i]
            exy += (gy * u[1, i] + gx * u[2, i]) / 2
            dx += gx * da
            dy += gy * da
            sd += da
            sd2 += da * da
        end
        dsquare = (sd * sd + sd2) / 12
        degradation = 1 - 2sd / 3 + dsquare + g.kappa
        sx, sy, sxy, plus, minus = miehe_split(exx, eyy, exy, g.lam, g.mu, degradation)
        qp[e] = plus
        el[e] = g.area[e] * (degradation * plus + minus)
        fr[e] = g.area[e] * g.gc / 2 * (dsquare / g.ell + g.ell * (dx^2 + dy^2))
        for a in 1:3
            i = g.tri[a, e]
            gx = g.G[1, a, e]
            gy = g.G[2, a, e]
            lf[1, a, e] = g.area[e] * (gx * sx + gy * sxy)
            lf[2, a, e] = g.area[e] * (gy * sy + gx * sxy)
            lp[a, e] =
                g.area[e] * (
                    2plus / 3 - (2plus + g.gc / g.ell) * (sd + d[i]) / 12 -
                    g.gc * g.ell * (gx * dx + gy * dy)
                )
        end
    end
    nothing
end
function main_element_kernel!(u, d, g, lf, lp, qp, el, fr)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e <= g.ne && main_element!(e, u, d, g, lf, lp, qp, el, fr)
    return nothing
end
function main_operators!(w, u, d)
    g = w.g
    @cuda threads = 256 blocks = cld(g.ne, 256) main_element_kernel!(
        u, d, g, w.localf, w.localp, w.qp, w.elastic, w.fracture
    )
    @cuda threads = 256 blocks = cld(g.n, 256) gather_kernel!(g, w.localf, w.localp, w.f, w.phase)
    nothing
end

@inline function coefficient_node!(i, b, diagonal, H, g)
    rhs = diag = 0.0
    @inbounds for j in g.offsets[i]:(g.offsets[i + 1] - 1)
        k = g.adj[j]
        e = div(k - 1, 3) + 1
        a = (k - 1) % 3 + 1
        rhs += 2H[e] * g.area[e] / 3
        diag +=
            g.area[e] *
            (g.gc * g.ell * (g.G[1, a, e]^2 + g.G[2, a, e]^2) + (g.gc / g.ell + 2H[e]) / 6)
    end
    @inbounds b[i] = rhs
    @inbounds diagonal[i] = diag
    nothing
end
@inline function phase_node!(i, y, x, H, g)
    value = 0.0
    @inbounds for j in g.offsets[i]:(g.offsets[i + 1] - 1)
        k = g.adj[j]
        e = div(k - 1, 3) + 1
        a = (k - 1) % 3 + 1
        gx = gy = sd = 0.0
        for bb in 1:3
            z = x[g.tri[bb, e]]
            gx += g.G[1, bb, e] * z
            gy += g.G[2, bb, e] * z
            sd += z
        end
        value +=
            g.area[e] * (
                g.gc * g.ell * (g.G[1, a, e] * gx + g.G[2, a, e] * gy) +
                (g.gc / g.ell + 2H[e]) * (sd + x[i]) / 12
            )
    end
    @inbounds y[i] = value
    nothing
end
function coefficient_kernel!(b, diagonal, H, g)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= g.n && coefficient_node!(i, b, diagonal, H, g)
    return nothing
end
function phase_kernel!(y, x, H, g)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= g.n && phase_node!(i, y, x, H, g)
    return nothing
end
function coefficients!(b, diagonal, H, g)
    @cuda threads = 256 blocks = cld(g.n, 256) coefficient_kernel!(b, diagonal, H, g)
    nothing
end
function action!(y, x, H, g)
    @cuda threads = 256 blocks = cld(g.n, 256) phase_kernel!(y, x, H, g)
    y
end

@inline function supply_element!(e, out, qp, H, d, dn, g)
    oldsum = newsum = oldsq = newsq = 0.0
    @inbounds for a in 1:3
        i = g.tri[a, e]
        oldsum += d[i]
        newsum += dn[i]
        oldsq += d[i]^2
        newsq += dn[i]^2
    end
    change = -2 * (newsum - oldsum) / 3 + (newsum^2 + newsq - oldsum^2 - oldsq) / 12
    @inbounds out[e] = g.area[e] * (qp[e] - H[e]) * change
    nothing
end
function supply_kernel!(out, qp, H, d, dn, g)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e <= g.ne && supply_element!(e, out, qp, H, d, dn, g)
    return nothing
end
function driver_supply!(out, qp, H, d, dn, g)
    @cuda threads = 256 blocks = cld(g.ne, 256) supply_kernel!(out, qp, H, d, dn, g)
    sum(out)
end
