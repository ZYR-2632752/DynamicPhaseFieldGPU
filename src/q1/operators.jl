using CUDA

function make_gpu_elliptic_grid(g)
    ne = g.nx * g.ny
    merge(
        g,
        (;
            cells = CuArray(Int32.(g.cells)),
            quad = Tuple(g.quad),
            volume = CuArray(g.volume),
            free = CuArray(g.free),
            upper = CuArray(g.upper),
            loads = CuArray(g.loads),
            mass = CuArray(g.mass),
            C = CuArray(g.C),
            local_force = CUDA.zeros(Float64, 2, 4, ne),
            local_phase = CUDA.zeros(Float64, 4, ne),
            local_rhs = CUDA.zeros(Float64, 4, ne),
            elastic = CUDA.zeros(Float64, ne),
            fracture = CUDA.zeros(Float64, ne),
            curvature = CUDA.zeros(Float64, ne),
            supply = CUDA.zeros(Float64, ne),
            reaction = CUDA.zeros(Float64, g.ny + 1),
        ),
    )
end

@inline gpu_index() = (blockIdx().x - 1) * blockDim().x + threadIdx().x

function gpu_mechanics_kernel!(positive, negative, u, d, stress, g)
    e = gpu_index()
    if e <= g.nx * g.ny
        @inbounds begin
            for k in 1:4
                g.local_force[1, k, e] = 0.0
                g.local_force[2, k, e] = 0.0
                g.local_phase[k, e] = 0.0
            end
            elastic = fracture = 0.0
            for p in 1:4
                q = g.quad[p]
                exx = eyy = exy = dg = dx = dy = 0.0
                for k in 1:4
                    i = g.cells[k, e]
                    N = q.N[k]
                    gx = q.gx[k]
                    gy = q.gy[k]
                    exx += gx * u[1, i]
                    eyy += gy * u[2, i]
                    exy += (gy * u[1, i] + gx * u[2, i]) / 2
                    dg += N * d[i]
                    dx += gx * d[i]
                    dy += gy * d[i]
                end
                degradation = (1 - dg)^2 + g.kappa
                sx, sy, sxy, qp, qm = mersel_q1_split(exx, eyy, exy, g.E, degradation)
                if !isnothing(stress)
                    stress[1, p, e] = sx
                    stress[2, p, e] = sy
                    stress[3, p, e] = 0.0
                    stress[4, p, e] = sqrt(2.0) * sxy
                end
                positive[p, e] = qp
                negative[p, e] = qm
                elastic += g.weight * (degradation * qp + qm)
                fracture += g.weight * g.gc / 2 * (dg^2 / g.ell + g.ell * (dx^2 + dy^2))
                for k in 1:4
                    N = q.N[k]
                    gx = q.gx[k]
                    gy = q.gy[k]
                    g.local_force[1, k, e] += g.weight * (gx * sx + gy * sxy)
                    g.local_force[2, k, e] += g.weight * (gy * sy + gx * sxy)
                    g.local_phase[k, e] +=
                        g.weight * (
                            N * (2qp * (1 - dg) - g.gc / g.ell * dg) -
                            g.gc * g.ell * (gx * dx + gy * dy)
                        )
                end
            end
            g.elastic[e] = elastic
            g.fracture[e] = fracture
        end
    end
    return nothing
end

# Nodes gather their four adjacent cells in source cell order; no atomic accumulation.
@inline corner(ix, iy, ex, ey) = iy == ey ? (ix == ex ? 1 : 2) : (ix == ex ? 4 : 3)
function gpu_force_gather!(force, phaseforce, g)
    i = gpu_index()
    if i <= g.n
        ix = (i - 1) % (g.nx + 1)
        iy = (i - 1) ÷ (g.nx + 1)
        fx = fy = fd = 0.0
        @inbounds for ey in max(0, iy - 1):min(g.ny - 1, iy), ex in max(0, ix - 1):min(g.nx - 1, ix)
            e = 1 + ex + g.nx * ey
            a = corner(ix, iy, ex, ey)
            fx += g.local_force[1, a, e]
            fy += g.local_force[2, a, e]
            fd += g.local_phase[a, e]
        end
        @inbounds force[1, i] = fx
        @inbounds force[2, i] = fy
        @inbounds phaseforce[i] = fd
    end
    return nothing
end

function mersel_q1_operators!(
    force::CuArray, phaseforce, positive, negative, u, d, g; stress = nothing
)
    @cuda threads = 128 blocks = cld(g.nx * g.ny, 128) gpu_mechanics_kernel!(
        positive, negative, u, d, stress, g
    )
    @cuda threads = 128 blocks = cld(g.n, 128) gpu_force_gather!(force, phaseforce, g)
    (; elastic = sum(g.elastic), fracture = sum(g.fracture))
end

function gpu_coefficients_kernel!(K, H, g)
    e = gpu_index()
    if e <= g.nx * g.ny
        @inbounds begin
            for a in 1:4
                g.local_rhs[a, e] = 0.0
                for b in 1:4
                    K[a, b, e] = 0.0
                end
            end
            for p in 1:4, a in 1:4
                q = g.quad[p]
                g.local_rhs[a, e] += g.weight * 2H[p, e] * q.N[a]
                for b in 1:4
                    K[a, b, e] +=
                        g.weight * (
                            (2H[p, e] + g.gc / g.ell) * q.N[a] * q.N[b] +
                            g.gc * g.ell * (q.gx[a] * q.gx[b] + q.gy[a] * q.gy[b])
                        )
                end
            end
        end
    end
    return nothing
end

function gpu_coefficients_gather!(rhs, diagonal, K, g)
    i = gpu_index()
    if i <= g.n
        ix = (i - 1) % (g.nx + 1)
        iy = (i - 1) ÷ (g.nx + 1)
        r = diag = 0.0
        @inbounds for ey in max(0, iy - 1):min(g.ny - 1, iy), ex in max(0, ix - 1):min(g.nx - 1, ix)
            e = 1 + ex + g.nx * ey
            a = corner(ix, iy, ex, ey)
            r += g.local_rhs[a, e]
            diag += K[a, a, e]
        end
        @inbounds rhs[i] = r * g.upper[i]
        @inbounds diagonal[i] = g.upper[i] == 0 ? 1.0 : diag
    end
    return nothing
end

function elliptic_coefficients!(K::CuArray, rhs, diagonal, H, g)
    @cuda threads = 128 blocks = cld(g.nx * g.ny, 128) gpu_coefficients_kernel!(K, H, g)
    @cuda threads = 128 blocks = cld(g.n, 128) gpu_coefficients_gather!(rhs, diagonal, K, g)
end

function gpu_action_kernel!(y, x, K, g)
    i = gpu_index()
    if i <= g.n
        ix = (i - 1) % (g.nx + 1)
        iy = (i - 1) ÷ (g.nx + 1)
        value = 0.0
        @inbounds begin
            for ey in max(0, iy - 1):min(g.ny - 1, iy), ex in max(0, ix - 1):min(g.nx - 1, ix)
                e = 1 + ex + g.nx * ey
                a = corner(ix, iy, ex, ey)
                localvalue = 0.0
                for b in 1:4
                    j = g.cells[b, e]
                    localvalue += K[a, b, e] * x[j] * g.upper[j]
                end
                value += g.upper[i] * localvalue
            end
            y[i] = g.upper[i] == 0 ? x[i] : value
        end
    end
    return nothing
end

function elliptic_action!(y::CuArray, x, K, g)
    @cuda threads = 128 blocks = cld(g.n, 128) gpu_action_kernel!(y, x, K, g)
    y
end

function gpu_energy_kernel!(qp, qm, d, increment, g)
    e = gpu_index()
    if e <= g.nx * g.ny
        elastic = fracture = curvature = 0.0
        @inbounds for p in 1:4
            q = g.quad[p]
            dg = dx = dy = dd = ddx = ddy = 0.0
            for k in 1:4
                i = g.cells[k, e]
                N = q.N[k]
                gx = q.gx[k]
                gy = q.gy[k]
                dg += N * d[i]
                dx += gx * d[i]
                dy += gy * d[i]
                dd += N * increment[i]
                ddx += gx * increment[i]
                ddy += gy * increment[i]
            end
            elastic += g.weight * (((1 - dg)^2 + g.kappa) * qp[p, e] + qm[p, e])
            fracture += g.weight * g.gc / 2 * (dg^2 / g.ell + g.ell * (dx^2 + dy^2))
            curvature +=
                g.weight / 2 * ((2qp[p, e] + g.gc / g.ell) * dd^2 + g.gc * g.ell * (ddx^2 + ddy^2))
        end
        @inbounds g.elastic[e] = elastic
        @inbounds g.fracture[e] = fracture
        @inbounds g.curvature[e] = curvature
    end
    return nothing
end

function mersel_q1_phase_energy(qp::CuArray, qm, d, increment, g)
    @cuda threads = 128 blocks = cld(g.nx * g.ny, 128) gpu_energy_kernel!(qp, qm, d, increment, g)
    (; elastic = sum(g.elastic), fracture = sum(g.fracture), curvature = sum(g.curvature))
end
