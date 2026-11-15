# KW-specific diagnostics: reactions include prescribed-boundary inertia.
const sum_parts = P.sum_parts
@inline function loading(t, scale, speed, ramp)
    v0 = mul(speed, scale)
    ub = t < ramp ? divrn(mul(mul(mul(0.5, v0), t), t), ramp) : mul(v0, add(t, -mul(0.5, ramp)))
    vb = mul(v0, min(divrn(t, ramp), 1.0))
    ab = t < ramp ? divrn(v0, ramp) : 0.0
    (ub, vb, ab)
end
function initialize_ledger!(diag, parts, oldreaction)
    if threadIdx().x == 1 && blockIdx().x == 1
        kin = 0.5sum_parts(parts, 4)
        elastic = sum_parts(parts, 1)
        fracture = sum_parts(parts, 2)
        @inbounds begin
            diag[9, 1] = oldreaction
            diag[10, 1] = kin
            diag[11, 1] = elastic
            diag[12, 1] = fracture
            diag[29, 1] = sum_parts(parts, 3)
            diag[37, 1] = kin + elastic + fracture + diag[14, 1] - diag[13, 1]
            diag[15, 1] = diag[37, 1] - diag[27, 1] - diag[28, 1]
        end
    end
    return nothing
end
# Batch diagnostic reductions; field updates and independent KKT stay unchanged.
# Six shared rows cover both mechanics (four values) and the energy ledger.
function mechanics_kw_kernel!(parts, elastic, fracture, f, rate, mass, v, a, n, ne)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    s = CUDA.@cuStaticSharedMem(Float64, (6, 256))
    @inbounds begin
        s[1, lane] = i <= ne ? elastic[i] : 0.0
        s[2, lane] = i <= ne ? fracture[i] : 0.0
        s[3, lane] =
            i <= n ?
            (f[1, i] + mass[i] * a[1, i]) * rate[1, i] +
            (f[2, i] + mass[i] * a[2, i]) * rate[2, i] : 0.0
        s[4, lane] = i <= n ? mass[i] * v[1, i]^2 + mass[i] * v[2, i]^2 : 0.0
    end
    sync_threads()
    offset = 128
    while offset > 0
        if lane <= offset
            @inbounds for row in 1:4
                s[row, lane] += s[row, lane + offset]
            end
        end
        sync_threads()
        offset = div(offset, 2)
    end
    if lane == 1
        @inbounds for row in 1:4
            parts[row, blockIdx().x] = s[row, 1]
        end
    end
    return nothing
end

function energy_kw_kernel!(
    parts, delta, z, gradient, capacity, elastic, fracture, supply, f, rate, mass, a, n, ne
)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    s = CUDA.@cuStaticSharedMem(Float64, (7, 256))
    @inbounds begin
        s[1, lane] = i <= n ? delta[i] * z[i] : 0.0
        s[2, lane] = i <= n ? gradient[i] * delta[i] : 0.0
        s[3, lane] = i <= n ? capacity[i] * delta[i]^2 : 0.0
        s[4, lane] = i <= ne ? elastic[i] : 0.0
        s[5, lane] = i <= ne ? fracture[i] : 0.0
        s[6, lane] = i <= ne ? supply[i] : 0.0
        s[7, lane] =
            i <= n ?
            (f[1, i] + mass[i] * a[1, i]) * rate[1, i] +
            (f[2, i] + mass[i] * a[2, i]) * rate[2, i] : 0.0
    end
    sync_threads()
    offset = 128
    while offset > 0
        if lane <= offset
            @inbounds for row in 1:7
                s[row, lane] += s[row, lane + offset]
            end
        end
        sync_threads()
        offset = div(offset, 2)
    end
    if lane == 1
        @inbounds for row in 1:7
            parts[row, blockIdx().x] = s[row, 1]
        end
    end
    return nothing
end

function finish_kw_kernel!(
    diag,
    pred,
    mech,
    energy,
    before,
    after,
    ctrl,
    h,
    t,
    step,
    isbe,
    oldforce,
    tol,
    phasebound,
    told,
    velocity_scale,
    speed,
    ramp,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        kin = 0.5sum_parts(mech, 4)
        elastic = sum_parts(energy, 4)
        fracture = sum_parts(energy, 5)
        newE = elastic + fracture
        accepted_reaction = sum_parts(energy, 7)
        reaction = oldforce ? sum_parts(pred, 3) : accepted_reaction
        predicted = sum_parts(pred, 1) + sum_parts(pred, 2)
        curv = isbe ? 0.5sum_parts(energy, 1) : 0.0
        normal = isbe ? -sum_parts(energy, 2) : 0.0
        diss = sum_parts(energy, 3)
        supply = isbe ? sum_parts(energy, 6) : 0.0
        @inbounds begin
            workstep =
                0.5 *
                (diag[9, 1] + reaction) *
                (
                    loading(t, velocity_scale, speed, ramp)[1] -
                    loading(told, velocity_scale, speed, ramp)[1]
                )
            identity = isbe ? newE - predicted + diss + curv + normal - supply : 0.0
            mechstep =
                isbe ? kin - diag[10, 1] + predicted - diag[11, 1] - diag[12, 1] - workstep : 0.0
            diag[13, 1] += workstep
            diag[14, 1] += diss
            diag[16, 1] += curv
            diag[17, 1] += normal
            diag[18, 1] += supply
            diag[19, 1] += mechstep
            physical_total = kin + newE + diag[14, 1] - diag[13, 1]
            physical = physical_total - diag[27, 1] - diag[28, 1]
            balance = physical + diag[16, 1] + diag[17, 1] - diag[18, 1] - diag[19, 1]
            kkt = isbe ? after[3] : 0.0
            residual = isbe ? ctrl[2] : 0.0
            correction = isbe && P.correction_needed(before, tol) ? before[3] : 0.0
            cg = isbe ? ctrl[4] : 0.0
            outer = isbe ? ctrl[3] : 0.0
            actions = isbe ? ctrl[5] + 2 + Float64(correction > 0) : 2.0
            failure = 0.0
            if isbe && !(ctrl[1] == 1 && isfinite(residual) && residual <= tol)
                failure = -10.0
            end
            if !(
                after[4] == 0 &&
                isfinite(after[1]) &&
                after[1] >= 0 &&
                after[2] <= 1 &&
                isfinite(kkt) &&
                kkt <= tol
            )
                failure = -20.0
            end
            if !(
                isfinite(physical) &&
                isfinite(kin) &&
                isfinite(elastic) &&
                isfinite(fracture) &&
                isfinite(diag[13, 1]) &&
                isfinite(diag[14, 1])
            )
                failure = -30.0
            end
            scale = max(abs(diag[13, 1]), abs(newE), 1.0)
            if isbe && !(
                isfinite(identity) &&
                abs(identity) <= 1e-8scale &&
                isfinite(balance) &&
                abs(balance) <= 1e-8scale
            )
                failure = -40.0
            end
            diag[1, 1] == 0 && (diag[1, 1] = failure)
            diag[2, 1] = max(diag[2, 1], isfinite(kkt) ? kkt : Inf)
            diag[3, 1] = max(diag[3, 1], isfinite(residual) ? residual : Inf)
            diag[4, 1] = min(diag[4, 1], after[1])
            diag[5, 1] = after[2]
            diag[6, 1] = t
            diag[7, 1] = h
            diag[8, 1] = step
            diag[9, 1] = reaction
            diag[10, 1] = kin
            diag[11, 1] = elastic
            diag[12, 1] = fracture
            diag[15, 1] = physical
            diag[20, 1] = balance
            diag[21, 1] = identity
            diag[22, 1] += cg
            diag[23, 1] += outer
            diag[24, 1] += actions
            diag[25, 1] += Float64(correction > 0)
            diag[26, 1] = max(diag[26, 1], correction)
            diag[29, 1] = accepted_reaction
            diag[30, 1] = kkt
            diag[31, 1] = residual
            diag[32, 1] = cg
            diag[33, 1] = outer
            diag[34, 1] = actions
            diag[35, 1] = after[1]
            diag[36, 1] = phasebound
            diag[37, 1] = physical_total
            column = mod(step - 1, size(diag, 2) - 1) + 2
            for row in 1:P.NDIAG
                diag[row, column] = diag[row, 1]
            end
        end
    end
    return nothing
end
