module BoxGPUResident
# Cooperative-grid GPU implementation of the existing projected Newton/CG box
# algorithm. CPU launches once and receives status once, no per-iteration readback.
# Requires cooperative launch; bounded to <=64 blocks (one resident block/SM on4090).
using CUDA
const NT = 128
@inline grid_index() = (blockIdx().x - 1) * blockDim().x + threadIdx().x
@inline grid_stride() = blockDim().x * gridDim().x

@inline function reduction(value, parts, ctrl, slot, gg, ismax)
    shared = CUDA.@cuStaticSharedMem(Float64, NT)
    lane = threadIdx().x
    shared[lane] = value
    sync_threads()
    offset = div(NT, 2)
    while offset > 0
        if lane <= offset
            shared[lane] =
                ismax ? max(shared[lane], shared[lane + offset]) :
                shared[lane] + shared[lane + offset]
        end
        sync_threads()
        offset = div(offset, 2)
    end
    lane == 1 && (parts[blockIdx().x] = shared[1])
    CUDA.CG.sync(gg)
    if grid_index() == 1
        total = 0.0
        for b in 1:gridDim().x
            total = ismax ? max(total, parts[b]) : total + parts[b]
        end
        ctrl[slot] = total
    end
    CUDA.CG.sync(gg)
    ctrl[slot]
end

@inline function applyQ!(y, x, K, capacity, g, gg)
    @inbounds for i in grid_index():grid_stride():g.n
        value = capacity[i] * x[i]
        if K isa NamedTuple
            for j in K.rowptr[i]:(K.rowptr[i + 1] - 1)
                value += K.values[j] * x[K.cols[j]]
            end
        else
            for j in g.offsets[i]:(g.offsets[i + 1] - 1)
                k = g.adj[j]
                e = div(k - 1, 3) + 1
                a = (k - 1) % 3 + 1
                for b in 1:3
                    value += K[a, b, e] * x[g.tri[b, e]]
                end
            end
        end
        y[i] = value
    end
    CUDA.CG.sync(gg)
    nothing
end
@inline function finish!(ctrl, status, residual, outer, cg, actions)
    if grid_index() == 1
        ctrl[1] = status
        ctrl[2] = residual
        ctrl[3] = outer
        ctrl[4] = cg
        ctrl[5] = actions
    end
    nothing
end

function kernel!(w, K, capacity, diagonal, rhs, lower, upper, g, tolerance, max_outer)
    gg = CUDA.CG.this_grid()
    if !CUDA.CG.is_valid(gg)
        finish!(w.ctrl, -9, Inf, 0, 0, 0)
        return nothing
    end
    x = w.x
    y = w.y
    grad = w.grad
    proj = w.proj
    free = w.free
    p = w.p
    r = w.r
    z = w.z
    direction = w.direction
    trial = w.trial
    delta = w.delta
    invalid = 0.0
    @inbounds for i in grid_index():grid_stride():g.n
        valid =
            isfinite(capacity[i]) &&
            capacity[i] > 0 &&
            isfinite(diagonal[i]) &&
            isfinite(rhs[i]) &&
            isfinite(lower[i]) &&
            isfinite(upper[i]) &&
            diagonal[i] > 0 &&
            lower[i] <= upper[i]
        invalid += !valid
        x[i] = lower[i]
    end
    if reduction(invalid, w.parts, w.ctrl, 2, gg, false) > 0
        finish!(w.ctrl, -1, Inf, 0, 0, 0)
        return nothing
    end
    actions = cg_total = 0
    for outer in 0:max_outer
        applyQ!(y, x, K, capacity, g, gg)
        actions += 1
        localres = 0.0
        @inbounds for i in grid_index():grid_stride():g.n
            grad[i] = y[i] - rhs[i]
            proj[i] = clamp(x[i] - grad[i] / diagonal[i], lower[i], upper[i]) - x[i]
            localres = max(localres, abs(proj[i]))
        end
        residual = reduction(localres, w.parts, w.ctrl, 2, gg, true)
        if residual <= tolerance
            finish!(w.ctrl, 1, residual, outer, cg_total, actions)
            return nothing
        elseif outer == max_outer
            finish!(w.ctrl, -2, residual, outer, cg_total, actions)
            return nothing
        end
        r2 = rzlocal = 0.0
        @inbounds for i in grid_index():grid_stride():g.n
            free[i] =
                ((x[i] > lower[i]) || (grad[i] < 0)) &&
                ((x[i] < upper[i]) || (grad[i] > 0)) &&
                lower[i] < upper[i]
            p[i] = 0.0
            r[i] = -grad[i] * free[i]
            z[i] = r[i] / diagonal[i]
            direction[i] = z[i]
            r2 += r[i]^2
            rzlocal += r[i] * z[i]
        end
        rr = reduction(r2, w.parts, w.ctrl, 2, gg, false)
        rz = reduction(rzlocal, w.parts, w.ctrl, 3, gg, false)
        target = max(1e-14, 1e-12 * sqrt(rr))
        for _ in 1:min(2g.n + 10, 2000)
            sqrt(rr) <= target && break
            applyQ!(y, direction, K, capacity, g, gg)
            actions += 1
            localcurv = 0.0
            @inbounds for i in grid_index():grid_stride():g.n
                y[i] *= free[i]
                localcurv += direction[i] * y[i]
            end
            curvature = reduction(localcurv, w.parts, w.ctrl, 2, gg, false)
            if !isfinite(curvature) || curvature <= 0
                finish!(w.ctrl, -3, residual, outer, cg_total, actions)
                return nothing
            end
            alpha = rz / curvature
            r2 = rzlocal = 0.0
            @inbounds for i in grid_index():grid_stride():g.n
                p[i] += alpha * direction[i]
                r[i] -= alpha * y[i]
                z[i] = r[i] / diagonal[i]
                r2 += r[i]^2
                rzlocal += r[i] * z[i]
            end
            rr = reduction(r2, w.parts, w.ctrl, 2, gg, false)
            newrz = reduction(rzlocal, w.parts, w.ctrl, 3, gg, false)
            beta = newrz / rz
            @inbounds for i in grid_index():grid_stride():g.n
                direction[i] = z[i] + beta * direction[i]
            end
            CUDA.CG.sync(gg)
            rz = newrz
            cg_total += 1
        end
        accepted = false
        for which in 1:2
            alpha = 1.0
            for _ in 1:55
                localslope = 0.0
                @inbounds for i in grid_index():grid_stride():g.n
                    s = which == 1 ? p[i] : proj[i]
                    trial[i] = clamp(x[i] + alpha * s, lower[i], upper[i])
                    delta[i] = trial[i] - x[i]
                    localslope += grad[i] * delta[i]
                end
                slope = reduction(localslope, w.parts, w.ctrl, 2, gg, false)
                if slope < 0
                    applyQ!(y, delta, K, capacity, g, gg)
                    actions += 1
                    localcurv = 0.0
                    @inbounds for i in grid_index():grid_stride():g.n
                        localcurv += delta[i] * y[i]
                    end
                    curvature = reduction(localcurv, w.parts, w.ctrl, 2, gg, false)
                    if slope + 0.5curvature <= 1e-4 * slope
                        @inbounds for i in grid_index():grid_stride():g.n
                            x[i] = trial[i]
                        end
                        CUDA.CG.sync(gg)
                        accepted = true
                        break
                    end
                end
                alpha *= 0.5
            end
            accepted && break
        end
        if !accepted
            finish!(w.ctrl, -4, residual, outer, cg_total, actions)
            return nothing
        end
    end
    return nothing
end

function workspace(d)
    names = (:x, :y, :grad, :proj, :p, :r, :z, :direction, :trial, :delta)
    v = NamedTuple{names}(Tuple(similar(d) for _ in names))
    merge(
        v,
        (; free = similar(d, Bool), parts = CUDA.zeros(Float64, 64), ctrl = CUDA.zeros(Float64, 5)),
    )
end
function solve!(
    w,
    K,
    capacity,
    diagonal,
    rhs,
    lower,
    upper,
    g;
    tolerance = 1e-12,
    blocks = min(64, cld(g.n, NT)),
)
    @assert 1 <= blocks <= 64 &&
        all(length(v) == g.n for v in (capacity, diagonal, rhs, lower, upper))
    @assert (
            K isa NamedTuple ? (length(K.rowptr) == g.n + 1 && length(K.cols) == length(K.values)) :
            size(K) == (3, 3, g.ne)
        ) &&
        isfinite(tolerance) &&
        tolerance > 0
    @cuda cooperative = true threads = NT blocks = blocks kernel!(
        w, K, capacity, diagonal, rhs, lower, upper, g, tolerance, 300
    )
    status = Array(w.ctrl)
    status[1] == 1 || error("GPU-resident box solve failed: status=$status")
    (;
        x = copy(w.x),
        residual = status[2],
        outer = Int(status[3]),
        cg_total = Int(status[4]),
        actions = Int(status[5]),
    )
end
end
