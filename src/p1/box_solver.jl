module BoxKWOptimized
# Projected Newton and Jacobi-PCG in a cooperative GPU kernel.
# The launch grid must fit the device's measured residency.
using CUDA
include("../q1/action.jl")
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

# Operator descriptors carry concrete Val tags; Julia specializes each path.
@inline six_index(a, b) = a <= b ? div(b * (b - 1), 2) + a : div(a * (a - 1), 2) + b
@inline function local_value(K, x, g, i, e, a)
    value = 0.0
    if K.kind == Val(:direct) || K.kind == Val(:decoded) || K.kind == Val(:atomic)
        gx = gy = sd = 0.0
        @inbounds for b in 1:3
            z = x[g.tri[b, e]]
            gx += g.G[1, b, e] * z
            gy += g.G[2, b, e] * z
            sd += z
        end
        @inbounds value =
            g.area[e] * (
                g.gc * g.ell * (g.G[1, a, e] * gx + g.G[2, a, e] * gy) +
                (g.gc / g.ell + 2K.q[e]) * (sd + x[i]) / 12
            )
    elseif K.kind == Val(:geom6)
        sd = 0.0
        @inbounds for b in 1:3
            z = x[g.tri[b, e]]
            value += K.values[six_index(a, b), e] * z
            sd += z
        end
        @inbounds value += K.reaction[e] * (sd + x[i])
    else
        @inbounds for b in 1:3
            v =
                K.kind == Val(:six_soa) ? K.values[e, six_index(a, b)] :
                K.values[six_index(a, b), e]
            value += v * x[g.tri[b, e]]
        end
    end
    value
end

@inline function applyQ!(y, x, K, capacity, g, gg, free, masked)
    if K.kind == Val(:atomic)
        @inbounds for i in grid_index():grid_stride():g.n
            y[i] = (!masked || free[i]) ? capacity[i] * x[i] : 0.0
        end
        CUDA.CG.sync(gg)
        @inbounds for e in grid_index():grid_stride():g.ne
            gx = gy = sd = 0.0
            for b in 1:3
                z = x[g.tri[b, e]]
                gx += g.G[1, b, e] * z
                gy += g.G[2, b, e] * z
                sd += z
            end
            for a in 1:3
                i = g.tri[a, e]
                if !masked || free[i]
                    value =
                        g.area[e] * (
                            g.gc * g.ell * (g.G[1, a, e] * gx + g.G[2, a, e] * gy) +
                            (g.gc / g.ell + 2K.q[e]) * (sd + x[i]) / 12
                        )
                    CUDA.@atomic y[i] += value
                end
            end
        end
    else
        @inbounds for i in grid_index():grid_stride():g.n
            if masked && !free[i]
                y[i] = 0.0
                continue
            end
            value = 0.0 # Match canonical phase sum first, then capacity.
            if K.kind == Val(:q1)
                value = q1_row(i, x, K.values, g)
            elseif K.kind == Val(:csr)
                for j in K.rowptr[i]:(K.rowptr[i + 1] - 1)
                    value += K.values[j] * x[K.cols[j]]
                end
            else
                for j in g.offsets[i]:(g.offsets[i + 1] - 1)
                    if K.kind == Val(:decoded)
                        e = K.elements[j]
                        a = K.locals[j]
                    else
                        k = g.adj[j]
                        e = div(k - 1, 3) + 1
                        a = (k - 1) % 3 + 1
                    end
                    if K.kind == Val(:local9)
                        for b in 1:3
                            value += K.values[a, b, e] * x[g.tri[b, e]]
                        end
                    else
                        value += local_value(K, x, g, i, e, a)
                    end
                end
            end
            y[i] = value + capacity[i] * x[i]
        end
    end
    CUDA.CG.sync(gg)
    nothing
end

@inline function reduction_pair(v1, v2, parts, ctrl, gg)
    shared = CUDA.@cuStaticSharedMem(Float64, (NT, 2))
    lane = threadIdx().x
    shared[lane, 1] = v1
    shared[lane, 2] = v2
    sync_threads()
    offset = div(NT, 2)
    while offset > 0
        if lane <= offset
            shared[lane, 1] += shared[lane + offset, 1]
            shared[lane, 2] += shared[lane + offset, 2]
        end
        sync_threads()
        offset = div(offset, 2)
    end
    if lane == 1
        parts[blockIdx().x, 1] = shared[1, 1]
        parts[blockIdx().x, 2] = shared[1, 2]
    end
    CUDA.CG.sync(gg)
    if grid_index() == 1
        a = b = 0.0
        for k in 1:gridDim().x
            a += parts[k, 1]
            b += parts[k, 2]
        end
        ctrl[2] = a
        ctrl[3] = b
    end
    CUDA.CG.sync(gg)
    ctrl[2], ctrl[3]
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

function kernel!(
    w, K, capacity, diagonal, rhs, lower, upper, g, tolerance, max_outer, skip_fixed, fused
)
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
            capacity[i] >= 0 &&
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
        applyQ!(y, x, K, capacity, g, gg, free, false)
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
        rr, rz = if fused
            reduction_pair(r2, rzlocal, w.pairs, w.ctrl, gg)
        else
            (
                reduction(r2, w.parts, w.ctrl, 2, gg, false),
                reduction(rzlocal, w.parts, w.ctrl, 3, gg, false),
            )
        end
        target = max(1e-14, 1e-12 * sqrt(rr))
        for _ in 1:min(2g.n + 10, 2000)
            sqrt(rr) <= target && break
            applyQ!(y, direction, K, capacity, g, gg, free, skip_fixed)
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
            rr, newrz = if fused
                reduction_pair(r2, rzlocal, w.pairs, w.ctrl, gg)
            else
                (
                    reduction(r2, w.parts, w.ctrl, 2, gg, false),
                    reduction(rzlocal, w.parts, w.ctrl, 3, gg, false),
                )
            end
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
                    applyQ!(y, delta, K, capacity, g, gg, free, false)
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

function workspace(d; maxblocks = 384)
    @assert maxblocks > 0
    names = (:x, :y, :grad, :proj, :p, :r, :z, :direction, :trial, :delta)
    v = NamedTuple{names}(Tuple(similar(d) for _ in names))
    merge(
        v,
        (;
            free = similar(d, Bool),
            parts = CUDA.zeros(Float64, maxblocks),
            pairs = CUDA.zeros(Float64, maxblocks, 2),
            ctrl = CUDA.zeros(Float64, 5),
        ),
    )
end
# Keep the old damage bound separate from mutable solver buffers.
# Copy borrowed output before reusing the workspace.
function launch_box!(
    w,
    K,
    capacity,
    diagonal,
    rhs,
    lower,
    upper,
    g;
    tolerance = 1e-12,
    blocks = 64,
    skip_fixed = false,
    fused = false,
)
    @assert 1 <= blocks <= min(length(w.parts), size(w.pairs, 1)) &&
        all(length(v) == g.n for v in (capacity, diagonal, rhs, lower, upper))
    @assert isfinite(tolerance) && tolerance > 0
    @assert all(!Base.mightalias(lower, v) for v in values(w)) "lower/dold must not alias mutable solver workspace"
    kernel = @cuda launch = false kernel!(
        w, K, capacity, diagonal, rhs, lower, upper, g, tolerance, 300, skip_fixed, fused
    )
    # Caller must qualify cooperative residency for each distinct launch configuration.
    kernel(
        w,
        K,
        capacity,
        diagonal,
        rhs,
        lower,
        upper,
        g,
        tolerance,
        300,
        skip_fixed,
        fused;
        cooperative = true,
        threads = NT,
        blocks,
    )
    # No device-to-host read. Same-stream work may consume x/ctrl in launch order.
    # ctrl=[status,residual,outer,cg,actions] is overwritten at the next launch:
    # production must accumulate failure/max residual on device before that reuse.
    w.x
end

# Explicit host synchronization/readback; no numerical update or correction.
# Read before the workspace is reused. own_result=false borrows workspace x.
function read_box_result(w; own_result = true)
    status = Array(w.ctrl)
    status[1] == 1 || error("GPU-resident box solve failed: status=$status")
    (;
        x = own_result ? copy(w.x) : w.x,
        residual = status[2],
        outer = Int(status[3]),
        cg_total = Int(status[4]),
        actions = Int(status[5]),
    )
end

# Synchronous solve returns owned damage by default.
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
    blocks = 64,
    skip_fixed = false,
    fused = false,
    own_result = true,
)
    launch_box!(
        w, K, capacity, diagonal, rhs, lower, upper, g; tolerance, blocks, skip_fixed, fused
    )
    read_box_result(w; own_result)
end
const solve_box! = solve!

function apply_kernel!(y, x, K, capacity, g, free, masked)
    gg = CUDA.CG.this_grid()
    applyQ!(y, x, K, capacity, g, gg, free, masked)
    return nothing
end
end
