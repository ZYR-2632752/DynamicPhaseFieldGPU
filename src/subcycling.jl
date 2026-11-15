module PhaseSubcycle
using CUDA, LinearAlgebra, SparseArrays
const NT = 128
const CUD = CUDA.CUDACore
@inline add(a, b) = CUD.add_rn(a, b)
@inline mul(a, b) = CUD.mul_rn(a, b)
@inline divrn(a, b) = CUD.div_rn(a, b)
@inline lerp(a, b, t) = add(mul(1.0 - t, a), mul(t, b))
@inline index() = (blockIdx().x - 1) * blockDim().x + threadIdx().x
@inline stride() = blockDim().x * gridDim().x

function workspace(d, nonzeros; read_every = 100, blocks = 128)
    (;
        x = similar(d),
        pred = similar(d),
        r0 = similar(d),
        r1 = similar(d),
        b0 = similar(d),
        b1 = similar(d),
        a0 = CUDA.zeros(Float64, nonzeros),
        mask = CUDA.zeros(Bool, length(d)),
        dissipation = similar(d),
        parts = CUDA.zeros(Float64, blocks),
        ctrl = CUDA.zeros(Float64, 12),
        history = CUDA.zeros(Float64, 5, read_every),
    )
end

@inline function rates!(rate, mask, x, theta, rowptr, cols, a0, a1, b0, b1, capacity, n, gg)
    @inbounds for i in index():stride():n
        value = 0.0
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            value = add(value, mul(lerp(a0[k], a1[k], theta), x[cols[k]]))
        end
        raw = lerp(b0[i], b1[i], theta) - value
        rate[i] = divrn(max(raw, 0.0), capacity[i])
        mask[i] = x[i] < 1 - 1e-12 && raw >= 0.0
    end
    CUDA.CG.sync(gg)
    nothing
end

@inline function free_bound!(ws, slot, theta, rowptr, cols, a0, a1, capacity, n, gg)
    value = 0.0
    @inbounds for i in index():stride():n
        row = 0.0
        if ws.mask[i]
            for k in rowptr[i]:(rowptr[i + 1] - 1)
                ws.mask[cols[k]] && (row = add(row, abs(lerp(a0[k], a1[k], theta))))
            end
            value = max(value, divrn(row, capacity[i]))
        end
    end
    shared = CUDA.@cuStaticSharedMem(Float64, NT)
    lane = threadIdx().x
    shared[lane] = value
    sync_threads()
    offset = div(NT, 2)
    while offset > 0
        lane <= offset && (shared[lane] = max(shared[lane], shared[lane + offset]))
        sync_threads()
        offset = div(offset, 2)
    end
    lane == 1 && (ws.parts[blockIdx().x] = shared[1])
    CUDA.CG.sync(gg)
    if index() == 1
        bound = 0.0
        @inbounds for block in 1:gridDim().x
            bound = max(bound, ws.parts[block])
        end
        @inbounds ws.ctrl[slot] = bound
    end
    CUDA.CG.sync(gg)
    @inbounds ws.ctrl[slot]
end

function kernel!(ws, rowptr, cols, a1, capacity, old, out, clock, diagnostic, n)
    gg = CUDA.CG.this_grid()
    @inbounds for i in index():stride():n
        ws.x[i] = old[i]
        ws.dissipation[i] = 0.0
    end
    CUDA.CG.sync(gg)
    # Registered rule: predicted endpoint psi+, old accepted damage; zero rates retained.
    rates!(ws.r0, ws.mask, ws.x, 1.0, rowptr, cols, ws.a0, a1, ws.b0, ws.b1, capacity, n, gg)
    bound = free_bound!(ws, 1, 1.0, rowptr, cols, ws.a0, a1, capacity, n, gg)
    @inbounds h = clock[2]
    if index() == 1
        proposed = divrn(mul(h, bound), 1.8)
        valid =
            isfinite(h) && h > 0 && isfinite(proposed) && proposed >= 0 && proposed <= 20_000_000
        @inbounds begin
            ws.ctrl[2] = valid ? max(1.0, ceil(proposed)) : 1.0
            !valid && (ws.ctrl[9] = -60.0)
            ws.ctrl[10] = 0.0
            clock[4] = bound
        end
    end
    CUDA.CG.sync(gg)
    # Every block takes the same exit, preserving the last accepted state on failure.
    if ws.ctrl[9] != 0.0
        @inbounds for i in index():stride():n
            out[i] = old[i]
        end
        return nothing
    end
    @inbounds count = Int(ws.ctrl[2])
    hs = divrn(h, Float64(count))
    for sub in 1:count
        theta0 = divrn(Float64(sub - 1), Float64(count))
        theta1 = divrn(Float64(sub), Float64(count))
        rates!(ws.r0, ws.mask, ws.x, theta0, rowptr, cols, ws.a0, a1, ws.b0, ws.b1, capacity, n, gg)
        @inbounds for i in index():stride():n
            ws.pred[i] = min(1.0, add(ws.x[i], mul(hs, ws.r0[i])))
        end
        CUDA.CG.sync(gg)
        rates!(
            ws.r1, ws.mask, ws.pred, theta1, rowptr, cols, ws.a0, a1, ws.b0, ws.b1, capacity, n, gg
        )
        @inbounds for i in index():stride():n
            before = ws.x[i]
            after = min(1.0, add(before, mul(mul(0.5, hs), add(ws.r0[i], ws.r1[i]))))
            delta = after - before
            ws.x[i] = after
            ws.dissipation[i] = add(
                ws.dissipation[i], divrn(mul(capacity[i], mul(delta, delta)), hs)
            )
        end
        CUDA.CG.sync(gg)
        if diagnostic && sub == cld(count, 2)
            rates!(
                ws.r1, ws.mask, ws.x, theta1, rowptr, cols, ws.a0, a1, ws.b0, ws.b1, capacity, n, gg
            )
            middle = free_bound!(ws, 10, theta1, rowptr, cols, ws.a0, a1, capacity, n, gg)
            index() == 1 && (@inbounds ws.ctrl[10] = mul(hs, middle))
            CUDA.CG.sync(gg)
        end
    end
    rates!(ws.r1, ws.mask, ws.x, 1.0, rowptr, cols, ws.a0, a1, ws.b0, ws.b1, capacity, n, gg)
    final_bound = free_bound!(ws, 3, 1.0, rowptr, cols, ws.a0, a1, capacity, n, gg)
    @inbounds for i in index():stride():n
        out[i] = ws.x[i]
    end
    if index() == 1
        z = mul(hs, final_bound)
        @inbounds begin
            ws.ctrl[4] = z
            ws.ctrl[5] += count
            ws.ctrl[6] = max(ws.ctrl[6], z)
            ws.ctrl[7] += Float64(z > 2.0)
            ws.ctrl[8] += Float64(z > 1.8)
            ws.ctrl[11] = max(ws.ctrl[11], Float64(count))
            ws.ctrl[12] += 1.0
            col = mod(Int(clock[3]) - 1, size(ws.history, 2)) + 1
            ws.history[1, col] = count
            ws.history[2, col] = z
            ws.history[3, col] = bound
            ws.history[4, col] = final_bound
            ws.history[5, col] = ws.ctrl[10]
        end
    end
    return nothing
end

function launch!(ws, csr, capacity, old, out, clock; blocks = 128, diagnostic = false)
    @cuda cooperative = true threads = NT blocks = blocks kernel!(
        ws, csr.rowptr, csr.cols, csr.values, capacity, old, out, clock, diagnostic, length(old)
    )
    nothing
end

function residency(ws, csr, capacity, old, out, clock; diagnostic = false)
    k = @cuda launch = false kernel!(
        ws, csr.rowptr, csr.cols, csr.values, capacity, old, out, clock, diagnostic, length(old)
    )
    Int(
        CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *
        CUDA.active_blocks(k.fun, NT),
    )
end

# Independent sparse CPU reference; no device functions or GPU assembly are used.
function cpu_bound(A, b, capacity, d)
    raw = b - A * d
    free = (d .< 1 - 1e-12) .& (raw .>= 0.0)
    any(free) ? maximum(vec(sum(abs.(A[free, free]); dims = 2)) ./ capacity[free]) : 0.0
end
function cpu_step(A0, b0, A1, b1, capacity, old, h; diagnostic = false)
    @assert isfinite(h) && h > 0 && all(isfinite, capacity) && all(capacity .> 0)
    bound = cpu_bound(A1, b1, capacity, old)
    proposed = h * bound / 1.8
    @assert isfinite(proposed) && 0 <= proposed <= 20_000_000
    count = max(1, ceil(Int, proposed))
    hs = h / count
    d = copy(old)
    diss = zeros(length(d))
    middle = 0.0
    for sub in 1:count
        t0 = (sub - 1) / count
        t1 = sub / count
        A = (1 - t0) * A0 + t0 * A1
        b = (1 - t0) * b0 + t0 * b1
        r0 = max.(b - A * d, 0.0) ./ capacity
        pred = min.(1.0, d + hs * r0)
        A = (1 - t1) * A0 + t1 * A1
        b = (1 - t1) * b0 + t1 * b1
        r1 = max.(b - A * pred, 0.0) ./ capacity
        dn = min.(1.0, d + 0.5hs * (r0 + r1))
        diss .+= capacity .* (dn - d) .^ 2 / hs
        d = dn
        diagnostic && sub == cld(count, 2) && (middle = hs * cpu_bound(A, b, capacity, d))
    end
    final_bound = cpu_bound(A1, b1, capacity, d)
    (;
        d,
        dissipation = diss,
        n_sub = count,
        predicted_bound = bound,
        final_bound,
        z_sub = hs * final_bound,
        middle_z = middle,
    )
end
end
