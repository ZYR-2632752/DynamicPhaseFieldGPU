module DeviceKW
using CUDA
using ..P1V2
const P = P1V2
const C = CUDA.CUDACore
# Explicit rounding preserves the separate baseline broadcast operations.
@inline add(x, y) = C.add_rn(x, y)
@inline mul(x, y) = C.mul_rn(x, y)
@inline divrn(x, y) = C.div_rn(x, y)

mutable struct Context{S,V}
    sim::S
    clock::V
    boundparts::V
    cap::Float64
    stop::Float64
    lattice::Bool
    phase::Module
    refinement::Int
    graph::Any
end
include("kw_diagnostics.jl")

function context(
    sim,
    cap;
    stop = Inf,
    lattice = sim.options.method == :BE && sim.options.layer == :L2,
    phase = P.OPT,
    refinement = 1,
)
    @assert sim.options.layer in (:L0, :L2) && refinement >= 1
    @assert sim.options.rule in (Symbol("H-LIM"), Symbol("H-FREE"), Symbol("H-UNLIM"))
    @assert 0 < cap <= sim.w.g.dtmechanical * (1 + 64eps()) && stop > sim.t
    if phase !== P.OPT
        @assert sim.options.method == :BE
        b = sim.scratch
        g = sim.w.g
        o = sim.options
        s = sim.state
        ws = phase.workspace(s.d; maxblocks = o.blocks)
        kernel = @cuda launch = false phase.kernel!(
            ws,
            b.descriptor,
            b.capacity,
            b.diagcap,
            b.rhs,
            s.d,
            b.upper,
            g,
            o.tolerance,
            300,
            true,
            true,
        )
        limit =
            CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *
            CUDA.active_blocks(kernel.fun, 128)
        @assert o.blocks <= limit
        sim = P.Simulation(
            sim.w,
            s,
            merge(b, (; solver = ws)),
            merge(o, (; residency = limit)),
            sim.diag,
            sim.t,
            sim.step,
            sim.host_reads,
            sim.bound_reads,
            sim.diagnostic_reads,
            sim.last_read_step,
        )
    end
    Context(
        sim,
        CuArray([
            sim.t,
            cap,
            Float64(sim.step),
            0.0,
            sim.t,
            sim.w.g.velocity_scale,
            sim.w.g.impact_speed,
            sim.w.g.ramp_time,
        ]),
        CUDA.zeros(Float64, cld(max(sim.w.g.ne, sim.w.g.n), 256)),
        Float64(cap),
        Float64(stop),
        lattice,
        phase,
        Int(refinement),
        nothing,
    )
end
function bound_parts!(parts, bounds, n)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * 256 + lane
    s = CUDA.@cuStaticSharedMem(Float64, 256)
    @inbounds s[lane] = i <= n ? bounds[i] : 0.0
    sync_threads()
    offset = 128
    while offset > 0
        lane <= offset && (@inbounds s[lane] = max(s[lane], s[lane + offset]))
        sync_threads()
        offset = div(offset, 2)
    end
    lane == 1 && (@inbounds parts[blockIdx().x] = s[1])
    return nothing
end
function clock_step!(clock, parts, diag, cap, stop, isbe, lattice, refinement)
    if threadIdx().x == 1 && blockIdx().x == 1
        bound = 0.0
        if !isbe
            @inbounds for i in eachindex(parts)
                bound = max(bound, parts[i])
            end
        end
        @inbounds begin
            h = min(cap, stop - clock[1])
            h = min(h, bound > 0 ? divrn(2.0, mul(Float64(refinement), bound)) : Inf)
            valid = isfinite(bound) && bound >= 0 && isfinite(h) && h > 0 && clock[1] < stop
            !valid && diag[1, 1] == 0 && (diag[1, 1] = -50.0)
            clock[2] = h
            clock[3] = add(clock[3], 1.0)
            clock[4] = bound
            clock[5] = lattice ? min(stop, mul(clock[3], cap)) : add(clock[1], h)
        end
    end
    return nothing
end
function position!(un, u, v, a, g, clock, isbe)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(u)
        @inbounds begin
            h = clock[2]
            half = isbe ? mul(0.5, mul(h, h)) : mul(mul(0.5, h), h)
            un[i] = add(
                mul(add(add(u[i], mul(h, v[i])), mul(half, a[i])), g.free[i]),
                mul(loading(clock[5], clock[6], clock[7], clock[8])[1], g.rate[i]),
            )
        end
    end
    return nothing
end
function velocity!(an, vn, f, v, a, g, clock)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(v)
        @inbounds begin
            _, vb, ab = loading(clock[5], clock[6], clock[7], clock[8])
            acc = add(mul(divrn(-f[i], g.mass[div(i - 1, 2) + 1]), g.free[i]), mul(ab, g.rate[i]))
            an[i] = acc
            vn[i] = add(
                mul(add(v[i], mul(mul(0.5, clock[2]), add(a[i], acc))), g.free[i]),
                mul(vb, g.rate[i]),
            )
        end
    end
    return nothing
end
function heun_predict!(rate, pred, d, phase, capacity, clock, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            r = divrn(max(phase[i], 0.0), capacity[i])
            rate[i] = r
            pred[i] = min(1.0, add(d[i], mul(clock[2], r)))
        end
    end
    return nothing
end
function heun_accept!(dn, rs, d, rate, phase, capacity, clock, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            r = divrn(max(phase[i], 0.0), capacity[i])
            rs[i] = r
            dn[i] = min(1.0, add(d[i], mul(mul(0.5, clock[2]), add(rate[i], r))))
        end
    end
    return nothing
end
function capacity_rhs!(capacity, diagcap, rhs, Cv, diagonal, b, d, clock, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            c = divrn(Cv[i], clock[2])
            capacity[i] = c
            diagcap[i] = add(diagonal[i], c)
            rhs[i] = add(b[i], mul(c, d[i]))
        end
    end
    return nothing
end
function heun_bookkeeping!(delta, capacity, dn, d, Cv, clock, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds delta[i] = dn[i] - d[i]
        capacity[i] = divrn(Cv[i], clock[2])
    end
    return nothing
end
function finish!(diag, pred, mech, energy, before, after, ctrl, clock, isbe, oldforce, tol)
    @inbounds finish_kw_kernel!(
        diag,
        pred,
        mech,
        energy,
        before,
        after,
        ctrl,
        clock[2],
        clock[5],
        Int(clock[3]),
        isbe,
        oldforce,
        tol,
        clock[4],
        clock[1],
        clock[6],
        clock[7],
        clock[8],
    )
    if threadIdx().x == 1 && blockIdx().x == 1
        @inbounds clock[1] = clock[5]
    end
    return nothing
end
function enqueue!(ctx)
    sim = ctx.sim
    w = sim.w
    g = w.g
    s = sim.state
    b = sim.scratch
    o = sim.options
    clock = ctx.clock
    isbe = o.method == :BE
    nb = cld(g.n, 256)
    ne = cld(g.ne, 256)
    oldforce = o.layer == :L0
    if !isbe && o.rule == Symbol("H-LIM")
        @cuda threads = 256 blocks = ne P.phase_bound_kernel!(w.bounds, s.cached, g)
        @cuda threads = 256 blocks = length(ctx.boundparts) bound_parts!(
            ctx.boundparts, w.bounds, g.ne
        )
    elseif !isbe && o.rule == Symbol("H-FREE")
        P.assemble!(b.sp, s.cached, g)
        csr = b.sp.global_csr
        @cuda threads = 256 blocks = nb P.free_bound_kernel!(
            b.node_bounds, s.d, w.phase, csr.rowptr, csr.cols, csr.values, g.C, g.n
        )
        @cuda threads = 256 blocks = length(ctx.boundparts) bound_parts!(
            ctx.boundparts, b.node_bounds, g.n
        )
    elseif !isbe
        fill!(ctx.boundparts, 0.0)
    end
    @cuda threads = 1 blocks = 1 clock_step!(
        clock, ctx.boundparts, sim.diag, ctx.cap, ctx.stop, isbe, ctx.lattice, ctx.refinement
    )
    @cuda threads = 256 blocks = cld(2g.n, 256) position!(b.un, s.u, s.v, s.a, g, clock, isbe)
    if isbe
        P.main_operators!(w, b.un, s.d)
        copyto!(b.qp, w.qp)
        if oldforce
            @cuda threads = 256 blocks = cld(2g.n, 256) velocity!(
                b.an, b.vn, w.f, s.v, s.a, g, clock
            )
        end
        @cuda threads = 256 blocks = size(b.predicted_parts, 2) mechanics_kw_kernel!(
            b.predicted_parts,
            w.elastic,
            w.fracture,
            w.f,
            g.rate,
            g.mass,
            oldforce ? b.vn : s.v,
            oldforce ? b.an : s.a,
            g.n,
            g.ne,
        )
        copyto!(b.q, oldforce ? s.cached : b.qp)
        P.coefficients!(b.b, b.diagonal, b.q, g)
        @cuda threads = 256 blocks = nb capacity_rhs!(
            b.capacity, b.diagcap, b.rhs, g.C, b.diagonal, b.b, s.d, clock, g.n
        )
        o.representation == :SP && P.assemble!(b.sp, b.q, g)
        ctx.phase.launch_box!(
            b.solver,
            b.descriptor,
            b.capacity,
            b.diagcap,
            b.rhs,
            s.d,
            b.upper,
            g;
            tolerance = o.tolerance,
            blocks = o.blocks,
            skip_fixed = o.skip_fixed,
            fused = o.fused,
        )
        copyto!(b.dn, b.solver.x)
        P.action!(b.z, b.dn, b.q, g)
        @cuda threads = 256 blocks = nb P.phase_gradient_kernel!(
            b.delta, b.gradient, b.dn, s.d, b.z, b.b, b.capacity, g.n
        )
        P.audit!(sim, b.audit_before, s.v)
        @cuda threads = 256 blocks = nb P.maybe_correct_kernel!(
            b.dn, b.gradient, b.diagcap, s.d, b.upper, b.audit_before, o.tolerance, g.n
        )
        @cuda threads = 256 blocks = nb P.maybe_action_kernel!(
            b.z, b.dn, b.q, g, b.audit_before, o.tolerance
        )
        @cuda threads = 256 blocks = nb P.maybe_gradient_kernel!(
            b.delta, b.gradient, b.dn, s.d, b.z, b.b, b.capacity, b.audit_before, o.tolerance, g.n
        )
        P.action!(b.z, b.delta, b.q, g)
        P.main_operators!(w, b.un, b.dn)
    else
        copyto!(b.q, s.cached)
        @cuda threads = 256 blocks = nb heun_predict!(b.rate, b.pred, s.d, w.phase, g.C, clock, g.n)
        P.main_operators!(w, b.un, b.pred)
        @cuda threads = 256 blocks = nb heun_accept!(
            b.dn, b.rs, s.d, b.rate, w.phase, g.C, clock, g.n
        )
        P.main_operators!(w, b.un, b.dn)
        copyto!(b.qp, w.qp)
        @cuda threads = 256 blocks = nb heun_bookkeeping!(
            b.delta, b.capacity, b.dn, s.d, g.C, clock, g.n
        )
        # These arrays remain zero for every explicit step after initialization.
    end
    if !oldforce || !isbe
        @cuda threads = 256 blocks = cld(2g.n, 256) velocity!(b.an, b.vn, w.f, s.v, s.a, g, clock)
    end
    P.audit!(sim, b.audit_after, b.vn; isbe)
    if isbe
        @cuda threads = 256 blocks = ne P.supply_kernel!(b.supply, b.qp, b.q, s.d, b.dn, g)
    end
    @cuda threads = 256 blocks = size(b.mechanics_parts, 2) mechanics_kw_kernel!(
        b.mechanics_parts, w.elastic, w.fracture, w.f, g.rate, g.mass, b.vn, b.an, g.n, g.ne
    )
    @cuda threads = 256 blocks = size(b.energy_parts, 2) energy_kw_kernel!(
        b.energy_parts,
        b.delta,
        b.z,
        b.gradient,
        b.capacity,
        w.elastic,
        w.fracture,
        b.supply,
        w.f,
        g.rate,
        g.mass,
        b.an,
        g.n,
        g.ne,
    )
    @cuda threads = 1 blocks = 1 finish!(
        sim.diag,
        b.predicted_parts,
        b.mechanics_parts,
        b.energy_parts,
        b.audit_before,
        b.audit_after,
        b.solver.ctrl,
        clock,
        isbe,
        oldforce,
        o.tolerance,
    )
    s.H .= max.(s.H, b.qp)
    copyto!(s.cached, b.qp)
    swap!(sim)
    nothing
end
function swap!(sim)
    s = sim.state
    b = sim.scratch
    sim.state = (; u = b.un, v = b.vn, a = b.an, d = b.dn, H = s.H, cached = s.cached)
    sim.scratch = merge(b, (; un = s.u, vn = s.v, an = s.a, dn = s.d))
    nothing
end
function capture!(ctx)
    # Call enqueue! twice on a throwaway context first, to compile both parities.
    @assert isnothing(ctx.graph)
    CUDA.synchronize()
    origin = ctx.sim.state.u
    first = CUDA.capture() do
        enqueue!(ctx)
        enqueue!(ctx)
    end
    swap!(ctx.sim)
    second = CUDA.capture() do
        enqueue!(ctx)
        enqueue!(ctx)
    end
    swap!(ctx.sim)
    ctx.graph = (
        first = CUDA.instantiate(first), second = CUDA.instantiate(second), origin = origin
    )
    ctx
end
function advance!(ctx, count; read = true)
    @assert 1 <= count <= ctx.sim.options.read_every
    if !isnothing(ctx.graph)
        graph = ctx.sim.state.u === ctx.graph.origin ? ctx.graph.first : ctx.graph.second
        for _ in 1:div(count, 2)
            CUDA.launch(graph)
        end
        isodd(count) && enqueue!(ctx)
    else
        for _ in 1:count
            enqueue!(ctx)
        end
    end
    ctx.sim.step += count
    if read
        CUDA.synchronize()
        result = P.read_diagnostics!(ctx.sim)
        ctx.sim.t = result.current.time_s
        return result
    end
    nothing
end
end
