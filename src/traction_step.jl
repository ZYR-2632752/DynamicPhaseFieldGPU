module TractionGPU
using CUDA, LinearAlgebra
using ..P1V2, ..DeviceL1, ..PhaseSubcycle
const P = P1V2
const D = DeviceL1
const H = PhaseSubcycle
const add = D.add
const mul = D.mul
const divrn = D.divrn

mutable struct Context{S,V,W}
    sim::S
    clock::V
    boundparts::V
    workspace::W
    cap::Float64
    stop::Float64
    refinement::Float64
    lattice::Bool
    phase::Module
    method::Symbol
    history_box::Bool
    diagnostic::Bool
    graph::Any
end

function initialize(
    base, cap; method = :BE, stop = 0.0006, refinement = 1, seed = nothing, diagnostic = false
)
    @assert method in (:BE, :Heun, :HSUB, :HBOX)
    @assert method in (:BE, :HBOX) || base.eta > 0
    @assert 0 < cap <= base.dtmechanical * (1 + 64eps()) && stop > 0 && refinement >= 1
    # P1V2.buffers converts known base fields. Convert the three adapter fields
    # here as well, so the complete grid argument remains GPU-compatible.
    gpu_base = merge(
        base,
        (;
            upper = CuArray(base.upper),
            loads = CuArray(base.loads),
            reaction_mask = CuArray(base.reaction_mask),
        ),
    )
    if isnothing(seed)
        seed = (;
            u = zeros(2, base.n),
            v = zeros(2, base.n),
            a = base.loads ./ reshape(base.mass, 1, :) .* base.free,
            d = zeros(base.n),
            H = zeros(base.ne),
            cached = zeros(base.ne),
            t = 0.0,
            step = 0,
        )
    end
    rule = method == :HSUB ? Symbol("H-FREE") : Symbol("H-LIM")
    sim = P.initialize(
        gpu_base;
        method = method in (:BE, :HBOX) ? :BE : :Heun,
        rule,
        layer = :L2,
        seed,
        read_every = 100,
        blocks = 0,
        tolerance = 1e-12,
        representation = :MF,
    )
    b = sim.scratch
    scratch = merge(b, (; upper = sim.w.g.upper, lower = CUDA.zeros(Float64, base.n)))
    sim = P.Simulation(
        sim.w,
        sim.state,
        scratch,
        sim.options,
        sim.diag,
        sim.t,
        sim.step,
        sim.host_reads,
        sim.bound_reads,
        sim.diagnostic_reads,
        sim.last_read_step,
    )
    clock = CuArray([sim.t, cap, Float64(sim.step), 0.0, sim.t])
    workspace = if method == :HSUB
        csr = scratch.sp.global_csr
        ws = H.workspace(
            sim.state.d, length(csr.values); read_every = 100, blocks = sim.options.blocks
        )
        limit = H.residency(ws, csr, sim.w.g.C, sim.state.d, scratch.dn, clock; diagnostic)
        @assert limit >= sim.options.blocks "Traction subcycling exceeds measured cooperative residency"
        ws
    else
        nothing
    end
    Context(
        sim,
        clock,
        CUDA.zeros(Float64, cld(max(base.n, base.ne), 256)),
        workspace,
        Float64(cap),
        Float64(stop),
        Float64(refinement),
        method != :Heun,
        P.OPT,
        method,
        method == :HBOX,
        Bool(diagnostic),
        nothing,
    )
end

function clock_step!(clock, parts, diag, cap, stop, limited, lattice, refinement)
    if threadIdx().x == 1 && blockIdx().x == 1
        bound = 0.0
        if limited
            @inbounds for i in eachindex(parts)
                bound = max(bound, parts[i])
            end
        end
        @inbounds begin
            next = add(clock[3], 1.0)
            h = min(cap, stop - clock[1])
            h = min(h, bound > 0 ? divrn(divrn(2.0, bound), refinement) : Inf)
            target = lattice ? min(stop, mul(next, cap)) : min(stop, add(clock[1], h))
            h = target - clock[1]
            valid = isfinite(bound) && bound >= 0 && isfinite(h) && h > 0
            !valid && diag[1, 1] == 0 && (diag[1, 1] = -50.0)
            clock[2] = h
            clock[3] = next
            clock[4] = bound
            clock[5] = target
        end
    end
    return nothing
end

function traction_velocity!(an, vn, f, v, a, g, clock)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(v)
        @inbounds begin
            acc = mul(divrn(g.loads[i] - f[i], g.mass[div(i - 1, 2) + 1]), g.free[i])
            an[i] = acc
            vn[i] = mul(add(v[i], mul(mul(0.5, clock[2]), add(a[i], acc))), g.free[i])
        end
    end
    return nothing
end

function positive_predict!(rate, pred, d, phase, g, clock)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= g.n
        @inbounds begin
            r = g.upper[i] > 0 ? divrn(max(phase[i], 0.0), g.C[i]) : 0.0
            rate[i] = r
            pred[i] = min(g.upper[i], add(d[i], mul(clock[2], r)))
        end
    end
    return nothing
end
function positive_accept!(dn, rs, d, rate, phase, g, clock)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= g.n
        @inbounds begin
            r = g.upper[i] > 0 ? divrn(max(phase[i], 0.0), g.C[i]) : 0.0
            rs[i] = r
            dn[i] = min(g.upper[i], add(d[i], mul(mul(0.5, clock[2]), add(rate[i], r))))
        end
    end
    return nothing
end

function eliminate_fixed_csr!(rowptr, cols, values, b, upper, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            upper[i] == 0 && (b[i] = 0.0)
            for k in rowptr[i]:(rowptr[i + 1] - 1)
                (upper[i] == 0 || upper[cols[k]] == 0) && (values[k] = 0.0)
            end
        end
    end
    return nothing
end

function audit_blocks!(
    parts, un, vn, dn, old, lower, upper, gradient, diagcap, n, isbe, history_box
)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    shared = CUDA.@cuStaticSharedMem(Float64, (4, 256))
    @inbounds if i <= n
        valid =
            isfinite(dn[i]) &&
            0 <= dn[i] <= upper[i] &&
            isfinite(un[1, i]) &&
            isfinite(un[2, i]) &&
            isfinite(vn[1, i]) &&
            isfinite(vn[2, i])
        kkt = 0.0
        if isbe
            valid = valid && isfinite(gradient[i]) && isfinite(diagcap[i]) && diagcap[i] > 0
            kkt =
                valid ? abs(clamp(dn[i] - gradient[i] / diagcap[i], lower[i], upper[i]) - dn[i]) :
                Inf
        end
        delta = dn[i] - old[i]
        valid = valid && (history_box || delta >= 0)
        shared[1, lane] = delta
        shared[2, lane] = dn[i]
        shared[3, lane] = kkt
        shared[4, lane] = Float64(!valid)
    else
        shared[1, lane] = Inf
        shared[2, lane] = -Inf
        shared[3, lane] = 0.0
        shared[4, lane] = 0.0
    end
    sync_threads()
    offset = 128
    while offset > 0
        if lane <= offset
            @inbounds begin
                shared[1, lane] = min(shared[1, lane], shared[1, lane + offset])
                for row in 2:4
                    shared[row, lane] = max(shared[row, lane], shared[row, lane + offset])
                end
            end
        end
        sync_threads()
        offset = div(offset, 2)
    end
    lane == 1 && (@inbounds for row in 1:4
        parts[row, blockIdx().x] = shared[row, 1]
    end)
    return nothing
end
function audit!(ctx, out, velocity, isbe)
    sim = ctx.sim
    b = sim.scratch
    s = sim.state
    g = sim.w.g
    @cuda threads = 256 blocks = size(b.audit_parts, 2) audit_blocks!(
        b.audit_parts,
        b.un,
        velocity,
        b.dn,
        s.d,
        b.lower,
        g.upper,
        b.gradient,
        b.diagcap,
        g.n,
        isbe,
        ctx.history_box,
    )
    @cuda threads = 1 blocks = 1 P.finish_audit_kernel!(out, b.audit_parts)
    nothing
end

function mechanics_parts!(parts, elastic, fracture, f, u, un, v, g)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    shared = CUDA.@cuStaticSharedMem(Float64, (5, 256))
    @inbounds begin
        shared[1, lane] = i <= g.ne ? elastic[i] : 0.0
        shared[2, lane] = i <= g.ne ? fracture[i] : 0.0
        shared[3, lane] = i <= g.n ? -f[1, i] * g.reaction_mask[i] : 0.0
        shared[4, lane] = i <= g.n ? g.mass[i] * (v[1, i]^2 + v[2, i]^2) : 0.0
        shared[5, lane] =
            i <= g.n ? g.loads[1, i] * (un[1, i] - u[1, i]) + g.loads[2, i] * (un[2, i] - u[2, i]) :
            0.0
    end
    sync_threads()
    offset = 128
    while offset > 0
        if lane <= offset
            @inbounds for row in 1:5
                shared[row, lane] += shared[row, lane + offset]
            end
        end
        sync_threads()
        offset = div(offset, 2)
    end
    lane == 1 && (@inbounds for row in 1:5
        parts[row, blockIdx().x] = shared[row, 1]
    end)
    return nothing
end
function dissipation_parts!(parts, diss, n)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    shared = CUDA.@cuStaticSharedMem(Float64, 256)
    @inbounds shared[lane] = i <= n ? diss[i] : 0.0
    sync_threads()
    offset = 128
    while offset > 0
        lane <= offset && (@inbounds shared[lane] += shared[lane + offset])
        sync_threads()
        offset = div(offset, 2)
    end
    lane == 1 && (@inbounds parts[3, blockIdx().x] = shared[1])
    return nothing
end

function finish!(
    diag,
    pred,
    mech,
    energy,
    before,
    after,
    ctrl,
    subctrl,
    clock,
    isbe,
    issub,
    tol,
    history_box,
    diagnostic,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        kin = 0.5 * P.sum_parts(mech, 4)
        elastic = P.sum_parts(energy, 4)
        fracture = P.sum_parts(energy, 5)
        newE = elastic + fracture
        reaction = P.sum_parts(mech, 3)
        workstep = P.sum_parts(mech, 5)
        predicted = P.sum_parts(pred, 1) + P.sum_parts(pred, 2)
        curv = isbe ? 0.5 * P.sum_parts(energy, 1) : 0.0
        normal = isbe ? -P.sum_parts(energy, 2) : 0.0
        diss = P.sum_parts(energy, 3)
        supply = isbe ? P.sum_parts(energy, 6) : 0.0
        @inbounds begin
            identity = isbe ? newE - predicted + diss + curv + normal - supply : 0.0
            mechstep =
                isbe ? kin - diag[10, 1] + predicted - diag[11, 1] - diag[12, 1] - workstep : 0.0
            diag[13, 1] += workstep
            diag[14, 1] += diss
            diag[16, 1] += curv
            diag[17, 1] += normal
            diag[18, 1] += supply
            diag[19, 1] += mechstep
            total = kin + newE + diag[14, 1] - diag[13, 1]
            physical = total - diag[27, 1] - diag[28, 1]
            balance = physical + diag[16, 1] + diag[17, 1] - diag[18, 1] - diag[19, 1]
            kkt = isbe ? after[3] : 0.0
            residual = isbe ? ctrl[2] : 0.0
            correction = isbe && P.correction_needed(before, tol) ? before[3] : 0.0
            cg = isbe ? ctrl[4] : 0.0
            outer = isbe ? ctrl[3] : 0.0
            actions = if isbe
                ctrl[5] + 2 + Float64(correction > 0)
            elseif issub
                2subctrl[2] + 2 + Float64(diagnostic)
            else
                2.0
            end
            failure = 0.0
            isbe && !(ctrl[1] == 1 && isfinite(residual) && residual <= tol) && (failure = -10.0)
            !(
                after[4] == 0 &&
                isfinite(after[1]) &&
                (history_box || after[1] >= 0) &&
                after[2] <= 1 &&
                isfinite(kkt) &&
                kkt <= tol
            ) && (failure = -20.0)
            !(isfinite(physical) && isfinite(kin) && isfinite(elastic) && isfinite(fracture)) &&
                (failure = -30.0)
            scale = max(abs(diag[13, 1]), abs(newE), 1.0)
            isbe &&
                !(isfinite(identity) && abs(identity) <= 1e-8scale && abs(balance) <= 1e-8scale) &&
                (failure = -40.0)
            issub && subctrl[9] != 0 && (failure = subctrl[9])
            diag[1, 1] == 0 && (diag[1, 1] = failure)
            diag[2, 1] = max(diag[2, 1], isfinite(kkt) ? kkt : Inf)
            diag[3, 1] = max(diag[3, 1], isfinite(residual) ? residual : Inf)
            diag[4, 1] = min(diag[4, 1], after[1])
            diag[5, 1] = after[2]
            diag[6, 1] = clock[5]
            diag[7, 1] = clock[2]
            diag[8, 1] = clock[3]
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
            diag[29, 1] = reaction
            diag[30, 1] = kkt
            diag[31, 1] = residual
            diag[32, 1] = cg
            diag[33, 1] = outer
            diag[34, 1] = actions
            diag[35, 1] = after[1]
            diag[36, 1] = clock[4]
            diag[37, 1] = total
            column = mod(Int(clock[3]) - 1, size(diag, 2) - 1) + 2
            for row in 1:P.NDIAG
                diag[row, column] = diag[row, 1]
            end
            clock[1] = clock[5]
        end
    end
    return nothing
end

function D.enqueue!(ctx::Context)
    sim = ctx.sim
    w = sim.w
    g = w.g
    s = sim.state
    b = sim.scratch
    o = sim.options
    clock = ctx.clock
    isbe = ctx.method in (:BE, :HBOX)
    issub = ctx.method == :HSUB
    limited = ctx.method == :Heun
    nb = cld(g.n, 256)
    ne = cld(g.ne, 256)
    if limited
        @cuda threads = 256 blocks = ne P.phase_bound_kernel!(w.bounds, s.cached, g)
        @cuda threads = 256 blocks = length(ctx.boundparts) D.bound_parts!(
            ctx.boundparts, w.bounds, g.ne
        )
    end
    @cuda threads = 1 blocks = 1 clock_step!(
        clock, ctx.boundparts, sim.diag, ctx.cap, ctx.stop, limited, ctx.lattice, ctx.refinement
    )
    @cuda threads = 256 blocks = cld(2g.n, 256) D.position!(b.un, s.u, s.v, s.a, g, clock, isbe)
    if isbe
        P.main_operators!(w, b.un, s.d)
        copyto!(b.qp, w.qp)
        @cuda threads = 256 blocks = size(b.predicted_parts, 2) mechanics_parts!(
            b.predicted_parts, w.elastic, w.fracture, w.f, s.u, b.un, s.v, g
        )
        if ctx.history_box
            b.q .= max.(s.H, b.qp)
            fill!(b.lower, 0.0)
        else
            copyto!(b.q, b.qp)
            copyto!(b.lower, s.d)
        end
        P.coefficients!(b.b, b.diagonal, b.q, g)
        @cuda threads = 256 blocks = nb D.capacity_rhs!(
            b.capacity, b.diagcap, b.rhs, g.C, b.diagonal, b.b, s.d, clock, g.n
        )
        ctx.phase.launch_box!(
            b.solver,
            b.descriptor,
            b.capacity,
            b.diagcap,
            b.rhs,
            b.lower,
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
        audit!(ctx, b.audit_before, s.v, true)
        @cuda threads = 256 blocks = nb P.maybe_correct_kernel!(
            b.dn, b.gradient, b.diagcap, b.lower, b.upper, b.audit_before, o.tolerance, g.n
        )
        @cuda threads = 256 blocks = nb P.maybe_action_kernel!(
            b.z, b.dn, b.q, g, b.audit_before, o.tolerance
        )
        @cuda threads = 256 blocks = nb P.maybe_gradient_kernel!(
            b.delta, b.gradient, b.dn, s.d, b.z, b.b, b.capacity, b.audit_before, o.tolerance, g.n
        )
        P.action!(b.z, b.delta, b.q, g)
        P.main_operators!(w, b.un, b.dn)
    elseif issub
        ws = ctx.workspace
        csr = b.sp.global_csr
        copyto!(b.q, s.cached)
        P.assemble!(b.sp, b.q, g)
        P.coefficients!(ws.b0, b.diagonal, b.q, g)
        @cuda threads = 256 blocks = nb eliminate_fixed_csr!(
            csr.rowptr, csr.cols, csr.values, ws.b0, g.upper, g.n
        )
        copyto!(ws.a0, csr.values)
        P.main_operators!(w, b.un, s.d)
        copyto!(b.qp, w.qp)
        P.assemble!(b.sp, b.qp, g)
        P.coefficients!(ws.b1, b.diagonal, b.qp, g)
        @cuda threads = 256 blocks = nb eliminate_fixed_csr!(
            csr.rowptr, csr.cols, csr.values, ws.b1, g.upper, g.n
        )
        H.launch!(ws, csr, g.C, s.d, b.dn, clock; blocks = o.blocks, diagnostic = ctx.diagnostic)
        P.main_operators!(w, b.un, b.dn)
        @cuda threads = 256 blocks = nb D.heun_bookkeeping!(
            b.delta, b.capacity, b.dn, s.d, g.C, clock, g.n
        )
    else
        copyto!(b.q, s.cached)
        @cuda threads = 256 blocks = nb positive_predict!(b.rate, b.pred, s.d, w.phase, g, clock)
        P.main_operators!(w, b.un, b.pred)
        @cuda threads = 256 blocks = nb positive_accept!(b.dn, b.rs, s.d, b.rate, w.phase, g, clock)
        P.main_operators!(w, b.un, b.dn)
        copyto!(b.qp, w.qp)
        @cuda threads = 256 blocks = nb D.heun_bookkeeping!(
            b.delta, b.capacity, b.dn, s.d, g.C, clock, g.n
        )
    end
    @cuda threads = 256 blocks = cld(2g.n, 256) traction_velocity!(
        b.an, b.vn, w.f, s.v, s.a, g, clock
    )
    audit!(ctx, b.audit_after, b.vn, isbe)
    if isbe
        @cuda threads = 256 blocks = ne P.supply_kernel!(b.supply, b.qp, b.q, s.d, b.dn, g)
    end
    @cuda threads = 256 blocks = size(b.mechanics_parts, 2) mechanics_parts!(
        b.mechanics_parts, w.elastic, w.fracture, w.f, s.u, b.un, b.vn, g
    )
    @cuda threads = 256 blocks = size(b.energy_parts, 2) P.energy_reduction_kernel!(
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
        g.n,
        g.ne,
    )
    issub && @cuda threads = 256 blocks = size(b.energy_parts, 2) dissipation_parts!(
        b.energy_parts, ctx.workspace.dissipation, g.n
    )
    subctrl = issub ? ctx.workspace.ctrl : b.solver.ctrl
    @cuda threads = 1 blocks = 1 finish!(
        sim.diag,
        b.predicted_parts,
        b.mechanics_parts,
        b.energy_parts,
        b.audit_before,
        b.audit_after,
        b.solver.ctrl,
        subctrl,
        clock,
        isbe,
        issub,
        o.tolerance,
        ctx.history_box,
        ctx.diagnostic,
    )
    s.H .= max.(s.H, b.qp)
    copyto!(s.cached, b.qp)
    D.swap!(sim)
    nothing
end

function subcycle_summary(ctx)
    ctx.method == :HSUB || return Dict{String,Any}()
    c = Array(ctx.workspace.ctrl)
    ctx.sim.host_reads += 1
    Dict(
        "mechanical_steps" => Int(c[12]),
        "total_substeps" => Int(c[5]),
        "maximum_substeps" => Int(c[11]),
        "maximum_z_sub" => c[6],
        "count_z_sub_above_2" => Int(c[7]),
        "count_z_sub_above_1p8" => Int(c[8]),
        "fraction_z_sub_above_2" => c[12] > 0 ? c[7] / c[12] : 0.0,
        "subcycle_failure" => c[9],
        "adaptive_scheme2_required" => c[7] > 0.01c[12] || c[6] > 2.2 || c[9] != 0,
        "fixed_damage_treatment" => "zero CSR rows/columns and RHS; fixed zero damage remains exact; zero eigenvalues do not change either spectral maximum",
    )
end
end
