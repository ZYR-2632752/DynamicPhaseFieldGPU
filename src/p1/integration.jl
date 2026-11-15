module P1V2
include("mesh.jl")
include("mechanics.jl")
include("kw_mesh.jl")
include("mesh_io.jl")
include("box_baseline.jl")
include("phase_matrix.jl")
include("box_solver.jl")
include("reductions.jl")
const OPT = BoxKWOptimized
const DIAG_NAMES = (
    :failure,
    :max_KKT,
    :max_solver_residual,
    :min_damage_increment,
    :damage_max,
    :time_s,
    :dt_s,
    :step,
    :reaction,
    :kinetic,
    :elastic,
    :fracture,
    :work,
    :viscous,
    :physical_balance,
    :curvature,
    :normal_work,
    :driver_supply,
    :mechanical_defect,
    :accounting_error,
    :phase_identity,
    :CG_total,
    :AS_total,
    :operator_actions,
    :corrections,
    :max_correction,
    :initialK,
    :initialE,
    :accepted_reaction,
    :KKT,
    :solver_residual,
    :CG,
    :AS,
    :actions,
    :min_delta_step,
    :phase_bound,
    :physical_total,
)
const NDIAG = length(DIAG_NAMES)

function make_grid(p)
    Pi_eta = p.eta_Pa_s * p.c_R_m_s / p.gc_N_m
    kappa = p.kappa
    @assert isfinite(Pi_eta) && Pi_eta >= 0 && isfinite(kappa) && kappa >= 0
    @assert isfinite(p.c_R_m_s) && p.c_R_m_s > 0
    base = grid(p)
    eta = Float64(Pi_eta) * base.gc / p.c_R_m_s
    merge(
        base,
        (;
            eta,
            C = eta * base.volume,
            kappa = Float64(kappa),
            Pi_eta = Float64(Pi_eta),
            c_R = p.c_R_m_s,
            dtmechanical = base.dtmechanical / sqrt(1 + kappa),
        ),
    )
end

# advance! mutates/reuses these buffers. Copy a state before retaining old frames.
# diag[:,1] is current/cumulative data; remaining columns form a device ring.
mutable struct Simulation{W,S,B,O,D}
    w::W
    state::S
    scratch::B
    options::O
    diag::D
    t::Float64
    step::Int
    host_reads::Int
    bound_reads::Int
    diagnostic_reads::Int
    last_read_step::Int
end

seedget(seed, key, default) = isnothing(seed) ? default : get(seed, key, default)
device_copy(x) = x isa CuArray ? copy(x) : CuArray(x)

function sum_parts(parts, row)
    value = 0.0
    @inbounds for b in 1:size(parts, 2)
        value += parts[row, b]
    end
    value
end

# Single-thread device bookkeeping: fixed summation order, no host scalars.
function initial_diagnostics_kernel!(diag, parts, initialK_given, initialE_given, loading_rate)
    if threadIdx().x == 1 && blockIdx().x == 1
        kin = 0.5sum_parts(parts, 4)
        elastic = sum_parts(parts, 1)
        fracture = sum_parts(parts, 2)
        reaction = sum_parts(parts, 3) / loading_rate
        @inbounds begin
            diag[9, 1] = reaction
            diag[10, 1] = kin
            diag[11, 1] = elastic
            diag[12, 1] = fracture
            !initialK_given && (diag[27, 1] = kin)
            !initialE_given && (diag[28, 1] = elastic + fracture)
            diag[29, 1] = reaction
            diag[37, 1] = kin + elastic + fracture + diag[14, 1] - diag[13, 1]
            diag[15, 1] = diag[37, 1] - diag[27, 1] - diag[28, 1]
        end
    end
    return nothing
end

function initialize(
    base;
    method = :BE,
    layer = :L2,
    rule = Symbol("H-LIM"),
    mode = :production,
    read_every = 100,
    blocks = 0,
    tolerance = 1e-12,
    representation = :MF,
    seed = nothing,
    skip_fixed = true,
    fused = true,
)
    method = Symbol(method)
    method in (:HEUN, :heun) && (method = :Heun)
    layer = Symbol(layer)
    rule = Symbol(replace(String(rule), "_" => "-"))
    mode = Symbol(mode)
    mode == :diag && (mode = :diagnostic)
    representation = Symbol(representation)
    @assert method in (:BE, :Heun) && layer in (:L0, :L1, :L2)
    @assert rule in (Symbol("H-LIM"), Symbol("H-FREE"), Symbol("H-UNLIM"))
    @assert mode in (:production, :diagnostic) && read_every >= 1 && blocks >= 0
    @assert representation in (:MF, :SP) && isfinite(tolerance) && tolerance > 0
    @assert method != :Heun || base.eta > 0 "Explicit Heun requires eta>0"
    w = buffers(base; gpu = true)
    g = w.g
    state = (;
        u = device_copy(seedget(seed, :u, zeros(2, g.n))),
        v = device_copy(seedget(seed, :v, base.rate)),
        a = device_copy(seedget(seed, :a, zeros(2, g.n))),
        d = device_copy(seedget(seed, :d, zeros(g.n))),
        H = device_copy(seedget(seed, :H, zeros(g.ne))),
        cached = device_copy(seedget(seed, :cached, zeros(g.ne))),
    )
    @assert size(state.u) == size(state.v) == size(state.a) == (2, g.n)
    @assert length(state.d) == g.n && length(state.H) == length(state.cached) == g.ne
    main_operators!(w, state.u, state.d)
    if isnothing(seed) || !haskey(seed, :cached)
        copyto!(state.cached, w.qp)
    end
    state.H .= max.(state.H, w.qp)
    q = copy(state.cached)
    solver = OPT.workspace(state.d; maxblocks = blocks == 0 ? 128 : blocks)
    # CSR is also used for an exact free-principal-matrix Gershgorin row sum.
    needs_sp = representation == :SP || (method == :Heun && rule == Symbol("H-FREE"))
    sp = needs_sp ? sparse_workspace(base; gpu = true) : nothing
    descriptor =
        representation == :SP ? (; kind = Val(:csr), sp.global_csr...) : (; kind = Val(:direct), q)
    nb = cld(max(g.n, g.ne), 256)
    scratch = (;
        solver,
        sp,
        descriptor,
        q,
        qp = copy(state.cached),
        un = similar(state.u),
        vn = similar(state.v),
        an = similar(state.a),
        dn = similar(state.d),
        pred = similar(state.d),
        rate = similar(state.d),
        rs = similar(state.d),
        b = zero(state.d),
        diagonal = zero(state.d),
        capacity = zero(state.d),
        diagcap = zero(state.d),
        rhs = zero(state.d),
        delta = zero(state.d),
        gradient = zero(state.d),
        z = zero(state.d),
        upper = CUDA.ones(Float64, g.n),
        supply = CUDA.zeros(Float64, g.ne),
        node_bounds = CUDA.zeros(Float64, g.n),
        audit_parts = CUDA.zeros(Float64, 4, cld(g.n, 256)),
        audit_before = CUDA.zeros(Float64, 4),
        audit_after = CUDA.zeros(Float64, 4),
        predicted_parts = CUDA.zeros(Float64, 7, nb),
        mechanics_parts = CUDA.zeros(Float64, 7, nb),
        energy_parts = CUDA.zeros(Float64, 7, nb),
    )
    t = Float64(seedget(seed, :t, seedget(seed, :time_s, 0.0)))
    step = Int(seedget(seed, :step, 0))
    hd = zeros(Float64, NDIAG, read_every + 1)
    hd[6, 1] = t
    hd[8, 1] = step
    for (row, key) in
        ((13, :work), (14, :visc), (16, :curv), (17, :normal), (18, :supply), (19, :mech))
        hd[row, 1] = seedget(seed, key, 0.0)
    end
    initialK_given = !isnothing(seed) && haskey(seed, :initialK)
    initialE_given = (!isnothing(seed) && haskey(seed, :initialE)) || t > 0
    hd[27, 1] = seedget(seed, :initialK, 0.0)
    hd[28, 1] = seedget(seed, :initialE, 0.0)
    diag = CuArray(hd)
    loading_rate = get(base, :loading_rate, 0.01)
    @cuda threads = 256 blocks = nb mechanics_reduction_kernel!(
        scratch.mechanics_parts, w.elastic, w.fracture, w.f, g.rate, g.mass, state.v, g.n, g.ne
    )
    @cuda threads = 1 blocks = 1 initial_diagnostics_kernel!(
        diag, scratch.mechanics_parts, initialK_given, initialE_given, loading_rate
    )
    # Seeded L0 work uses its saved old-damage reaction, not a recomputed accepted force.
    if !isnothing(seed) && (haskey(seed, :oldR) || haskey(seed, :reaction))
        saved_reaction = Float64(seedget(seed, :oldR, seedget(seed, :reaction, 0.0)))
        @cuda threads = 1 blocks = 1 seed_reaction_kernel!(diag, saved_reaction)
    end
    kernel = @cuda launch = false OPT.kernel!(
        solver,
        descriptor,
        scratch.capacity,
        scratch.diagcap,
        scratch.rhs,
        state.d,
        scratch.upper,
        g,
        tolerance,
        300,
        skip_fixed,
        fused,
    )
    residency =
        CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *
        CUDA.active_blocks(kernel.fun, OPT.NT)
    # Limit the cooperative grid to the capacity of the current device.
    blocks = blocks == 0 ? min(128, residency) : blocks
    @assert 1 <= blocks <= residency "Cooperative launch exceeds measured residency"
    options = (;
        method,
        layer,
        rule,
        mode,
        read_every,
        blocks,
        tolerance,
        representation,
        residency,
        skip_fixed,
        fused,
        loading_rate,
    )
    Simulation(w, state, scratch, options, diag, t, step, 0, 0, 0, step)
end

function seed_reaction_kernel!(diag, value)
    threadIdx().x == 1 && blockIdx().x == 1 && (diag[9, 1] = value)
    return nothing
end

function phase_gradient_kernel!(delta, gradient, dn, lower, z, b, capacity, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            delta[i] = dn[i] - lower[i]
            gradient[i] = capacity[i] * delta[i] + z[i] - b[i]
        end
    end
    return nothing
end

function audit_blocks_v2!(parts, un, vn, dn, lower, gradient, diagcap, n, isbe)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    shared = CUDA.@cuStaticSharedMem(Float64, (4, 256))
    @inbounds if i <= n
        valid =
            isfinite(dn[i]) &&
            dn[i] >= 0 &&
            isfinite(un[1, i]) &&
            isfinite(un[2, i]) &&
            isfinite(vn[1, i]) &&
            isfinite(vn[2, i])
        kkt = 0.0
        if isbe
            valid = valid && isfinite(gradient[i]) && isfinite(diagcap[i]) && diagcap[i] > 0
            kkt = valid ? abs(clamp(dn[i] - gradient[i] / diagcap[i], lower[i], 1.0) - dn[i]) : Inf
        end
        shared[1, lane] = dn[i] - lower[i]
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
    if lane == 1
        @inbounds for row in 1:4
            parts[row, blockIdx().x] = shared[row, 1]
        end
    end
    return nothing
end

function finish_audit_kernel!(out, parts)
    if threadIdx().x == 1 && blockIdx().x == 1
        lo = Inf
        hi = -Inf
        kkt = bad = 0.0
        @inbounds for b in 1:size(parts, 2)
            lo = min(lo, parts[1, b])
            hi = max(hi, parts[2, b])
            kkt = max(kkt, parts[3, b])
            bad = max(bad, parts[4, b])
        end
        @inbounds out[1] = lo
        out[2] = hi
        out[3] = kkt
        out[4] = bad
    end
    return nothing
end

function audit!(sim, destination, velocity; isbe = true)
    b = sim.scratch
    s = sim.state
    g = sim.w.g
    @cuda threads = 256 blocks = size(b.audit_parts, 2) audit_blocks_v2!(
        b.audit_parts, b.un, velocity, b.dn, s.d, b.gradient, b.diagcap, g.n, isbe
    )
    @cuda threads = 1 blocks = 1 finish_audit_kernel!(destination, b.audit_parts)
    nothing
end

@inline correction_needed(before, tol) = isfinite(before[3]) && before[3] > tol
function maybe_correct_kernel!(dn, gradient, diagcap, lower, upper, before, tol, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n && correction_needed(before, tol)
        @inbounds dn[i] = clamp(dn[i] - gradient[i] / diagcap[i], lower[i], upper[i])
    end
    return nothing
end
function maybe_action_kernel!(z, dn, q, g, before, tol)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= g.n && correction_needed(before, tol)
        phase_node!(i, z, dn, q, g)
    end
    return nothing
end
function maybe_gradient_kernel!(delta, gradient, dn, lower, z, b, capacity, before, tol, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n && correction_needed(before, tol)
        @inbounds begin
            delta[i] = dn[i] - lower[i]
            gradient[i] = capacity[i] * delta[i] + z[i] - b[i]
        end
    end
    return nothing
end

function finish_step_kernel!(
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
    loading_rate,
)
    if threadIdx().x == 1 && blockIdx().x == 1
        kin = 0.5sum_parts(mech, 4)
        elastic = sum_parts(energy, 4)
        fracture = sum_parts(energy, 5)
        newE = elastic + fracture
        accepted_reaction = sum_parts(energy, 7) / loading_rate
        reaction = oldforce ? sum_parts(pred, 3) / loading_rate : accepted_reaction
        predicted = sum_parts(pred, 1) + sum_parts(pred, 2)
        curv = isbe ? 0.5sum_parts(energy, 1) : 0.0
        normal = isbe ? -sum_parts(energy, 2) : 0.0
        diss = sum_parts(energy, 3)
        supply = isbe ? sum_parts(energy, 6) : 0.0
        @inbounds begin
            workstep = 0.5 * (diag[9, 1] + reaction) * (loading_rate * h)
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
            correction = isbe && correction_needed(before, tol) ? before[3] : 0.0
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
            scale = max(abs(diag[13, 1]), 1.0)
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
            for row in 1:NDIAG
                diag[row, column] = diag[row, 1]
            end
        end
    end
    return nothing
end

# One D2H transaction reads current scalars and the ring. Host-only reporting
# must never change a numerical/device branch. Force a read at every saved state.
function read_diagnostics!(sim; check = true)
    host = Array(sim.diag)
    sim.host_reads += 1
    sim.diagnostic_reads += 1
    current = NamedTuple{DIAG_NAMES}(Tuple(host[:, 1]))
    firststep = max(sim.last_read_step + 1, sim.step - sim.options.read_every + 1)
    steps = collect(firststep:sim.step)
    columns = [mod(k - 1, sim.options.read_every) + 2 for k in steps]
    history = host[:, columns]
    dropped = max(0, sim.step - sim.last_read_step - sim.options.read_every)
    sim.last_read_step = sim.step
    check &&
        current.failure != 0 &&
        error(
            "P1V2 sticky failure $(current.failure) at step $(sim.step); use read_diagnostics!(sim;check=false)",
        )
    (;
        current,
        history,
        steps,
        dropped,
        host_reads = sim.host_reads,
        bound_reads = sim.bound_reads,
        diagnostic_reads = sim.diagnostic_reads,
    )
end

function advance_be!(sim, h, tnew)
    w = sim.w
    g = w.g
    s = sim.state
    b = sim.scratch
    o = sim.options
    # Retain the legacy expression order for the L0 bridge.
    copyto!(b.un, (s.u + h * s.v + 0.5h^2 * s.a) .* g.free + tnew * g.rate)
    main_operators!(w, b.un, s.d)
    copyto!(b.qp, w.qp)
    oldforce = o.layer != :L2
    if oldforce
        copyto!(b.an, (-w.f) ./ reshape(g.mass, 1, :) .* g.free)
        copyto!(b.vn, (s.v + 0.5h * (s.a + b.an)) .* g.free + g.rate)
    end
    velocity = oldforce ? b.vn : s.v
    @cuda threads = 256 blocks = size(b.predicted_parts, 2) mechanics_reduction_kernel!(
        b.predicted_parts, w.elastic, w.fracture, w.f, g.rate, g.mass, velocity, g.n, g.ne
    )
    copyto!(b.q, o.layer == :L0 ? s.cached : b.qp)
    coefficients!(b.b, b.diagonal, b.q, g)
    copyto!(b.capacity, g.C / h)
    b.diagcap .= b.diagonal .+ b.capacity
    copyto!(b.rhs, b.b + b.capacity .* s.d)
    o.representation == :SP && assemble!(b.sp, b.q, g)
    OPT.launch_box!(
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
    # Owned candidate damage: solver workspace can never become next step's lower.
    copyto!(b.dn, b.solver.x)
    action!(b.z, b.dn, b.q, g)
    @cuda threads = 256 blocks = cld(g.n, 256) phase_gradient_kernel!(
        b.delta, b.gradient, b.dn, s.d, b.z, b.b, b.capacity, g.n
    )
    audit!(sim, b.audit_before, s.v)
    @cuda threads = 256 blocks = cld(g.n, 256) maybe_correct_kernel!(
        b.dn, b.gradient, b.diagcap, s.d, b.upper, b.audit_before, o.tolerance, g.n
    )
    @cuda threads = 256 blocks = cld(g.n, 256) maybe_action_kernel!(
        b.z, b.dn, b.q, g, b.audit_before, o.tolerance
    )
    @cuda threads = 256 blocks = cld(g.n, 256) maybe_gradient_kernel!(
        b.delta, b.gradient, b.dn, s.d, b.z, b.b, b.capacity, b.audit_before, o.tolerance, g.n
    )
    action!(b.z, b.delta, b.q, g)
    main_operators!(w, b.un, b.dn)
    if !oldforce
        copyto!(b.an, (-w.f) ./ reshape(g.mass, 1, :) .* g.free)
        copyto!(b.vn, (s.v + 0.5h * (s.a + b.an)) .* g.free + g.rate)
    end
    audit!(sim, b.audit_after, b.vn)
    @cuda threads = 256 blocks = cld(g.ne, 256) supply_kernel!(b.supply, b.qp, b.q, s.d, b.dn, g)
    finish_step!(sim, h, tnew; isbe = true, oldforce, phasebound = 0.0)
end

function finish_step!(sim, h, tnew; isbe, oldforce = false, phasebound = 0.0)
    w = sim.w
    g = w.g
    s = sim.state
    b = sim.scratch
    @cuda threads = 256 blocks = size(b.mechanics_parts, 2) mechanics_reduction_kernel!(
        b.mechanics_parts, w.elastic, w.fracture, w.f, g.rate, g.mass, b.vn, g.n, g.ne
    )
    @cuda threads = 256 blocks = size(b.energy_parts, 2) energy_reduction_kernel!(
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
    @cuda threads = 1 blocks = 1 finish_step_kernel!(
        sim.diag,
        b.predicted_parts,
        b.mechanics_parts,
        b.energy_parts,
        b.audit_before,
        b.audit_after,
        b.solver.ctrl,
        h,
        tnew,
        sim.step + 1,
        isbe,
        oldforce,
        sim.options.tolerance,
        phasebound,
        sim.options.loading_rate,
    )
    s.H .= max.(s.H, b.qp)
    copyto!(s.cached, b.qp)
    sim.state = (; u = b.un, v = b.vn, a = b.an, d = b.dn, H = s.H, cached = s.cached)
    sim.scratch = merge(b, (; un = s.u, vn = s.v, an = s.a, dn = s.d))
    sim.t = tnew
    sim.step += 1
    nothing
end

# dtmax is the caller's mechanical/final-remainder cap. Heun may reduce it.
function advance!(sim, dtmax; force_read = false, target_time = nothing)
    @assert isfinite(dtmax) && 0 < dtmax <= sim.w.g.dtmechanical * (1 + 64eps())
    h = Float64(dtmax)
    if sim.options.method == :BE
        tnew = isnothing(target_time) ? sim.t + h : Float64(target_time)
        @assert isfinite(tnew) &&
            tnew > sim.t &&
            isapprox(tnew - sim.t, h; rtol = 0.0, atol = 64eps(max(abs(tnew), abs(sim.t), h)))
        advance_be!(sim, h, tnew)
    else
        h = advance_heun!(sim, h; target_time)
    end
    should_read =
        force_read ||
        sim.options.mode == :diagnostic ||
        sim.step - sim.last_read_step >= sim.options.read_every
    diagnostics = should_read ? read_diagnostics!(sim) : nothing
    (; dt = h, time = sim.t, step = sim.step, diagnostics)
end

end # module P1V2

# Heun extension shares the accepted-state/diagnostic implementation above.
@eval P1V2 begin
    function free_bound_kernel!(out, d, phase, rowptr, cols, values, capacity, n)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if i <= n
            bound = 0.0
            @inbounds if d[i] < 1 - 1e-12 && phase[i] >= 0
                for k in rowptr[i]:(rowptr[i + 1] - 1)
                    j = cols[k]
                    if d[j] < 1 - 1e-12 && phase[j] >= 0
                        bound += abs(values[k])
                    end
                end
                bound /= capacity[i]
            end
            @inbounds out[i] = bound
        end
        return nothing
    end

    function heun_step_size!(sim, dtmax)
        b = sim.scratch
        s = sim.state
        w = sim.w
        g = w.g
        rule = sim.options.rule
        if rule == Symbol("H-UNLIM")
            return dtmax, 0.0
        end
        bound = if rule == Symbol("H-LIM")
            @cuda threads = 256 blocks = cld(g.ne, 256) phase_bound_kernel!(w.bounds, s.cached, g)
            maximum(w.bounds)
        else
            assemble!(b.sp, s.cached, g)
            csr = b.sp.global_csr
            @cuda threads = 256 blocks = cld(g.n, 256) free_bound_kernel!(
                b.node_bounds, s.d, w.phase, csr.rowptr, csr.cols, csr.values, g.C, g.n
            )
            maximum(b.node_bounds)
        end
        sim.host_reads += 1
        sim.bound_reads += 1
        @assert isfinite(bound) && bound >= 0 "Nonfinite explicit phase bound"
        min(dtmax, bound > 0 ? 2 / bound : Inf), Float64(bound)
    end

    function advance_heun!(sim, dtmax; target_time = nothing)
        b = sim.scratch
        s = sim.state
        w = sim.w
        g = w.g
        h, bound = heun_step_size!(sim, dtmax)
        tnew = isnothing(target_time) ? sim.t + h : Float64(target_time)
        @assert isapprox(tnew - sim.t, h; rtol = 0.0, atol = 64eps(max(abs(tnew), abs(sim.t), h)))
        copyto!(b.q, s.cached)
        # First rate uses accepted (u,d); second uses (un,predicted d).
        b.rate .= max.(w.phase, 0.0) ./ g.C
        copyto!(b.pred, min.(1.0, s.d + h * b.rate))
        copyto!(b.un, (s.u + h * s.v + 0.5h * h * s.a) .* g.free + tnew * g.rate)
        main_operators!(w, b.un, b.pred)
        b.rs .= max.(w.phase, 0.0) ./ g.C
        copyto!(b.dn, min.(1.0, s.d + 0.5h * (b.rate + b.rs)))
        main_operators!(w, b.un, b.dn)
        copyto!(b.qp, w.qp)
        copyto!(b.an, (-w.f) ./ reshape(g.mass, 1, :) .* g.free)
        copyto!(b.vn, (s.v + 0.5h * (s.a + b.an)) .* g.free + g.rate)
        b.delta .= b.dn .- s.d
        b.capacity .= g.C ./ h
        fill!(b.z, 0.0)
        fill!(b.gradient, 0.0)
        fill!(b.supply, 0.0)
        audit!(sim, b.audit_after, b.vn; isbe = false)
        # Heun D_eta is the existing secant C*delta^2/h diagnostic, not exact RK dissipation.
        finish_step!(sim, h, tnew; isbe = false, phasebound = bound)
        h
    end
end
