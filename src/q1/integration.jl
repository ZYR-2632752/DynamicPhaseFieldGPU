
function initialize(host)
    g = make_gpu_elliptic_grid(host)
    n = g.n
    ne = g.nx * g.ny
    u = CUDA.zeros(Float64, 2, n)
    d = CUDA.zeros(Float64, n)
    w = (;
        g,
        f = zero(u),
        fd = zero(d),
        qp = CUDA.zeros(Float64, 4, ne),
        qm = CUDA.zeros(Float64, 4, ne),
        K = CUDA.zeros(Float64, 4, 4, ne),
        b = zero(d),
        diagonal = zero(d),
        capacity = zero(d),
        diagQ = zero(d),
        rhs = zero(d),
        action = zero(d),
    )
    descriptor = (; kind = Val(:q1), values = w.K)
    solver = BoxKWOptimized.workspace(d; maxblocks = 128)
    kernel = @cuda launch = false BoxKWOptimized.kernel!(
        solver, descriptor, w.capacity, w.diagQ, w.rhs, d, g.upper, g, 1e-12, 300, true, true
    )
    limit =
        CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) *
        CUDA.active_blocks(kernel.fun, BoxKWOptimized.NT)
    limit > 0 || error("No cooperative kernel residency")
    s = (;
        u,
        v = zero(u),
        a = g.loads ./ reshape(g.mass, 1, :) .* g.free,
        d,
        step = 0,
        t = 0.0,
        kin = 0.0,
        E = 0.0,
        initialE = 0.0,
        work = 0.0,
        visc = 0.0,
        curv = 0.0,
        normal = 0.0,
        mech = 0.0,
        physical = 0.0,
        balance = 0.0,
    )
    (; w, descriptor, solver, blocks = min(128, limit)), s
end

function advance(ctx, s, dt)
    w = ctx.w
    g = w.g
    un = (s.u .+ dt .* s.v .+ (0.5dt^2) .* s.a) .* g.free
    predicted = mersel_q1_operators!(w.f, w.fd, w.qp, w.qm, un, s.d, g)
    elliptic_coefficients!(w.K, w.b, w.diagonal, w.qp, g)
    w.capacity .= g.C ./ dt
    w.diagQ .= w.diagonal .+ w.capacity
    w.rhs .= w.b .+ w.capacity .* s.d
    BoxKWOptimized.launch_box!(
        ctx.solver,
        ctx.descriptor,
        w.capacity,
        w.diagQ,
        w.rhs,
        s.d,
        g.upper,
        g;
        tolerance = 1e-12,
        blocks = ctx.blocks,
        skip_fixed = true,
        fused = true,
    )
    result = BoxKWOptimized.read_box_result(ctx.solver)
    dn = result.x
    delta = dn .- s.d
    elliptic_action!(w.action, dn, w.K, g)
    gradient = w.action .+ w.capacity .* delta .- w.b
    kkt = maximum(abs.(clamp.(dn .- gradient ./ w.diagQ, s.d, g.upper) .- dn))
    mininc = minimum(delta)
    dmax = maximum(dn)
    kkt <= 1e-12 && mininc >= 0 && dmax <= 1 && minimum(dn) >= 0 ||
        error("Q1 box/KKT acceptance failed")
    normalstep = -dot(gradient, delta)
    elliptic_action!(w.action, delta, w.K, g)
    curvstep = 0.5dot(delta, w.action)
    viscstep = sum(w.capacity .* delta .^ 2)
    energy = mersel_q1_operators!(w.f, w.fd, w.qp, w.qm, un, dn, g)
    an = (g.loads .- w.f) ./ reshape(g.mass, 1, :) .* g.free
    vn = (s.v .+ (0.5dt) .* (s.a .+ an)) .* g.free
    kin = 0.5sum(reshape(g.mass, 1, :) .* vn .^ 2)
    E = energy.elastic + energy.fracture
    workstep = sum(g.loads .* (un .- s.u))
    predictedE = predicted.elastic + predicted.fracture
    identity = E - predictedE + viscstep + curvstep + normalstep
    work = s.work + workstep
    visc = s.visc + viscstep
    curv = s.curv + curvstep
    normal = s.normal + normalstep
    mech = s.mech + kin - s.kin + predictedE - s.E - workstep
    physical = kin + E - s.initialE + visc - work
    balance = physical + curv + normal - mech
    scale = max(abs(work), abs(E), abs(kin), 1.0)
    all(x -> all(isfinite, x), (un, vn, an, dn)) || error("Nonfinite Q1 state")
    abs(identity) <= 1e-8scale && abs(balance) <= 1e-8scale || error("Q1 energy identity failed")
    next = (;
        u = un,
        v = vn,
        a = an,
        d = dn,
        step = s.step + 1,
        t = (s.step + 1) * dt,
        kin,
        E,
        initialE = s.initialE,
        work,
        visc,
        curv,
        normal,
        mech,
        physical,
        balance,
    )
    detail = (;
        energy.elastic,
        energy.fracture,
        kkt,
        min_increment = mininc,
        max_damage = dmax,
        reaction = -sum(@view w.f[1, 1:(g.nx + 1):g.n]),
        identity,
        outer = result.outer,
        cg = result.cg_total,
    )
    next, detail
end
