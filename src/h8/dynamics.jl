include("boundary.jl")
function initial_state(w, g; speed_mm_s, ramp_s)
    u = fill!(similar(g.free), 0.0)
    v = copy(u)
    a = copy(u)
    d = copy(g.initial_damage)
    impose_impact!(u, v, a, g.free, g.velocity_unit, 0.0; speed_mm_s, ramp_s)
    H8Operators.mechanics!(w, u, d)
    E = sum(w.elastic) + sum(w.fracture)
    (;
        u,
        v,
        a,
        d,
        f = copy(w.f),
        step = 0,
        t = 0.0,
        E,
        initialE = E,
        kin = 0.0,
        work = 0.0,
        visc = 0.0,
        curv = 0.0,
        normal = 0.0,
        mech = 0.0,
        physical = 0.0,
        balance = 0.0,
    )
end

function advance(w, p, solvebox, s, g, dt; speed_mm_s, ramp_s)
    tn = (s.step + 1) * dt
    bc = impact_kinematics(tn; speed_mm_s, ramp_s)
    un = g.free .* (s.u .+ dt .* s.v .+ (0.5dt^2) .* s.a) .+ g.velocity_unit .* bc.u
    H8Operators.mechanics!(w, un, s.d)
    predicted = sum(w.elastic) + sum(w.fracture)
    backend_sync()
    started = time_ns()
    # w.q is instantaneous psi+(u_new), overwritten every mechanics evaluation.
    assemble_phase!(p, g, w.q, dt, s.d)
    backend_sync()
    assembly_s = (time_ns() - started) / 1e9
    started = time_ns()
    result = solvebox(p, s.d)
    dn = result.x
    backend_sync()
    solve_s = (time_ns() - started) / 1e9
    Base.mightalias(dn, s.d) && error("Accepted damage aliases lower bound")
    info = phase_check!(p, g, dn, s.d)
    all(isfinite, (info.kkt, info.min_damage, info.max_damage, info.min_increment)) ||
        error("Nonfinite phase diagnostics")
    info.kkt <= 1e-12 && info.min_increment >= 0 && info.min_damage >= 0 && info.max_damage <= 1 ||
        error("Box/KKT acceptance failed: $info")
    delta = dn .- s.d
    normalstep = -dot(p.audit, delta)
    phase_action!(p.action, delta, p, g)
    viscstep = sum(p.capacity .* delta .^ 2)
    curvstep = 0.5 * (dot(delta, p.action) - viscstep)
    # L2: the NEW damage force supplies both acceleration and velocity.
    H8Operators.mechanics!(w, un, dn)
    an = g.free .* (-w.f ./ reshape(g.mass, 1, :)) .+ g.velocity_unit .* bc.a
    vn = g.free .* (s.v .+ (0.5dt) .* (s.a .+ an)) .+ g.velocity_unit .* bc.v
    kin = 0.5sum(reshape(g.mass, 1, :) .* vn .^ 2)
    workstep = dirichlet_work(s.f, w.f, s.u, un, s.v, vn, g.mass, g.free)
    elastic = sum(w.elastic)
    fracture = sum(w.fracture)
    E = elastic + fracture
    identity = E - predicted + viscstep + curvstep + normalstep
    work = s.work + workstep
    visc = s.visc + viscstep
    curv = s.curv + curvstep
    normal = s.normal + normalstep
    mech = s.mech + kin - s.kin + predicted - s.E - workstep
    physical = kin + E - s.initialE + visc - work
    balance = physical + curv + normal - mech
    scale = max(abs(work), abs(E), abs(kin), 1.0)
    all(x -> all(isfinite, x), (un, vn, an, dn)) || error("Nonfinite state")
    all(isfinite, (kin, E, work, visc, identity, balance)) || error("Nonfinite energy")
    abs(identity) <= 1e-8scale && abs(balance) <= 1e-8scale || error("Energy identity failure")
    next = (;
        u = un,
        v = vn,
        a = an,
        d = dn,
        f = copy(w.f),
        step = s.step + 1,
        t = tn,
        E,
        initialE = s.initialE,
        kin,
        work,
        visc,
        curv,
        normal,
        mech,
        physical,
        balance,
    )
    detail = merge(
        info,
        (;
            elastic,
            fracture,
            identity,
            assembly_s,
            solve_s,
            solver_kkt = result.residual,
            outer = result.outer,
            cg = result.cg_total,
            actions = result.actions,
            identity_scaled = abs(identity) / scale,
            visc_step = viscstep,
            curv_step = curvstep,
            normal_step = normalstep,
            work_step = workstep,
            predicted_energy = predicted,
            history_supply = 0.0,
            reaction = sum(impact_reaction(w.f, an, g.mass, g.free) .* g.velocity_unit),
        ),
    )
    next, detail
end
