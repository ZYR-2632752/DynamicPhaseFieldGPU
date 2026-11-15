const REPOSITORY_ROOT = dirname(@__DIR__)

function prepare_case(case, p)
    if case == :bar
        host = BarQ1.make_grid(p)
        total = ceil(Int, p.end_time_s / host.dt)
        dt = p.end_time_s / total
        ctx, state = BarQ1.initialize(host.grid)
        return (;
            kind = :single,
            ctx,
            state,
            advance = BarQ1.advance,
            dt,
            total,
            xy = host.xy,
            cells = host.grid.cells,
            celltype = 9,
            length_unit = "m",
            energy_unit = "J",
            qualification = "Recovered Q1 operators with shared persistent box solver",
        )
    elseif case == :impact3d
        host = Impact3D.load_compact_mesh(p.mesh, p)
        pattern = Impact3D.csr_pattern(host)
        g = Impact3D.device_grid(host)
        w = Impact3D.H8Operators.workspace(g)
        bound = Impact3D.H8Operators.mechanical_bound(w).dt
        total = ceil(Int, p.end_time_s / bound)
        dt = p.end_time_s / total
        phase = Impact3D.phase_workspace(
            g, (; rowptr = CuArray(pattern.rowptr), cols = CuArray(pattern.cols))
        )
        Impact3D.assemble_phase!(phase, g, fill!(similar(w.q), 0.0), dt, g.initial_damage)
        solver = Impact3D.resident_solver(phase, g, g.initial_damage)
        state = Impact3D.initial_state(
            w, g; speed_mm_s = p.impact_speed_mm_s, ramp_s = p.ramp_time_s
        )
        advance =
            (ctx, s, h) -> begin
                next, info = Impact3D.advance(
                    w,
                    phase,
                    solver.solvebox,
                    s,
                    g,
                    h;
                    speed_mm_s = p.impact_speed_mm_s,
                    ramp_s = p.ramp_time_s,
                )
                Impact3D.check_monitor(solver)
                next, info
            end
        return (;
            kind = :single,
            ctx = nothing,
            state,
            advance,
            dt,
            total,
            xy = host.coords,
            cells = reshape(host.conn, 8, host.ne),
            celltype = 12,
            length_unit = "mm",
            energy_unit = "N mm",
            qualification = host.mesh_qualification,
        )
    elseif case == :sent
        base = P1V2.make_grid(p)
        total = ceil(Int, p.end_time_s / base.dtmechanical)
        dt = p.end_time_s / total
        sim = P1V2.initialize(base; method = :BE, layer = :L2)
        ctx = DeviceL1.context(sim, dt; stop = p.end_time_s)
        return (;
            kind = :batched,
            ctx,
            advance = DeviceL1.advance!,
            dt,
            total,
            xy = base.xy,
            cells = base.tri,
            celltype = 5,
            length_unit = "m",
            energy_unit = "J",
            reaction_available = true,
            qualification = "Recovered spectral P1 SENT configuration",
        )
    elseif case == :kw
        spec = TOML.parsefile(joinpath(p.mesh, "manifest.toml"))
        check_file_hashes(p.mesh, spec["mesh_sha256"])
        input = P1V2.load_research_problem(
            p.mesh;
            expected_size = (spec["nodes"], spec["elements"]),
            expected_boundaries = (spec["impact_nodes"], spec["bottom_nodes"]),
        )
        base = P1V2.kw_grid(input, p)
        eta = p.eta_Pa_s * 1e-6
        kappa = p.kappa
        cR = p.c_R_mm_s
        base = merge(
            base,
            (;
                eta,
                C = eta * base.volume,
                ell = p.ell_mm,
                kappa,
                c_R = cR,
                Pi_eta = eta * cR / base.gc,
                velocity_scale = 1.0,
                impact_speed = p.impact_speed_mm_s,
                ramp_time = p.ramp_time_s,
                dtmechanical = base.dtmechanical * sqrt((1 + base.kappa) / (1 + kappa)),
            ),
        )
        total = ceil(Int, p.end_time_s / base.dtmechanical)
        dt = p.end_time_s / total
        seed = (;
            u = zeros(2, base.n),
            v = zeros(2, base.n),
            a = (p.impact_speed_mm_s / p.ramp_time_s) .* base.rate,
            d = zeros(base.n),
            H = zeros(base.ne),
            cached = zeros(base.ne),
            initialK = 0.0,
            initialE = 0.0,
        )
        sim = P1V2.initialize(base; method = :BE, layer = :L2, seed)
        b = sim.scratch
        w = sim.w
        g = w.g
        s = sim.state
        @cuda threads = 256 blocks = size(b.mechanics_parts, 2) DeviceKW.mechanics_kw_kernel!(
            b.mechanics_parts, w.elastic, w.fracture, w.f, g.rate, g.mass, s.v, s.a, g.n, g.ne
        )
        initial_reaction = sum(reshape(base.mass, 1, :) .* seed.a .* base.rate)
        @cuda threads = 1 blocks = 1 DeviceKW.initialize_ledger!(
            sim.diag, b.mechanics_parts, initial_reaction
        )
        ctx = DeviceKW.context(sim, dt; stop = p.end_time_s)
        return (;
            kind = :batched,
            ctx,
            advance = DeviceKW.advance!,
            dt,
            total,
            xy = base.xy,
            cells = base.tri,
            celltype = 5,
            length_unit = "mm",
            energy_unit = "N mm",
            reaction_available = true,
            qualification = "Archived spectral P1 KW; mesh and split differ from final manuscript",
        )
    elseif case == :branching
        base, loads = BranchingMesh.load_grid(joinpath(p.mesh, "mesh"), p)
        meshhash = BranchingMesh.file_sha(joinpath(p.mesh, "mesh/manifest.toml"))
        seed, _ = BranchingMesh.read_seed(
            base,
            loads,
            joinpath(p.mesh, "seed");
            load_scale = p.load_scale,
            mesh_manifest_sha256 = meshhash,
        )
        base = merge(
            base,
            (; loads = p.load_scale .* loads, upper = ones(base.n), reaction_mask = zeros(base.n)),
        )
        total = ceil(Int, p.end_time_s / base.dtmechanical)
        dt = p.end_time_s / total
        ctx = TractionGPU.initialize(base, dt; method = :BE, stop = p.end_time_s, seed)
        return (;
            kind = :batched,
            ctx,
            advance = DeviceL1.advance!,
            dt,
            total,
            xy = base.xy,
            cells = base.tri,
            celltype = 5,
            length_unit = "mm",
            energy_unit = "N mm",
            reaction_available = false,
            qualification = "Mirrored P1 mesh and seed adapted to shared GPU traction update; GPU trajectory unverified",
        )
    end
    error("Unknown example: $case")
end

function run_example(case, parameters, args = String[])
    options = command_options(args; steps = parameters.steps, out = parameters.output_dir)
    isnothing(options) && return nothing
    case in (:bar, :sent, :kw, :branching, :impact3d) || error("Unknown example: $case")
    p = parameters
    if !isempty(options.mesh)
        hasproperty(p, :mesh) || error("This example generates its mesh; --mesh is not used")
        p = merge(p, (; mesh = abspath(options.mesh)))
    end
    for (key, value) in pairs(p)
        value isa Real && !isfinite(value) && error("Nonfinite parameter: $key")
    end
    p.end_time_s > 0 || error("End time must be positive")
    hasproperty(p, :eta_Pa_s) && p.eta_Pa_s < 0 && error("Viscosity must be nonnegative")
    hasproperty(p, :ell_mm) && p.ell_mm <= 0 && error("Length scale must be positive")
    hasproperty(p, :load_scale) && p.load_scale <= 0 && error("Load scale must be positive")
    hasproperty(p, :impact_speed_mm_s) &&
        p.impact_speed_mm_s <= 0 &&
        error("Impact speed must be positive")
    hasproperty(p, :ramp_time_s) && p.ramp_time_s <= 0 && error("Ramp time must be positive")
    CUDA.functional() || error("A working NVIDIA CUDA device is required; there is no CPU fallback")
    CUDA.allowscalar(false)
    path =
        isempty(options.out) ?
        joinpath(
            REPOSITORY_ROOT,
            "outputs",
            string(case) * "_" * Dates.format(now(), "yyyymmdd_HHMMSSsss"),
        ) : options.out
    out = new_output(path)
    metadata = Dict{String,Any}(
        "case" => string(case),
        "status" => "initializing",
        "parameters" => Dict(string(k) => v for (k, v) in pairs(p)),
        "julia" => string(VERSION),
        "cuda_jl" => string(Base.pkgversion(CUDA)),
        "gpu" => CUDA.name(CUDA.device()),
        "cuda_driver" => string(CUDA.driver_version()),
        "cuda_runtime" => string(CUDA.runtime_version()),
        "created_utc" => string(now(UTC)),
        "requested_steps" => options.steps,
        "timing_scope" => "Time-stepping loop, including first-step compilation, synchronization and history CSV writes; initialization and final VTK export excluded",
    )
    metapath = joinpath(out, "run.toml")
    save_metadata(metapath, metadata)
    try
        model = prepare_case(case, p)
        steps = min(options.steps, model.total)
        metadata["nodes"] = size(model.xy, 2)
        metadata["elements"] = size(model.cells, 2)
        metadata["dt_s"] = model.dt
        metadata["steps_to_end_time"] = model.total
        metadata["length_unit"] = model.length_unit
        metadata["energy_unit"] = model.energy_unit
        metadata["force_unit"] = "N"
        metadata["configuration_note"] = model.qualification
        metadata["reaction_available"] = case != :branching
        metadata["status"] = "running"
        save_metadata(metapath, metadata)
        println("$case: $(metadata["nodes"]) nodes, $steps / $(model.total) steps; output: $out")
        CUDA.synchronize()
        started = time_ns()
        state = if model.kind == :single
            state = model.state
            open(joinpath(out, "history.csv"), "w") do io
                println(io, join(HISTORY_COLUMNS, ','))
                for step in 1:steps
                    state, info = model.advance(model.ctx, state, model.dt)
                    write_history(io, history_row(state, info))
                    step % 100 == 0 && flush(io)
                end
            end
            state
        else
            sim = model.ctx.sim
            open(joinpath(out, "history.csv"), "w") do io
                println(io, join(HISTORY_COLUMNS, ','))
                while sim.step < steps
                    count = min(sim.options.read_every, steps - sim.step)
                    report = model.advance(model.ctx, count)
                    report.dropped == 0 || error("History ring overflow")
                    for col in axes(report.history, 2)
                        row = NamedTuple{P1V2.DIAG_NAMES}(Tuple(report.history[:, col]))
                        write_history(io, row; reaction_available = model.reaction_available)
                    end
                    flush(io)
                end
            end
            merge(sim.state, (; step = sim.step, t = sim.t))
        end
        CUDA.synchronize()
        elapsed = (time_ns() - started) / 1e9
        metadata["loop_wall_seconds"] = elapsed
        metadata["executed_steps"] = state.step
        metadata["final_time_s"] = state.t
        metadata["termination"] = steps == model.total ? "end_time" : "step_limit"
        write_vtk(
            joinpath(out, "final.vtk"),
            model.xy,
            model.cells,
            model.celltype,
            Array(state.u),
            Array(state.d),
        )
        metadata["status"] = "completed"
        save_metadata(metapath, metadata)
        println(
            "Completed $(state.step) steps at t=$(state.t) s; loop wall time $(round(elapsed;digits=3)) s.",
        )
        out
    catch err
        metadata["status"] = "failed"
        metadata["error"] = sprint(showerror, err)
        save_metadata(metapath, metadata)
        rethrow()
    end
end
