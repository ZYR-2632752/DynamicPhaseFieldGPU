const HISTORY_COLUMNS = (
    :time_s,
    :step,
    :reaction,
    :kinetic,
    :elastic,
    :fracture,
    :work,
    :viscous,
    :physical_balance,
    :accounting_error,
    :phase_identity,
    :damage_max,
    :min_delta_step,
    :KKT,
    :CG,
    :AS,
)

function command_options(args; steps = 100, out = "")
    steps isa Integer && steps > 0 || error("Default step count must be positive")
    mesh = ""
    full = false
    i = 1
    while i <= length(args)
        flag = args[i]
        if flag == "--help"
            println(
                "Options: --steps N (default $steps), --full, --output NEW_DIR, --mesh INPUT_DIR"
            )
            return nothing
        elseif flag == "--full"
            full = true
        elseif flag in ("--steps", "--output", "--mesh")
            i < length(args) || error("Missing value after $flag")
            i += 1
            if flag == "--steps"
                steps = parse(Int, args[i])
                steps > 0 || error("--steps must be positive")
            elseif flag == "--output"
                out = args[i]
                isempty(out) && error("Empty output directory")
            else
                mesh = args[i]
                isdir(mesh) || error("Mesh directory does not exist: $mesh")
            end
        else
            error("Unknown argument: $flag")
        end
        i += 1
    end
    full && "--steps" in args && error("Choose --full or --steps, not both")
    (; steps = full ? typemax(Int) : steps, out, mesh)
end

function new_output(path)
    ispath(path) && error("Output already exists: $path. Choose a new directory.")
    mkpath(dirname(abspath(path)))
    mkdir(path)
    abspath(path)
end

function save_metadata(path, data)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        TOML.print(io, data; sorted = true)
    end
    mv(temporary, path; force = true)
end

function write_history(io, row; reaction_available = true)
    values = map(HISTORY_COLUMNS) do key
        key == :reaction && !reaction_available ? NaN : getproperty(row, key)
    end
    println(io, join(values, ','))
end

function history_row(s, info)
    (;
        time_s = s.t,
        step = s.step,
        info.reaction,
        kinetic = s.kin,
        info.elastic,
        info.fracture,
        work = s.work,
        viscous = s.visc,
        physical_balance = s.physical,
        accounting_error = s.balance,
        phase_identity = info.identity,
        damage_max = info.max_damage,
        min_delta_step = info.min_increment,
        KKT = info.kkt,
        CG = info.cg,
        AS = info.outer,
    )
end

function write_vtk(path, xy, cells, celltype, u, d)
    size(xy, 2) == length(d) == size(u, 2) || error("VTK node count mismatch")
    size(xy, 1) in (2, 3) && size(u, 1) == size(xy, 1) || error("VTK field dimension mismatch")
    n = length(d)
    ne = size(cells, 2)
    k = size(cells, 1)
    minimum(cells) >= 1 && maximum(cells) <= n || error("VTK connectivity out of range")
    ispath(path) && error("VTK output already exists: $path")
    padded(a) = size(a, 1) == 3 ? a : vcat(a, zeros(1, n))
    binary64(io, a) = write(io, hton.(reinterpret(UInt64, vec(Float64.(a)))))
    connectivity = vcat(fill(Int32(k), 1, ne), Int32.(cells .- 1))
    open(path * ".tmp", "w") do io
        println(
            io,
            "# vtk DataFile Version 3.0\nDynamic phase-field fracture\nBINARY\nDATASET UNSTRUCTURED_GRID",
        )
        println(io, "POINTS $n double")
        binary64(io, padded(xy))
        println(io)
        println(io, "CELLS $ne $(length(connectivity))")
        write(io, hton.(vec(connectivity)))
        println(io)
        println(io, "CELL_TYPES $ne")
        write(io, hton.(fill(Int32(celltype), ne)))
        println(io)
        println(io, "POINT_DATA $n\nSCALARS damage double 1\nLOOKUP_TABLE default")
        binary64(io, d)
        println(io)
        println(io, "VECTORS displacement double")
        binary64(io, padded(u))
        println(io)
    end
    mv(path * ".tmp", path)
end

function check_file_hashes(folder, hashes)
    for (name, digest) in hashes
        basename(name) == name && !isabspath(name) || error("Invalid mesh filename")
        open(joinpath(folder, name)) do io
            bytes2hex(sha256(io)) == digest || error("Mesh hash mismatch: $name")
        end
    end
end
