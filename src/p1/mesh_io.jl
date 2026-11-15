function read_research_npy(path, expected_shape::Tuple, allowed_dtypes::Tuple)
    ENDIAN_BOM == 0x04030201 || error("This NPY reader requires a little-endian host")
    1 <= length(expected_shape) <= 3 && all(d -> d isa Integer && d > 0, expected_shape) ||
        error("Invalid expected shape")
    expected_shape = Tuple(Int(d) for d in expected_shape)
    count = foldl(Base.Checked.checked_mul, expected_shape; init = 1)
    open(path, "r") do io
        read(io, 6) == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] ||
            error("Invalid NPY magic: $path")
        major, minor = read(io, UInt8), read(io, UInt8)
        major in (1, 2) && minor == 0 || error("Unsupported NPY version: $path")
        header_bytes = major == 1 ? Int(ltoh(read(io, UInt16))) : Int(ltoh(read(io, UInt32)))
        0 < header_bytes <= 65536 || error("Invalid NPY header length: $path")
        raw_header = read(io, header_bytes)
        length(raw_header) == header_bytes || error("Truncated NPY header: $path")
        header = String(raw_header)
        dtype = match(r"'descr':\s*'([^']+)'", header)
        shape = match(r"'shape':\s*\(([^)]*)\)", header)
        order = match(r"'fortran_order':\s*(True|False)", header)
        any(isnothing, (dtype, shape, order)) && error("Invalid NPY header fields: $path")
        all(
            length(collect(eachmatch(Regex("'" * key * "'\\s*:"), header))) == 1 for
            key in ("descr", "shape", "fortran_order")
        ) || error("Duplicate NPY header field: $path")
        dtype[1] in allowed_dtypes || error("NPY dtype differs from this mesh field: $path")
        T = get(Dict("<f8" => Float64, "<i8" => Int64, "<i4" => Int32), dtype[1], nothing)
        isnothing(T) && error("Unsupported NPY dtype: $path")
        dims = Tuple(
            parse(Int, strip(token)) for token in split(shape[1], ',') if !isempty(strip(token))
        )
        dims == expected_shape || error("NPY shape differs from expected $expected_shape: $path")
        payload_bytes = Base.Checked.checked_mul(count, sizeof(T))
        Base.Checked.checked_add(position(io), payload_bytes) == filesize(path) ||
            error("NPY file/payload size mismatch: $path")
        bytes = read(io, payload_bytes)
        length(bytes) == payload_bytes && eof(io) ||
            error("NPY payload changed or truncated: $path")
        data = copy(reinterpret(T, bytes))
        order[1] == "True" && return reshape(data, expected_shape)
        permutedims(reshape(data, reverse(expected_shape)), reverse(1:length(expected_shape)))
    end
end

function load_research_problem(folder; expected_size, expected_boundaries = (6, 21))
    length(expected_size) == 2 && all(v -> v isa Integer && v > 0, expected_size) ||
        error("Invalid expected mesh size")
    n, ne = Int.(expected_size)
    n <= typemax(Int32) || error("Node indices cannot be represented by the existing Int32 kernels")
    floats(name, shape) = read_research_npy(joinpath(folder, name * ".npy"), shape, ("<f8",))
    integers(name, shape) =
        read_research_npy(joinpath(folder, name * ".npy"), shape, ("<i8", "<i4"))
    raw_tri = integers("triangles", (ne, 3))
    0 <= minimum(raw_tri) && maximum(raw_tri) < n || error("Triangle index out of bounds")
    tri = Int32.(raw_tri .+ 1)
    nodes = floats("nodes", (n, 2))
    G = floats("gradients", (ne, 3, 2))
    volume = vec(floats("areas", (ne,)))
    mass = vec(floats("mass", (n,)))
    raw_impact = vec(integers("impact", (expected_boundaries[1],)))
    raw_bottom = vec(integers("bottom", (expected_boundaries[2],)))
    all(i -> 0 <= i < n, raw_impact) && all(i -> 0 <= i < n, raw_bottom) ||
        error("Boundary index out of bounds")
    impact = Int32.(raw_impact .+ 1)
    bottom = Int32.(raw_bottom .+ 1)
    @assert length(mass) == n &&
        length(volume) == ne &&
        size(nodes) == (n, 2) &&
        size(G) == (ne, 3, 2) &&
        size(tri) == (ne, 3)
    @assert 1 <= minimum(tri) && maximum(tri) <= n && minimum(volume) > 0 && minimum(mass) > 0
    @assert length(impact) == expected_boundaries[1] &&
        length(bottom) == expected_boundaries[2] &&
        all(i -> 1 <= i <= n, impact) &&
        all(i -> 1 <= i <= n, bottom)
    @assert all(x -> all(isfinite, x), (nodes, G, volume, mass))
    fixed_x = falses(n)
    fixed_x[impact] .= true
    fixed_y = falses(n)
    fixed_y[bottom] .= true
    (
        tri = tri,
        nodes = nodes,
        G = G,
        volume = volume,
        mass = mass,
        impact = impact,
        fixed_x = collect(fixed_x),
        fixed_y = collect(fixed_y),
    )
end
