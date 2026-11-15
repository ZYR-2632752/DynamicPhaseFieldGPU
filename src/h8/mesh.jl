# Cache geometry per extrusion spacing; all scientific arrays remain FP64.
using TOML, SHA, Mmap

struct SectionShape{A} <: AbstractMatrix{Float64}
    values::A
    supports::Int
end
Base.size(x::SectionShape) = (8, x.supports)
@inline Base.getindex(x::SectionShape, q::Integer, a::Integer) =
    @inbounds x.values[q, mod(a - 1, 8) + 1]

struct SectionGradient{A,B} <: AbstractArray{Float64,3}
    values::A
    layer_groups::B
    section_supports::Int
    supports::Int
end
Base.size(x::SectionGradient) = (3, 8, x.supports)
@inline function Base.getindex(x::SectionGradient, c::Integer, q::Integer, a::Integer)
    layer = div(a - 1, x.section_supports) + 1
    @inbounds k =
        (x.layer_groups[layer] - 1) * x.section_supports + mod(a - 1, x.section_supports) + 1
    @inbounds x.values[c, q, k]
end

struct SectionWeight{A,B} <: AbstractMatrix{Float64}
    values::A
    layer_groups::B
    section_elements::Int
    elements::Int
end
Base.size(x::SectionWeight) = (8, x.elements)
@inline function Base.getindex(x::SectionWeight, q::Integer, e::Integer)
    layer = div(e - 1, x.section_elements) + 1
    @inbounds k =
        (x.layer_groups[layer] - 1) * x.section_elements + mod(e - 1, x.section_elements) + 1
    @inbounds x.values[q, k]
end

struct H8Pointers <: AbstractVector{Int32}
    elements::Int
end
Base.size(x::H8Pointers) = (x.elements + 1,)
@inline Base.getindex(x::H8Pointers, e::Integer) = Int32(8(e - 1) + 1)
struct H8SupportElement <: AbstractVector{Int32}
    supports::Int
end
Base.size(x::H8SupportElement) = (x.supports,)
@inline Base.getindex(x::H8SupportElement, a::Integer) = Int32(div(a - 1, 8) + 1)

function verified_array(folder, meta, name, ::Type{T}, shape) where {T}
    a = meta["arrays"][name]
    Int.(a["shape"]) == collect(shape) || error("Array shape mismatch: $name")
    a["dtype"] == (T == Int32 ? "int32" : "float64") || error("Array precision mismatch: $name")
    relative = replace(a["filename"], '\\' => '/')
    (isabspath(relative) || ".." in split(relative, '/')) && error("Unsafe mesh path")
    path = joinpath(folder, relative)
    filesize(path) == sizeof(T) * prod(shape) || error("Array length mismatch: $name")
    open(path) do io
        bytes2hex(sha256(io)) == a["sha256"] || error("Mesh hash mismatch: $name")
    end
    open(path) do io
        Mmap.mmap(io, Array{T,length(shape)}, shape)
    end
end

"""Host mesh loading and deterministic adjacency, without any dynamics or CUDA allocation."""
function load_compact_mesh(folder, p)
    file = joinpath(folder, "manifest.toml")
    meta = TOML.parsefile(file)
    meta["schema"] == "s6_extruded_h8_v1" || error("Unsupported mesh schema")
    meta["endianness"] == "little" && ENDIAN_BOM == 0x04030201 || error("Endianness")
    meta["order"] == "Fortran" && meta["connectivity_index_base"] == 1 || error("Mesh ordering")
    n = Int(meta["ncp"])
    ne = Int(meta["nelem"])
    Q = Int(meta["section_quads"])
    m = 8ne
    n > 0 && Q > 0 && ne == Q * meta["layers"] && m + 1 < typemax(Int32) || error("Mesh extent")
    conn = verified_array(folder, meta, "conn", Int32, (m,))
    coords = verified_array(folder, meta, "coords", Float64, (3, n))
    nn = verified_array(folder, meta, "N", Float64, (8, 8))
    ng = Int(get(meta, "geometry_groups", 1))
    groups =
        haskey(meta["arrays"], "layer_groups") ?
        verified_array(folder, meta, "layer_groups", Int32, (Int(meta["layers"]),)) :
        ones(Int32, Int(meta["layers"]))
    ng > 0 && minimum(groups) >= 1 && maximum(groups) <= ng || error("Geometry group index")
    gg = verified_array(folder, meta, "grad", Float64, (3, 8, 8Q * ng))
    ww = verified_array(folder, meta, "weights", Float64, (8, Q * ng))
    all(isfinite, coords) && all(isfinite, gg) && all(isfinite, ww) && minimum(ww) > 0 ||
        error("Invalid mesh geometry")
    minimum(conn) >= 1 && maximum(conn) <= n || error("Connectivity range")
    N = SectionShape(nn, m)
    G = SectionGradient(gg, groups, 8Q, m)
    weight = SectionWeight(ww, groups, Q, ne)
    localvol = zeros(8, Q * ng)
    for e in 1:(Q * ng), a in 1:8, q in 1:8
        localvol[a, e] += ww[q, e] * nn[q, a]
    end
    volume = zeros(n)
    counts = zeros(Int32, n)
    for a in 1:m
        i = conn[a]
        counts[i] += 1
        e = div(a - 1, 8)
        layer = div(e, Q) + 1
        cache = (groups[layer] - 1) * Q + mod(e, Q) + 1
        volume[i] += localvol[mod(a - 1, 8) + 1, cache]
    end
    all(>(0), volume) && isapprox(sum(volume), 1e6; rtol = 1e-10) ||
        error("Lumped volume qualification")
    offsets = Vector{Int32}(undef, n + 1)
    offsets[1] = 1
    for i in 1:n
        offsets[i + 1] = offsets[i] + counts[i]
    end
    adj = Vector{Int32}(undef, m)
    slots = copy(offsets[1:n])
    for a in 1:m
        i = conn[a]
        adj[slots[i]] = a
        slots[i] += 1
    end
    free = ones(3, n)
    velocity_unit = zeros(3, n)
    P = Int(meta["section_nodes"])
    base_nodes = (Int(meta["layers"]) + 1) * P
    for i in 1:n
        x = coords[1, i]
        y = coords[2, i]
        z = coords[3, i]
        r = hypot(y, z)
        front = abs(x) < 1e-10
        impact = front && (r < p.impact_radius_mm - 1e-8 || i > base_nodes)
        fixed = abs(y - p.outer_y_mm) < 1e-8 || abs(z - p.outer_z_mm) < 1e-8 || (front && !impact)
        if fixed
            free[1, i] = 0
            free[2, i] = 0
            free[3, i] = 0
        end
        abs(y) < 1e-8 && (free[2, i] = 0)
        abs(z) < 1e-8 && (free[3, i] = 0)
        if impact
            free[1, i] = 0
            velocity_unit[1, i] = 1
        end
    end
    rho = p.density_tonne_mm3
    eta = p.eta_Pa_s * 1e-6
    gc = p.gc_N_mm
    ell = p.ell_mm
    kappa = p.kappa
    E = p.youngs_modulus_MPa
    nu = p.poisson_ratio
    @assert all(
        x -> isfinite(x) && x > 0, (rho, E, gc, ell, p.impact_radius_mm, p.outer_y_mm, p.outer_z_mm)
    )
    @assert isfinite(nu) &&
        -1 < nu < 0.5 &&
        isfinite(eta) &&
        eta >= 0 &&
        isfinite(kappa) &&
        kappa >= 0
    lam = E * nu / ((1 + nu) * (1 - 2nu))
    mu = E / (2 * (1 + nu))
    (;
        n,
        ne,
        nq = 8,
        m,
        eptr = H8Pointers(ne),
        conn,
        support_element = H8SupportElement(m),
        N,
        G,
        weight,
        offsets,
        adj,
        mass = rho .* volume,
        volume,
        free,
        velocity_unit,
        coords,
        gc,
        ell,
        eta,
        eta_Pa_s = p.eta_Pa_s,
        rho,
        kappa,
        lam,
        mu,
        initial_damage = zeros(n),
        mesh_sha256 = bytes2hex(sha256(read(file))),
        purpose = "s6_rudshaug_h8_reconstruction",
        mesh_qualification = get(
            meta, "mesh_qualification", "historical globally refined paper-scale reconstruction"
        ),
    )
end
