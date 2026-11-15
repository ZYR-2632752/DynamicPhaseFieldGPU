using Test, LinearAlgebra, TOML, SHA
const ROOT = dirname(@__DIR__)
include("parameters.jl")
const BAR = example_parameters("bar")
const SENT = example_parameters("sent")
const KW = example_parameters("kalthoff_winkler")
const BRANCHING = example_parameters("branching")
const H8 = example_parameters("impact3d")

module Q1Host
include("../src/q1/mesh.jl")
include("../src/q1/action.jl")
end
module SENTHost
include("../src/p1/grid.jl")
end
module KWHost
using LinearAlgebra, TOML, SHA
include("../src/p1/mesh_io.jl")
include("../src/p1/kw_mesh.jl")
end
module H8Host
using LinearAlgebra
include("../src/h8/mesh.jl")
include("../src/h8/phase_matrix.jl")
include("../src/h8/boundary.jl")
end
include("../src/branching_mesh.jl")
include("../src/output.jl")

@testset "Example parameter propagation" begin
    p = merge(BAR, (; nx = 4, ny = 2))
    g1 = Q1Host.make_grid(p)
    g2 = Q1Host.make_grid(merge(p, (; density_kg_m3 = 2p.density_kg_m3)))
    @test g2.grid.mass == 2g1.grid.mass
    @test g2.dt ≈ sqrt(2) * g1.dt

    p = merge(SENT, (; nx = 4, ny = 2, loading_rate_m_s = 0.02, length_m = 0.2))
    g = SENTHost.grid(p)
    @test all(g.rate[2, (end - g.nx):end] .== p.loading_rate_m_s)
    @test sum(g.area) ≈ p.length_m * p.height_m

    p = merge(H8, (; density_tonne_mm3 = 2H8.density_tonne_mm3))
    g = H8Host.load_compact_mesh(p.mesh, p)
    @test g.mass == p.density_tonne_mm3 .* g.volume
    bc = H8Host.impact_kinematics(1e-6; speed_mm_s = 20000.0, ramp_s = 2e-6)
    @test bc.v == 10000.0
    @test bc.a == 1e10

    @test command_options(String[]; steps = 7, out = "results").steps == 7
    @test command_options(String[]; steps = 7, out = "results").out == "results"
    @test command_options(["--steps", "3"]; steps = 7).steps == 3
    @test_throws Exception command_options(String[]; steps = 0)
end

@testset "Q1 topology, loading and operator" begin
    p = merge(BAR, (; nx = 4, ny = 2, stress_Pa = 123e6))
    host = Q1Host.make_grid(p)
    g = host.grid
    n = g.n
    @test size(host.xy) == (2, n)
    @test sum(g.volume) ≈ 0.05
    @test sum(g.loads) ≈ 123e6 * 0.05
    @test all(g.free[2, (end - g.nx):end] .== 0)
    @test host.dt > 0
    ne = size(g.cells, 2)
    values = zeros(4, 4, ne)
    A = zeros(n, n)
    for e in 1:ne
        B = [sin(i + 3j + e) for i in 1:4, j in 1:4]
        values[:, :, e] .= B' * B + I
        for a in 1:4, b in 1:4
            A[g.cells[a, e], g.cells[b, e]] += values[a, b, e]
        end
    end
    A = Diagonal(g.upper) * A * Diagonal(g.upper) + Diagonal(1 .- g.upper)
    x = sin.(1:n)
    @test [Q1Host.q1_row(i, x, values, g) for i in 1:n] ≈ A * x rtol = 1e-14
    @test_throws Exception Q1Host.make_grid(merge(p, (; ny = 3)))
    @test_throws Exception Q1Host.make_grid(merge(p, (; eta_Pa_s = -1)))
    @test_throws Exception Q1Host.make_grid(merge(p, (; stress_Pa = NaN)))
end

@testset "SENT mesh and symmetry boundary" begin
    g = SENTHost.grid(SENT)
    @test (g.n, g.ne) == (23005, 45368)
    @test sum(g.area) ≈ 0.005
    @test g.dtmechanical > 0
    @test all(g.rate[2, (end - g.nx):end] .== 0.01)
    @test_throws Exception SENTHost.grid(merge(SENT, (; nx = 5, ny = 2)))
end

@testset "KW input identity" begin
    folder = joinpath(ROOT, "meshes/kw")
    m = TOML.parsefile(joinpath(folder, "manifest.toml"))
    check_file_hashes(folder, m["mesh_sha256"])
    p = KWHost.load_research_problem(
        folder;
        expected_size = (m["nodes"], m["elements"]),
        expected_boundaries = (m["impact_nodes"], m["bottom_nodes"]),
    )
    @test length(p.mass) == 260361
    @test all(p.mass .> 0) && all(p.volume .> 0)
    @test_throws Exception KWHost.read_research_npy(joinpath(folder, "mass.npy"), (2,), ("<f8",))
end

@testset "Branching mesh, traction and seed" begin
    folder = joinpath(ROOT, "meshes/branching")
    g, f = BranchingMesh.load_grid(joinpath(folder, "mesh"), BRANCHING)
    seed, _ = BranchingMesh.read_seed(
        g,
        f,
        joinpath(folder, "seed");
        mesh_manifest_sha256 = BranchingMesh.file_sha(joinpath(folder, "mesh/manifest.toml")),
    )
    @test (g.n, g.ne) == (180301, 360020)
    @test sum(g.area) ≈ 4000
    @test maximum(abs.(sum(f; dims = 2))) < 1e-10
    @test sum(max.(f[2, :], 0)) ≈ 100
    @test all(0 .<= seed.d .<= 1)
    @test seed.a == f ./ reshape(g.mass, 1, :)
    partner = Vector{Int32}(undef, g.n)
    read!(joinpath(folder, "mesh/mirror_partner.i32"), partner)
    @test all(1 .<= partner .<= g.n)
    @test partner[partner] == collect(1:g.n)
    @test maximum(abs.(seed.d .- seed.d[partner])) < 1e-10
    @test_throws AssertionError BranchingMesh.read_seed(
        merge(g, (; ell = 2g.ell)),
        f,
        joinpath(folder, "seed");
        mesh_manifest_sha256 = BranchingMesh.file_sha(joinpath(folder, "mesh/manifest.toml")),
    )
end

@testset "H8 compact input and sparse connectivity" begin
    g = H8Host.load_compact_mesh(joinpath(ROOT, "meshes/h8_fixture"), H8)
    @test (g.n, g.ne) == (218, 120)
    @test sum(g.volume) ≈ 1e6
    pattern = H8Host.csr_pattern(g)
    @test pattern.rowptr[end] == length(pattern.cols) + 1
    for i in 1:g.n
        row = pattern.cols[pattern.rowptr[i]:(pattern.rowptr[i + 1] - 1)]
        @test issorted(row) && length(row) == length(unique(row)) && i in row
    end
end

@testset "CLI and output" begin
    @test command_options(String[]).steps == 100
    @test command_options(["--steps", "3"]).steps == 3
    @test_throws Exception command_options(["--steps", "0"])
    @test_throws Exception command_options(["--full", "--steps", "3"])
    @test_throws Exception command_options(["--output"])
    @test_throws Exception command_options(["--unknown"])
    mktempdir() do tmp
        out = new_output(joinpath(tmp, "run"))
        @test_throws Exception new_output(out)
        file = joinpath(out, "field.vtk")
        xy = [0.0 1.0 0.0; 0.0 0.0 1.0]
        cells = reshape([1, 2, 3], 3, 1)
        write_vtk(file, xy, cells, 5, zero(xy), [0.0, 0.5, 1.0])
        bytes = read(file)
        header = Vector{UInt8}(codeunits("POINTS 3 double\n"))
        pos = findfirst(header, bytes)
        @test !isnothing(pos)
        start = last(pos) + 1
        xyz = reinterpret(Float64, ntoh.(reinterpret(UInt64, bytes[start:(start + 71)])))
        @test xyz == vec(vcat(xy, zeros(1, 3)))
        @test_throws Exception write_vtk(file, xy, cells, 5, zero(xy), zeros(3))
        @test_throws Exception check_file_hashes(tmp, Dict("../outside" => "bad"))
    end
end

@testset "Syntax and example inventory" begin
    @test sort(readdir(joinpath(ROOT, "examples"))) ==
        sort(["bar.jl", "sent.jl", "kalthoff_winkler.jl", "branching.jl", "impact3d.jl"])
    for folder in ("src", "examples"),
        (dir, _, files) in walkdir(joinpath(ROOT, folder)),
        name in files

        endswith(name, ".jl") || continue
        parsed = Meta.parseall(read(joinpath(dir, name), String))
        @test !any(x -> x isa Expr && x.head in (:error, :incomplete), parsed.args)
    end
end
