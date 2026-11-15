using Test, TOML
using DynamicPhaseFieldGPU
include("parameters.jl")

DynamicPhaseFieldGPU.CUDA.functional() || error("GPU smoke checks require an NVIDIA CUDA device")
root = dirname(@__DIR__)
mktempdir() do out
    cases = (
        (:bar, merge(example_parameters("bar"), (; nx = 8, ny = 2))),
        (:sent, merge(example_parameters("sent"), (; nx = 10, ny = 4))),
        (:kw, example_parameters("kalthoff_winkler")),
        (:branching, example_parameters("branching")),
        (:impact3d, example_parameters("impact3d")),
    )
    @testset "Five GPU entry points" begin
        for (name, p) in cases
            dir = run_example(name, p, ["--steps", "2", "--output", joinpath(out, string(name))])
            meta = TOML.parsefile(joinpath(dir, "run.toml"))
            @test meta["status"] == "completed" && meta["executed_steps"] == 2
            lines = readlines(joinpath(dir, "history.csv"))
            header = split(first(lines), ',')
            values = [parse(Float64, s) for line in lines[2:end] for s in split(line, ',')]
            data = permutedims(reshape(values, length(header), :))
            column = Dict(String(header[i]) => i for i in eachindex(header))
            @test size(data, 1) == 2
            @test maximum(data[:, column["KKT"]]) <= 1e-12
            @test minimum(data[:, column["min_delta_step"]]) >= 0
            @test maximum(data[:, column["damage_max"]]) <= 1
            @test isfile(joinpath(dir, "final.vtk"))
        end
    end
end
