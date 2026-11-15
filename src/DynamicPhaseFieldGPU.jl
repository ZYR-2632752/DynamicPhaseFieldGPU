module DynamicPhaseFieldGPU

using CUDA, Dates, LinearAlgebra, SHA, TOML

include("p1/integration.jl")
include("device_step.jl")
include("kw_step.jl")
include("subcycling.jl")
include("traction_step.jl")
include("branching_mesh.jl")

module BarQ1
    using CUDA, LinearAlgebra
    using ..P1V2: BoxKWOptimized
    include("q1/mesh.jl")
    include("q1/operators.jl")
    include("q1/integration.jl")
end

module Impact3D
    using CUDA, LinearAlgebra, TOML
    include("h8/mesh.jl")
    include("h8/operators.jl")
    include("h8/phase_matrix.jl")
    include("h8/dynamics.jl")
    include("h8/gpu.jl")
end

include("output.jl")
include("run_cases.jl")

export run_example
end
