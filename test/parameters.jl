function example_parameters(name)
    scope = Module(gensym(:Example))
    file = joinpath(@__DIR__, "..", "examples", name * ".jl")
    # Read parameter declarations without importing CUDA or running an example.
    Base.include(expr -> expr isa Expr && expr.head == :using ? nothing : expr, scope, file)
    Base.invokelatest(getfield, scope, :PARAMETERS)
end
