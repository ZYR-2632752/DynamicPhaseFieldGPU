using CUDA
import CUDACore: Adapt
using ..P1V2: BoxKWOptimized
Adapt.@adapt_structure SectionShape
Adapt.@adapt_structure SectionGradient
Adapt.@adapt_structure SectionWeight
Adapt.adapt_structure(to, x::H8Pointers) = x
Adapt.adapt_structure(to, x::H8SupportElement) = x
function device_grid(g)
    names = (
        :eptr,
        :conn,
        :support_element,
        :N,
        :G,
        :weight,
        :offsets,
        :adj,
        :mass,
        :volume,
        :free,
        :velocity_unit,
        :initial_damage,
    )
    arrays = NamedTuple{names}(Tuple(Adapt.adapt(CuArray, getproperty(g, k)) for k in names))
    merge(
        (;
            n = g.n,
            ne = g.ne,
            nq = g.nq,
            m = g.m,
            gc = g.gc,
            ell = g.ell,
            eta = g.eta,
            rho = g.rho,
            kappa = g.kappa,
            lam = g.lam,
            mu = g.mu,
            eta_Pa_s = g.eta_Pa_s,
        ),
        arrays,
    )
end
function work_kernel!(f::F, n, args::Vararg{Any,N}) where {F,N}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= n && f(i, args...)
    return nothing
end
function H8Operators.gpu_work!(f, n, args...)
    @cuda threads = 128 blocks = cld(n, 128) work_kernel!(f, n, args...)
    nothing
end
backend_sync() = CUDA.synchronize()

function resident_solver(p, g, lower)
    ws = BoxKWOptimized.workspace(lower; maxblocks = 384)
    descriptor = (; n = g.n)
    kernel = @cuda launch = false BoxKWOptimized.kernel!(
        ws, p.K, p.capacity, p.diagQ, p.rhs, lower, p.upper, descriptor, 1e-12, 300, true, true
    )
    per_sm = CUDA.active_blocks(kernel.fun, 128)
    sms = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    limit = per_sm * sms
    limit > 0 || error("No cooperative residency")
    blocks = min(64, limit, 384)
    monitor = CUDA.zeros(Float64, 2)
    function solvebox(p, dold)
        BoxKWOptimized.launch_box!(
            ws,
            p.K,
            p.capacity,
            p.diagQ,
            p.rhs,
            dold,
            p.upper,
            descriptor;
            tolerance = 1e-12,
            blocks,
            skip_fixed = true,
            fused = true,
        )
        @cuda threads = 1 blocks = 1 accumulate_solver!(monitor, ws.ctrl)
        result = BoxKWOptimized.read_box_result(ws; own_result = true)
        result
    end
    (; solvebox, ws, monitor, blocks, per_sm, sms, limit)
end
function accumulate_solver!(monitor, ctrl)
    if threadIdx().x == 1 && blockIdx().x == 1
        @inbounds begin
            monitor[1] += (ctrl[1] != 1 || !isfinite(ctrl[2]) || ctrl[2] > 1e-12)
            monitor[2] = max(monitor[2], ctrl[2])
        end
    end
    return nothing
end
function check_monitor(solver)
    m = Array(solver.monitor)
    m[1] == 0 && isfinite(m[2]) && m[2] <= 1e-12 || error("Cumulative device failure: $m")
    m
end
function device_snapshot()
    Dict(
        "free_bytes" => Int(CUDA.free_memory()),
        "total_bytes" => Int(CUDA.total_memory()),
        "gpu" => CUDA.name(CUDA.device()),
        "uuid" => string(CUDA.uuid(CUDA.device())),
        "julia" => string(VERSION),
        "cuda_jl" => get(
            TOML.parsefile(joinpath(dirname(dirname(pathof(CUDA))), "Project.toml")),
            "version",
            "unknown",
        ),
        "driver" => string(CUDA.driver_version()),
        "runtime" => string(CUDA.runtime_version()),
    )
end
