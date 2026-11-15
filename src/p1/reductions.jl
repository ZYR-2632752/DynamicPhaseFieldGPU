# Batch diagnostic reductions; field updates and independent KKT stay unchanged.
# Six shared rows cover both mechanics (four values) and the energy ledger.
function mechanics_reduction_kernel!(parts, elastic, fracture, f, rate, mass, v, n, ne)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    s = CUDA.@cuStaticSharedMem(Float64, (6, 256))
    @inbounds begin
        s[1, lane] = i <= ne ? elastic[i] : 0.0
        s[2, lane] = i <= ne ? fracture[i] : 0.0
        s[3, lane] = i <= n ? f[1, i] * rate[1, i] + f[2, i] * rate[2, i] : 0.0
        s[4, lane] = i <= n ? mass[i] * v[1, i]^2 + mass[i] * v[2, i]^2 : 0.0
    end
    sync_threads()
    offset = 128
    while offset > 0
        if lane <= offset
            @inbounds for row in 1:4
                s[row, lane] += s[row, lane + offset]
            end
        end
        sync_threads()
        offset = div(offset, 2)
    end
    if lane == 1
        @inbounds for row in 1:4
            parts[row, blockIdx().x] = s[row, 1]
        end
    end
    return nothing
end

function energy_reduction_kernel!(
    parts, delta, z, gradient, capacity, elastic, fracture, supply, f, rate, n, ne
)
    lane = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + lane
    s = CUDA.@cuStaticSharedMem(Float64, (7, 256))
    @inbounds begin
        s[1, lane] = i <= n ? delta[i] * z[i] : 0.0
        s[2, lane] = i <= n ? gradient[i] * delta[i] : 0.0
        s[3, lane] = i <= n ? capacity[i] * delta[i]^2 : 0.0
        s[4, lane] = i <= ne ? elastic[i] : 0.0
        s[5, lane] = i <= ne ? fracture[i] : 0.0
        s[6, lane] = i <= ne ? supply[i] : 0.0
        s[7, lane] = i <= n ? f[1, i] * rate[1, i] + f[2, i] * rate[2, i] : 0.0
    end
    sync_threads()
    offset = 128
    while offset > 0
        if lane <= offset
            @inbounds for row in 1:7
                s[row, lane] += s[row, lane + offset]
            end
        end
        sync_threads()
        offset = div(offset, 2)
    end
    if lane == 1
        @inbounds for row in 1:7
            parts[row, blockIdx().x] = s[row, 1]
        end
    end
    return nothing
end

function mechanical_scalars!(parts, host, w, v)
    g = w.g
    @cuda threads = 256 blocks = size(parts, 2) mechanics_reduction_kernel!(
        parts, w.elastic, w.fracture, w.f, g.rate, g.mass, v, g.n, g.ne
    )
    copyto!(host, parts)
    vals = ntuple(row -> sum(@view(host[row, :])), 4)
    return vals[1] + vals[2], vals[3] / 0.01, 0.5vals[4]
end

function energy_scalars!(parts, host, w, delta, z, gradient, capacity, sup, qp, cached, d, dn)
    g = w.g
    @cuda threads = 256 blocks = cld(g.ne, 256) supply_kernel!(sup, qp, cached, d, dn, g)
    @cuda threads = 256 blocks = size(parts, 2) energy_reduction_kernel!(
        parts, delta, z, gradient, capacity, w.elastic, w.fracture, sup, w.f, g.rate, g.n, g.ne
    )
    copyto!(host, parts)
    vals = ntuple(row -> sum(@view(host[row, :])), 7)
    return 0.5vals[1], -vals[2], vals[3], vals[4], vals[5], vals[6], vals[7] / 0.01
end
