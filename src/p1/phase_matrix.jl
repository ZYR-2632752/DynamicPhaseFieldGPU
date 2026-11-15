function kw_local_matrix!(K, q, g)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if e <= g.ne
        @inbounds for a in 1:3, b in 1:3
            K[a, b, e] =
                g.area[e] * (
                    g.gc * g.ell * (g.G[1, a, e] * g.G[1, b, e] + g.G[2, a, e] * g.G[2, b, e]) +
                    (g.gc / g.ell + 2q[e]) * (1 + (a == b)) / 12
                )
        end
    end
    return nothing
end

# Assemble the scalar phase operator on a fixed sparse pattern.
@inline function phase_bound_element!(e, out, q, g)
    top = 0.0
    @inbounds for a in 1:3
        row = 0.0
        for b in 1:3
            row += abs(
                g.gc * g.ell * (g.G[1, a, e] * g.G[1, b, e] + g.G[2, a, e] * g.G[2, b, e]) +
                (g.gc / g.ell + 2q[e]) * (1 + (a == b)) / 12,
            )
        end
        top = max(top, 3row / g.eta)
    end
    @inbounds out[e] = top
    nothing
end
function phase_bound_kernel!(out, q, g)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    e <= g.ne && phase_bound_element!(e, out, q, g)
    return nothing
end
function phase_bound!(w, q)
    g = w.g
    @cuda threads = 256 blocks = cld(g.ne, 256) phase_bound_kernel!(w.bounds, q, g)
    2 / maximum(w.bounds)
end

function sparse_workspace(g; gpu = true)
    gpu || error("This distribution contains the GPU solver only")
    # Fixed topology; only values are assembled at each accepted step.
    rows = vec(repeat(g.tri; outer = (3, 1)))
    cols = vec(repeat(g.tri; inner = (3, 1)))
    A = sparse(rows, cols, ones(length(rows)), g.n, g.n)
    nz = nnz(A)
    position = zeros(Int32, length(rows))
    counts = zeros(Int, nz)
    for k in eachindex(rows)
        j = cols[k]
        first = A.colptr[j]
        last = A.colptr[j + 1] - 1
        i = searchsortedfirst(view(A.rowval, first:last), rows[k]) + first - 1
        @assert A.rowval[i] == rows[k]
        position[k] = i
        counts[i] += 1
    end
    offsets = Int32.(vcat(1, 1 .+ cumsum(counts)))
    slot = copy(offsets[1:(end - 1)])
    localids = zeros(Int32, length(rows))
    for k in eachindex(position)
        i = position[k]
        localids[slot[i]] = k
        slot[i] += 1
    end
    @assert issymmetric(A) # A symmetric CSC layout is also its CSR layout.
    fill!(A.nzval, 0.0)
    dev = CuArray
    values = dev(A.nzval)
    (;
        A,
        K = dev(zeros(3, 3, g.ne)),
        offsets = dev(offsets),
        localids = dev(localids),
        nz,
        global_csr = (; rowptr = dev(Int32.(A.colptr)), cols = dev(Int32.(A.rowval)), values),
        workspace = BoxGPUResident.workspace(dev(zeros(g.n))),
    )
end
@inline function global_value!(i, sp)
    value = 0.0
    @inbounds for j in sp.offsets[i]:(sp.offsets[i + 1] - 1)
        value += sp.K[sp.localids[j]]
    end
    @inbounds sp.global_csr.values[i] = value
    nothing
end
function global_values_kernel!(K, offsets, localids, values, nz)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= nz
        v = 0.0
        @inbounds for j in offsets[i]:(offsets[i + 1] - 1)
            v += K[localids[j]]
        end
        @inbounds values[i] = v
    end
    return nothing
end
function assemble!(sp, q, g)
    @cuda threads = 256 blocks = cld(g.ne, 256) kw_local_matrix!(sp.K, q, g)
    @cuda threads = 256 blocks = cld(sp.nz, 256) global_values_kernel!(
        sp.K, sp.offsets, sp.localids, sp.global_csr.values, sp.nz
    )
    nothing
end
function sparse_solve(sp, q, g, cap, diagonal, rhs, lower, upper)
    assemble!(sp, q, g)
    BoxGPUResident.solve!(
        sp.workspace, sp.global_csr, cap, diagonal, rhs, lower, upper, g; tolerance = 1e-12
    )
end

function explicit_phase(method, d, q, g, dt, b, z)
    action!(z, d, q, g)
    rate = max.((b - z) ./ g.C, 0.0)
    pred = min.(d + dt * rate, 1.0)
    dn = if method == :FE
        pred
    else
        @assert method == :Heun
        action!(z, pred, q, g)
        rate2 = max.((b - z) ./ g.C, 0.0)
        min.(d + 0.5dt * (rate + rate2), 1.0)
    end
    (; x = dn, residual = 0.0, cg_total = 0)
end
