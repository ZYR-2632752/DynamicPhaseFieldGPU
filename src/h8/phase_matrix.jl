# One H8 implementation: scalar CSR, diagonal eta*volume/dt, incremental box BE.
# No full-mesh COO/sorting arrays and no expanded quadrature geometry.
using LinearAlgebra

function csr_pattern(g)
    rowptr = Vector{Int32}(undef, g.n + 1)
    rowptr[1] = 1
    maxsupport = maximum(diff(g.offsets))
    # Julia 1.12 may number default workers after an interactive thread.
    slots = isdefined(Threads, :maxthreadid) ? Threads.maxthreadid() : Threads.nthreads()
    scratch = [Vector{Int32}(undef, 8maxsupport) for _ in 1:slots]
    function neighbors!(i)
        tmp = scratch[Threads.threadid()]
        k = 0
        @inbounds for j in g.offsets[i]:(g.offsets[i + 1] - 1)
            a = g.adj[j]
            e = g.support_element[a]
            for b in g.eptr[e]:(g.eptr[e + 1] - 1)
                k += 1
                tmp[k] = g.conn[b]
            end
        end
        sort!(view(tmp, 1:k))
        nunique = 0
        previous = Int32(0)
        @inbounds for j in 1:k
            v = tmp[j]
            if v != previous
                nunique += 1
                tmp[nunique] = v
                previous = v
            end
        end
        nunique
    end
    Threads.@threads :static for i in 1:g.n
        rowptr[i + 1] = neighbors!(i)
    end
    for i in 1:g.n
        total = Int64(rowptr[i]) + rowptr[i + 1]
        total <= typemax(Int32) || error("CSR exceeds Int32")
        rowptr[i + 1] = total
    end
    cols = Vector{Int32}(undef, Int(rowptr[end]) - 1)
    Threads.@threads :static for i in 1:g.n
        count = neighbors!(i)
        count == rowptr[i + 1] - rowptr[i] || error("CSR count mismatch")
        copyto!(cols, rowptr[i], scratch[Threads.threadid()], 1, count)
    end
    (; rowptr, cols)
end

function phase_workspace(g, pattern; gpu = true)
    gpu || error("This distribution contains the GPU solver only")
    z(dims...) = fill!(similar(g.mass, Float64, dims...), 0.0)
    (;
        K = (;
            kind = Val(:csr),
            rowptr = pattern.rowptr,
            cols = pattern.cols,
            values = z(length(pattern.cols)),
        ),
        localvalues = z(8, 8, g.ne),
        localrhs = z(g.m),
        rhs = z(g.n),
        diagA = z(g.n),
        diagQ = z(g.n),
        capacity = z(g.n),
        upper = fill!(similar(g.mass), 1.0),
        audit = z(g.n),
        action = z(g.n),
        gpu,
    )
end

@inline function phase_element!(e, values, rhs, psi, g)
    first = g.eptr[e]
    @inbounds for a in 1:8
        ka = first + a - 1
        bvalue = 0.0
        for q in 1:8
            bvalue += 2g.weight[q, e] * psi[q, e] * g.N[q, ka]
        end
        rhs[ka] = bvalue
        for b in 1:8
            kb = first + b - 1
            value = 0.0
            for q in 1:8
                nn = g.N[q, ka] * g.N[q, kb]
                gg =
                    g.G[1, q, ka] * g.G[1, q, kb] +
                    g.G[2, q, ka] * g.G[2, q, kb] +
                    g.G[3, q, ka] * g.G[3, q, kb]
                value += g.weight[q, e] * ((g.gc / g.ell + 2psi[q, e]) * nn + g.gc * g.ell * gg)
            end
            values[a, b, e] = value
        end
    end
    nothing
end

@inline function phase_row!(i, K, localvalues, localrhs, rhs, diagA, g)
    @inbounds begin
        start = K.rowptr[i]
        stop = K.rowptr[i + 1] - 1
        for j in start:stop
            K.values[j] = 0.0
        end
        bi = 0.0
        for j in g.offsets[i]:(g.offsets[i + 1] - 1)
            ka = g.adj[j]
            e = g.support_element[ka]
            a = ka - g.eptr[e] + 1
            bi += localrhs[ka]
            for b in 1:8
                target = g.conn[g.eptr[e] + b - 1]
                lo = start
                hi = stop
                while lo < hi
                    mid = (lo + hi) >>> 1
                    if K.cols[mid] < target
                        lo = mid + 1
                    else
                        hi = mid
                    end
                end
                K.values[lo] += localvalues[a, b, e]
            end
        end
        rhs[i] = bi
        for j in start:stop
            if K.cols[j] == i
                diagA[i] = K.values[j]
                break
            end
        end
    end
    nothing
end

function assemble_phase!(p, g, psi, dt, lower)
    dt > 0 && isfinite(dt) || error("Invalid dt")
    size(psi) == (8, g.ne) && length(lower) == g.n || error("Phase dimensions")
    H8Operators.eachwork!(
        phase_element!, g.ne, p.gpu, p.localvalues, p.localrhs, psi, H8Operators.operator_grid(g)
    )
    H8Operators.eachwork!(
        phase_row!,
        g.n,
        p.gpu,
        p.K,
        p.localvalues,
        p.localrhs,
        p.rhs,
        p.diagA,
        H8Operators.operator_grid(g),
    )
    p.capacity .= g.eta .* g.volume ./ dt
    p.diagQ .= p.diagA .+ p.capacity
    p.rhs .+= p.capacity .* lower
    p
end

@inline function csr_action_row!(i, y, x, K, capacity)
    value = 0.0
    @inbounds for j in K.rowptr[i]:(K.rowptr[i + 1] - 1)
        value += K.values[j] * x[K.cols[j]]
    end
    @inbounds y[i] = value + capacity[i] * x[i]
    nothing
end
function phase_action!(y, x, p, g)
    Base.mightalias(y, x) && error("Action cannot overwrite input")
    H8Operators.eachwork!(csr_action_row!, g.n, p.gpu, y, x, p.K, p.capacity)
    y
end

function phase_check!(p, g, d, lower)
    phase_action!(p.audit, d, p, g)
    p.audit .-= p.rhs
    kkt = maximum(abs.(clamp.(d .- p.audit ./ p.diagQ, lower, p.upper) .- d))
    (;
        kkt,
        min_damage = minimum(d),
        max_damage = maximum(d),
        min_increment = minimum(d .- lower),
        fixed = count(==(true), p.upper .== lower),
        lower_active = count(==(true), (d .== lower) .& (p.upper .> lower)),
        upper_active = count(==(true), (d .== p.upper) .& (p.upper .> lower)),
        free = count(==(true), (d .> lower) .& (d .< p.upper)),
    )
end
