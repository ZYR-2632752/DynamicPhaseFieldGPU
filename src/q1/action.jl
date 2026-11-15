# A node gathers its adjacent Q1 cells. Fixed damage nodes have an identity row.
@inline function q1_row(i, x, values, g)
    @inbounds g.upper[i] == 0 && return x[i]
    ix = (i - 1) % (g.nx + 1)
    iy = (i - 1) ÷ (g.nx + 1)
    value = 0.0
    @inbounds for ey in max(0, iy - 1):min(g.ny - 1, iy), ex in max(0, ix - 1):min(g.nx - 1, ix)
        e = 1 + ex + g.nx * ey
        a = iy == ey ? (ix == ex ? 1 : 2) : (ix == ex ? 4 : 3)
        for b in 1:4
            j = g.cells[b, e]
            value += values[a, b, e] * x[j] * g.upper[j]
        end
    end
    value
end
