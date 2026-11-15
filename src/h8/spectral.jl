module MieheSpectral3D

export miehe3d

# Three-dimensional small-strain Miehe spectral split. Pure scalar/tuple code:
# no heap allocation, LinearAlgebra dependency, host callbacks, or matrix storage.
# Component order is (xx, yy, zz, xy, yz, xz); xy/yz/xz are tensorial strains.
# lam and mu are the 3D Lame constants. deg is the already evaluated degradation
# factor, not damage itself. The return value is (stress6, psi_plus, psi_minus).
# Energies are undegraded: psi = deg * psi_plus + psi_minus.

@inline function _rotation(app::T, aqq::T, apq::T) where {T}
    if iszero(apq)
        return one(T), zero(T), zero(T)
    end
    h = (aqq - app) / T(2)
    # Equivalent to sign(tau)/(abs(tau)+sqrt(1+tau^2)), without forming
    # tau=(aqq-app)/(2apq). hypot prevents underflow for nearly repeated roots.
    t = iszero(h) ? copysign(one(T), apq) : copysign(one(T), h) * apq / (abs(h) + hypot(h, apq))
    c = inv(sqrt(one(T) + t * t))
    return c, t * c, t
end

@inline function _rotate12(a::NTuple{6,T}, v1, v2, v3) where {T}
    c, s, t = _rotation(a[1], a[2], a[4])
    b = (a[1] - t * a[4], a[2] + t * a[4], a[3], zero(T), s * a[6] + c * a[5], c * a[6] - s * a[5])
    w1 = (c * v1[1] - s * v2[1], c * v1[2] - s * v2[2], c * v1[3] - s * v2[3])
    w2 = (s * v1[1] + c * v2[1], s * v1[2] + c * v2[2], s * v1[3] + c * v2[3])
    return b, w1, w2, v3
end

@inline function _rotate13(a::NTuple{6,T}, v1, v2, v3) where {T}
    c, s, t = _rotation(a[1], a[3], a[6])
    b = (a[1] - t * a[6], a[2], a[3] + t * a[6], c * a[4] - s * a[5], s * a[4] + c * a[5], zero(T))
    w1 = (c * v1[1] - s * v3[1], c * v1[2] - s * v3[2], c * v1[3] - s * v3[3])
    w3 = (s * v1[1] + c * v3[1], s * v1[2] + c * v3[2], s * v1[3] + c * v3[3])
    return b, w1, v2, w3
end

@inline function _rotate23(a::NTuple{6,T}, v1, v2, v3) where {T}
    c, s, t = _rotation(a[2], a[3], a[5])
    b = (a[1], a[2] - t * a[5], a[3] + t * a[5], c * a[4] - s * a[6], zero(T), s * a[4] + c * a[6])
    w2 = (c * v2[1] - s * v3[1], c * v2[2] - s * v3[2], c * v2[3] - s * v3[3])
    w3 = (s * v2[1] + c * v3[1], s * v2[2] + c * v3[2], s * v2[3] + c * v3[3])
    return b, v1, w2, w3
end

@inline function miehe3d(e::NTuple{6,T}, lam::T, mu::T, deg::T) where {T<:AbstractFloat}
    z = zero(T)
    scale = max(abs(e[1]), abs(e[2]), abs(e[3]), abs(e[4]), abs(e[5]), abs(e[6]))
    if iszero(scale)
        return (z, z, z, z, z, z), z, z
    end
    a = (e[1] / scale, e[2] / scale, e[3] / scale, e[4] / scale, e[5] / scale, e[6] / scale)
    v1 = (one(T), z, z)
    v2 = (z, one(T), z)
    v3 = (z, z, one(T))
    # Fixed work and no eigenvalue-gap division: repeated eigenvalues and zero
    # roots are safe. Ten cyclic Jacobi sweeps are checked against a CPU oracle.
    for _ in 1:10
        a, v1, v2, v3 = _rotate12(a, v1, v2, v3)
        a, v1, v2, v3 = _rotate13(a, v1, v2, v3)
        a, v1, v2, v3 = _rotate23(a, v1, v2, v3)
    end
    l1, l2, l3 = a[1] * scale, a[2] * scale, a[3] * scale
    p1, p2, p3 = max(l1, z), max(l2, z), max(l3, z)
    m1, m2, m3 = min(l1, z), min(l2, z), min(l3, z)
    tr = e[1] + e[2] + e[3]
    tp, tm = max(tr, z), min(tr, z)
    psip = (lam / T(2)) * tp * tp + mu * (p1 * p1 + p2 * p2 + p3 * p3)
    psim = (lam / T(2)) * tm * tm + mu * (m1 * m1 + m2 * m2 + m3 * m3)
    b1 = T(2) * mu * (deg * p1 + m1)
    b2 = T(2) * mu * (deg * p2 + m2)
    b3 = T(2) * mu * (deg * p3 + m3)
    vol = lam * (deg * tp + tm)
    stress = (
        vol + b1 * v1[1] * v1[1] + b2 * v2[1] * v2[1] + b3 * v3[1] * v3[1],
        vol + b1 * v1[2] * v1[2] + b2 * v2[2] * v2[2] + b3 * v3[2] * v3[2],
        vol + b1 * v1[3] * v1[3] + b2 * v2[3] * v2[3] + b3 * v3[3] * v3[3],
        b1 * v1[1] * v1[2] + b2 * v2[1] * v2[2] + b3 * v3[1] * v3[2],
        b1 * v1[2] * v1[3] + b2 * v2[2] * v2[3] + b3 * v3[2] * v3[3],
        b1 * v1[1] * v1[3] + b2 * v2[1] * v2[3] + b3 * v3[1] * v3[3],
    )
    return stress, psip, psim
end

# Scalar integration API returns a flat eight-tuple for existing element kernels.
@inline function miehe3d(
    exx::T, eyy::T, ezz::T, exy::T, eyz::T, exz::T, lam::T, mu::T, deg::T
) where {T<:AbstractFloat}
    s, p, m = miehe3d((exx, eyy, ezz, exy, eyz, exz), lam, mu, deg)
    return s[1], s[2], s[3], s[4], s[5], s[6], p, m
end

end # module
