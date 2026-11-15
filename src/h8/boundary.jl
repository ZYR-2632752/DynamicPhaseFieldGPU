# Prescribed impact speed, N-mm-s units. The velocity direction is mesh data.
# At the ramp corner acceleration uses its right limit; velocity and displacement
# remain continuous. Inertial boundary work is integrated by its exact energy change.
function impact_kinematics(t; speed_mm_s, ramp_s)
    @assert isfinite(t) && t >= 0 && speed_mm_s > 0 && ramp_s > 0
    if t < ramp_s
        return (;
            u = speed_mm_s * t^2 / (2ramp_s), v = speed_mm_s * t / ramp_s, a = speed_mm_s / ramp_s
        )
    end
    (; u = speed_mm_s * (t - ramp_s / 2), v = speed_mm_s, a = 0.0)
end

function fixed_impact_step(bound; reference_end_s = 141.4e-6)
    @assert isfinite(bound) && bound > 0
    # Factor 1 bound, rounded once to reach the reference endpoint exactly.
    # Snapshot output uses nearest actual times; the ramp may fall inside a step.
    reference_end_s / ceil(Int, reference_end_s / bound)
end

function impose_impact!(u, v, a, free, velocity_unit, t; kwargs...)
    bc = impact_kinematics(t; kwargs...)
    u .= free .* u .+ velocity_unit .* bc.u
    v .= free .* v .+ velocity_unit .* bc.v
    a .= free .* a .+ velocity_unit .* bc.a
    bc
end

# Total applied support force = inertia + internal force, with no traction load.
function impact_reaction(internal, a, mass, free)
    (1 .- free) .* (internal .+ reshape(mass, 1, :) .* a)
end
reaction_work(r0, r1, u0, u1) = sum((r0 .+ r1) .* (u1 .- u0)) / 2

function dirichlet_work(f0, f1, u0, u1, v0, v1, mass, free)
    constrained = 1 .- free
    internal = sum(constrained .* (f0 .+ f1) .* (u1 .- u0)) / 2
    inertia = sum(constrained .* reshape(mass, 1, :) .* (v1 .^ 2 .- v0 .^ 2)) / 2
    internal + inertia
end
