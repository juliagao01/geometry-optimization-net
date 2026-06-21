"""
    Geometry

Fourier-parameterized channel-with-obstacle geometry for vicinity-resistance
optimization. Mirrors the layout we prototyped in Python.

Channel: rectangle [0, L_x] x [-W/2, W/2] with four contacts named
    :contact_source  (left edge,   x = 0,   length W)
    :contact_drain   (right edge,  x = L_x, length W)
    :probe_A         (top wall,    [x_probe ± L_probe/2])
    :probe_B         (bottom wall, [x_probe ± L_probe/2])
and remaining boundary :walls.

Obstacle: one closed star-shaped curve

    r(theta) = r0 + sum_{n=1..M} (a_n cos(n theta) + b_n sin(n theta))

centered at (x_c, y_c).

Parameter vector layout (used by the optimizer):

    p = [y_c, r0, a_1, b_1, a_2, b_2, ..., a_M, b_M]

with length 2 + 2M. x_c is held fixed at the channel center; release it by
adding a leading entry if you want. M = ChannelConfig.n_modes.
"""
module Geometry

using Printf

export ChannelConfig, ObstacleParams,
       unpack_params, sample_obstacle, validate,
       write_geo

# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------

Base.@kwdef struct ChannelConfig
    L_x::Float64       = 1.0
    W::Float64         = 0.6
    x_probe::Float64   = 0.5
    L_probe::Float64   = 0.15
    x_c::Float64       = 0.5
    n_modes::Int       = 0      # <-- was 6
    margin::Float64    = 0.02
    lc::Float64        = 0.06
    n_obst_pts::Int    = 160
end

"Unpacked obstacle description."
struct ObstacleParams
    y_c::Float64
    r0::Float64
    a_cos::Vector{Float64}   # length M, indices 1..M -> modes n = 1..M
    b_sin::Vector{Float64}
end

# ---------------------------------------------------------------------------
# Parameter pack / unpack
# ---------------------------------------------------------------------------

"""
    unpack_params(p, cfg)

Decode the optimizer's flat parameter vector into an `ObstacleParams`.
Length must be `2 + 2 * cfg.n_modes`.
"""
function unpack_params(p::AbstractVector{<:Real}, cfg::ChannelConfig)
    M = cfg.n_modes
    expected = 2 + 2M
    length(p) == expected || error("got length(p)=$(length(p)), expected $expected")
    y_c = float(p[1])
    r0  = float(p[2])
    a   = zeros(M); b = zeros(M)
    @inbounds for n in 1:M
        a[n] = float(p[2 + 2n - 1])
        b[n] = float(p[2 + 2n])
    end
    return ObstacleParams(y_c, r0, a, b)
end

"""Inverse: turn an `ObstacleParams` back into a flat vector."""
function pack_params(o::ObstacleParams)
    M = length(o.a_cos)
    p = zeros(2 + 2M)
    p[1] = o.y_c; p[2] = o.r0
    @inbounds for n in 1:M
        p[2 + 2n - 1] = o.a_cos[n]
        p[2 + 2n]     = o.b_sin[n]
    end
    return p
end

# ---------------------------------------------------------------------------
# Obstacle curve
# ---------------------------------------------------------------------------

"""
    sample_obstacle(theta, o, cfg) -> (x, y)

Evaluate the polar obstacle on a vector of angles, returning Cartesian
coordinates in the global frame.
"""
function sample_obstacle(theta::AbstractVector{<:Real},
                         o::ObstacleParams, cfg::ChannelConfig)
    M = length(o.a_cos)
    r = fill(o.r0, length(theta))
    @inbounds for n in 1:M
        @. r += o.a_cos[n] * cos(n * theta) + o.b_sin[n] * sin(n * theta)
    end
    # Clip to a tiny positive number; downstream validate() will reject any
    # geometry that needed clipping by a meaningful amount.
    r .= max.(r, 1e-3)
    x = cfg.x_c .+ r .* cos.(theta)
    y = o.y_c   .+ r .* sin.(theta)
    return x, y
end

# ---------------------------------------------------------------------------
# Cheap pre-mesh validation
# ---------------------------------------------------------------------------

"""
    validate(o, cfg) -> (ok::Bool, reason::String)

Catch obviously bad geometries before invoking gmsh. We check:
- obstacle stays inside [margin, L_x - margin] in x
- obstacle stays inside [-W/2 + margin, +W/2 - margin] in y
- minimum r stays above a floor (so the spline doesn't self-cross)
- r doesn't oscillate so wildly that |dr/dtheta| > r everywhere (cardioid limit)
"""
function validate(o::ObstacleParams, cfg::ChannelConfig;
                  n_samples::Int=720)
    theta = range(0, 2pi, length=n_samples + 1)[1:end-1]
    x, y = sample_obstacle(theta, o, cfg)

    tol = 1e-6
    if minimum(x) < cfg.margin - tol
        return false, "obstacle leaves channel on the left"
    end
    if maximum(x) > cfg.L_x - cfg.margin + tol
        return false, "obstacle leaves channel on the right"
    end
    if minimum(y) < -cfg.W/2 + cfg.margin - tol
        return false, "obstacle leaves channel on the bottom"
    end
    if maximum(y) > cfg.W/2 - cfg.margin + tol
        return false, "obstacle leaves channel on the top"
    end

    # r(theta) positivity with margin
    M = length(o.a_cos)
    r = fill(o.r0, length(theta))
    @inbounds for n in 1:M
        @. r += o.a_cos[n] * cos(n * theta) + o.b_sin[n] * sin(n * theta)
    end
    if minimum(r) < 0.02
        return false, "r(theta) collapses (min r = $(round(minimum(r); digits=4)))"
    end

    # |dr/dtheta| vs r; cardioid-style self-crossing threshold
    dr = zero(theta)
    @inbounds for n in 1:M
        @. dr += -n * o.a_cos[n] * sin(n * theta) + n * o.b_sin[n] * cos(n * theta)
    end
    if maximum(abs.(dr) ./ r) > 5.0
        return false, "shape oscillates too sharply (max |r'|/r > 5)"
    end

    return true, "ok"
end

# ---------------------------------------------------------------------------
# Write a Gmsh .geo file. Boundary names match what we read in simulate.jl.
# ---------------------------------------------------------------------------

"""
    write_geo(path, o, cfg)

Emit a `.geo` file describing the channel-with-obstacle geometry. Caller is
responsible for invoking gmsh to turn it into a `.inp` file (see mesh.jl).
"""
function write_geo(path::AbstractString, o::ObstacleParams, cfg::ChannelConfig)
    L = cfg.L_x; W = cfg.W
    xPL = cfg.x_probe - cfg.L_probe/2
    xPR = cfg.x_probe + cfg.L_probe/2

    # 8 outer points, counter-clockwise, with carve-outs for the two probes
    outer = [
        (0.0,  -W/2),   # 1 source bottom
        (xPL,  -W/2),   # 2 bot probe start
        (xPR,  -W/2),   # 3 bot probe end
        (L,    -W/2),   # 4 drain bottom
        (L,    +W/2),   # 5 drain top
        (xPR,  +W/2),   # 6 top probe end
        (xPL,  +W/2),   # 7 top probe start
        (0.0,  +W/2),   # 8 source top
    ]

    theta = range(0, 2pi, length=cfg.n_obst_pts + 1)[1:end-1]
    ox, oy = sample_obstacle(theta, o, cfg)

    open(path, "w") do io
        println(io, "SetFactory(\"OpenCASCADE\");")
        @printf(io, "lc = %.6f;\n\n", cfg.lc)

        for (i, (x, y)) in enumerate(outer)
            @printf(io, "Point(%d) = {%.10f, %.10f, 0, lc};\n", i, x, y)
        end
        nouter = length(outer)
        for i in 1:nouter
            j = i == nouter ? 1 : i + 1
            @printf(io, "Line(%d) = {%d, %d};\n", i, i, j)
        end
        ids = join(1:nouter, ", ")
        println(io, "Curve Loop(1) = {", ids, "};")

        p0 = nouter + 1
        for (i, (x, y)) in enumerate(zip(ox, oy))
            @printf(io, "Point(%d) = {%.10f, %.10f, 0, lc};\n", p0 + i - 1, x, y)
        end
        pts = collect(p0:(p0 + length(ox) - 1))
        push!(pts, p0)  # close
        sline = nouter + 1
        println(io, "Spline(", sline, ") = {", join(pts, ", "), "};")
        println(io, "Curve Loop(2) = {", sline, "};")
        
        # --- Adaptive mesh refinement near the obstacle ---
        println(io)
        println(io, "// Refinement near obstacle spline")
        println(io, "Field[1] = Distance;")
        println(io, "Field[1].CurvesList = {", sline, "};")
        println(io, "Field[1].Sampling = 200;")
        println(io)
        println(io, "Field[2] = Threshold;")
        println(io, "Field[2].InField = 1;")
        @printf(io, "Field[2].SizeMin = %.6f;\n", min(cfg.lc / 5, cfg.margin / 3))
        @printf(io, "Field[2].SizeMax = %.6f;\n", cfg.lc)      # coarse: cfg.lc
        @printf(io, "Field[2].DistMin = %.6f;\n", 0.0)
        @printf(io, "Field[2].DistMax = %.6f;\n", cfg.W / 4)   # transition over W/4
        println(io)
        println(io, "Background Field = 2;")
        println(io, "Mesh.MeshSizeExtendFromBoundary = 0;")
        println(io, "Mesh.MeshSizeFromPoints = 0;")
        println(io, "Mesh.MeshSizeFromCurvature = 0;")
        println(io)

        println(io, "Plane Surface(1) = {1, 2};")
        println(io)
        println(io, "Physical Surface(\"domain\")          = {1};")
        println(io, "Physical Curve(\"contact_source\")   = {8};")
        println(io, "Physical Curve(\"contact_drain\")    = {4};")
        println(io, "Physical Curve(\"probe_A\")          = {6};")
        println(io, "Physical Curve(\"probe_B\")          = {2};")
        println(io, "Physical Curve(\"walls\")            = {1, 3, 5, 7, ", sline, "};")
    end
    return path
end

end # module
