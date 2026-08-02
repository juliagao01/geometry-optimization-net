"""
    FixedMesh

Density-based ("fixed mesh" / topology-optimization) representation of the
channel-with-obstacle problem.

Instead of remeshing an obstacle boundary for every candidate shape (the
`Geometry` + `Mesh` path), we mesh the *empty* channel **once** and represent
the obstacle as a per-cell density field `ρ(x) ∈ [0, 1]` overlaid on that fixed
mesh. `ρ = 0` is open fluid, `ρ = 1` is solid obstacle, and intermediate values
are partially-filled cells (this is what makes the objective smooth in ρ, so a
gradient / continuation method can be run on top).

The obstacle is realized physically by **Brinkman penalization**: in the
Boltzmann-moment model, a solid cell is one where current cannot flow, i.e. a
locally huge momentum-relaxing collision rate. We add a source term

    S_brinkman(u, x) = -α · ρ(x) · diag(0, 1, 1, …, 1) · u

which damps every momentum harmonic (a1, b1, a2, …) — leaving the conserved
density mode a0 untouched — by `α · ρ(x)`. As ρ → 1 the local current is driven
to zero, so flow routes around the cell exactly as it would around a wall.

Because the mesh is fixed, one `semidiscretize`/projector setup is reused across
all ρ evaluations — only the ODE solve reruns. Mutating `field.rho` in place
changes the operator seen by the next solve.
"""
module FixedMesh

using FermiSea
using Trixi
using OrdinaryDiffEq
using Krylov
using LinearMaps
using SparseArrays
using Printf
using LinearAlgebra: norm, lu, dot
using ..Geometry: ChannelConfig
using ..Simulate: SimConfig

export DensityField, BrinkmanSource,
       write_channel_geo, write_channel_geo_symmetric, write_point_geo,
       write_vicinity_geo, paint_blob!, clear!,
       FixedEvaluator, run_f1_fixed, run_f1_fixed_steady,
       FixedOperator, assemble_fixed_operator, f1_exact, f1_adjoint_grad,
       f1_exact_perdof, fourier_perdof

# ---------------------------------------------------------------------------
# Density field on a fixed background grid
# ---------------------------------------------------------------------------

"""
    DensityField(cfg; nx, ny, alpha_max)

A Cartesian grid of design densities `ρ[i, j] ∈ [0, 1]` covering the channel
`[0, L_x] × [-W/2, W/2]`, with `nx` cells along x and `ny` along y. `alpha_max`
is the Brinkman penalization strength at ρ = 1 (the local momentum-relaxing rate
of a fully-solid cell). The `rho` matrix is mutable and read live by
`BrinkmanSource`.
"""
mutable struct DensityField
    x0::Float64
    y0::Float64
    dx::Float64
    dy::Float64
    nx::Int
    ny::Int
    alpha_max::Float64
    rho::Matrix{Float64}   # size (nx, ny); rho[i, j], i↔x, j↔y
end

function DensityField(cfg::ChannelConfig; nx::Int=24, ny::Int=16,
                      alpha_max::Real=500.0)
    x0 = 0.0
    y0 = -cfg.W / 2
    dx = cfg.L_x / nx
    dy = cfg.W / ny
    return DensityField(x0, y0, dx, dy, nx, ny, Float64(alpha_max),
                        zeros(Float64, nx, ny))
end

"Cell-center coordinate of design cell (i, j)."
@inline cell_center(f::DensityField, i::Int, j::Int) =
    (f.x0 + (i - 0.5) * f.dx, f.y0 + (j - 0.5) * f.dy)

"Zero out the density field."
clear!(f::DensityField) = (fill!(f.rho, 0.0); f)

"""
    penalization(field, px, py) -> α · ρ(px, py)

O(1) lookup of the Brinkman strength at physical point `(px, py)`. Points
outside the grid clamp to the nearest edge cell (they are never solid in
practice because the optimizer keeps ρ = 0 near the boundary).
"""
@inline function penalization(f::DensityField, px::Real, py::Real)
    i = clamp(floor(Int, (px - f.x0) / f.dx) + 1, 1, f.nx)
    j = clamp(floor(Int, (py - f.y0) / f.dy) + 1, 1, f.ny)
    return f.alpha_max * f.rho[i, j]
end

# ---------------------------------------------------------------------------
# Brinkman source term (spatially-varying momentum damping)
# ---------------------------------------------------------------------------

"""
    BrinkmanSource(field; damp_a0=false)

Trixi source term that damps all momentum harmonics by `α · ρ(x)`, leaving the
density mode a0 undamped (a reflecting-wall-like obstacle). With `damp_a0=true`
the density mode is damped too — an *absorbing* obstacle whose interior is truly
dead (no floating undamped-density region), for which `f_1` converges to a finite
hard-wall limit as `α → ∞` (the default momentum-only obstacle has `f_1 ∝ α`).
"""
struct BrinkmanSource
    field::DensityField
    damp_a0::Bool
end
BrinkmanSource(field::DensityField; damp_a0::Bool=false) =
    BrinkmanSource(field, damp_a0)

@inline function (src::BrinkmanSource)(u, x, t,
        equations::FermiSea.IsotropicFermiHarmonics2D{NVARS}) where {NVARS}
    σ = penalization(src.field, x[1], x[2])
    T = eltype(u)
    if src.damp_a0
        return Trixi.SVector{NVARS, T}(ntuple(i -> -T(σ) * u[i], Val(NVARS)))
    else
        return Trixi.SVector{NVARS, T}(ntuple(i -> i == 1 ? zero(T) : -T(σ) * u[i],
                                              Val(NVARS)))
    end
end

# ---------------------------------------------------------------------------
# Painting obstacles into the density field
# ---------------------------------------------------------------------------

"""
    paint_blob!(field, xc, yc, r; width=nothing, additive=false)

Paint a smooth (partially-filled) circular obstacle of radius `r` centered at
`(xc, yc)` into `field.rho`. Cells deep inside get ρ ≈ 1, cells far outside
ρ ≈ 0, with a smooth sigmoid transition of half-width `width` (defaults to one
cell diagonal) — this is the "partial fill" that keeps the map differentiable.

By default the field is overwritten; pass `additive=true` to superimpose onto
existing density (clamped to 1).
"""
function paint_blob!(f::DensityField, xc::Real, yc::Real, r::Real;
                     width::Union{Nothing,Real}=nothing, additive::Bool=false)
    w = width === nothing ? hypot(f.dx, f.dy) : Float64(width)
    additive || fill!(f.rho, 0.0)
    @inbounds for j in 1:f.ny, i in 1:f.nx
        cx, cy = cell_center(f, i, j)
        d = hypot(cx - xc, cy - yc)
        val = 1.0 / (1.0 + exp((d - r) / w))   # ≈1 inside, ≈0 outside
        f.rho[i, j] = additive ? min(1.0, f.rho[i, j] + val) : val
    end
    return f
end

# ---------------------------------------------------------------------------
# Fixed channel geometry (no obstacle) -> .geo
# ---------------------------------------------------------------------------

"""
    write_channel_geo(path, cfg)

Emit a `.geo` for the *empty* channel: rectangle `[0, L_x] × [-W/2, W/2]` with
the four contacts (source/drain on the ends, probe_A/probe_B carved into the top
and bottom walls) and `walls` for the remaining boundary. No obstacle — that is
supplied later as a density field. Boundary names match `Simulate`'s BC keys.
"""
function write_channel_geo(path::AbstractString, cfg::ChannelConfig)
    L = cfg.L_x; W = cfg.W
    xPL = cfg.x_probe - cfg.L_probe / 2
    xPR = cfg.x_probe + cfg.L_probe / 2

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

    open(path, "w") do io
        println(io, "SetFactory(\"OpenCASCADE\");")
        @printf(io, "lc = %.6f;\n\n", cfg.lc)
        for (i, (x, y)) in enumerate(outer)
            @printf(io, "Point(%d) = {%.10f, %.10f, 0, lc};\n", i, x, y)
        end
        n = length(outer)
        for i in 1:n
            j = i == n ? 1 : i + 1
            @printf(io, "Line(%d) = {%d, %d};\n", i, i, j)
        end
        println(io, "Curve Loop(1) = {", join(1:n, ", "), "};")
        println(io, "Plane Surface(1) = {1};")
        println(io)
        println(io, "Physical Surface(\"domain\")        = {1};")
        println(io, "Physical Curve(\"contact_source\") = {8};")
        println(io, "Physical Curve(\"contact_drain\")  = {4};")
        println(io, "Physical Curve(\"probe_A\")        = {6};")
        println(io, "Physical Curve(\"probe_B\")        = {2};")
        println(io, "Physical Curve(\"walls\")          = {1, 3, 5, 7};")
    end
    return path
end

"""
    write_channel_geo_symmetric(path, cfg; h=cfg.lc)

Emit a `.geo` for the empty channel as three **transfinite (structured)** strips
`[0,xPL] | [xPL,xPR] | [xPR,L]` in x, all sharing one *uniform* y-partition.
Because the y node distribution is identical and uniform across strips, the
resulting quad mesh is **exactly mirror-symmetric about y = 0** — so a centered
obstacle gives `f_1 = 0` to machine precision (no even-in-y_c discretization
artifact, unlike the unstructured `write_channel_geo`). The middle strip carries
probe_A (top) / probe_B (bottom); the outer strips' top/bottom are `walls`.
`h` is the target cell size. Boundary names match `Simulate`'s BC keys.
"""
function write_channel_geo_symmetric(path::AbstractString, cfg::ChannelConfig;
                                     h::Real=cfg.lc)
    L = cfg.L_x; W2 = cfg.W/2
    xPL = cfg.x_probe - cfg.L_probe/2
    xPR = cfg.x_probe + cfg.L_probe/2
    npts(len) = max(2, round(Int, len/h) + 1)
    Nx1 = npts(xPL - 0.0); Nx2 = npts(xPR - xPL); Nx3 = npts(L - xPR)
    Ny  = max(3, round(Int, cfg.W/h) + 1)

    open(path, "w") do io
        @printf(io, "lc = %.6f;\n", h)
        pts = [(0.0,-W2),(xPL,-W2),(xPR,-W2),(L,-W2),
               (L,W2),(xPR,W2),(xPL,W2),(0.0,W2)]
        for (i,(x,y)) in enumerate(pts)
            @printf(io, "Point(%d) = {%.10f, %.10f, 0, lc};\n", i, x, y)
        end
        # bottom (left→right), top (left→right: 8→7→6→5), verticals (bottom→top)
        println(io, "Line(1) = {1,2};\nLine(2) = {2,3};\nLine(3) = {3,4};")
        println(io, "Line(4) = {8,7};\nLine(5) = {7,6};\nLine(6) = {6,5};")
        println(io, "Line(7) = {1,8};\nLine(8) = {2,7};\nLine(9) = {3,6};\nLine(10) = {4,5};")
        println(io, "Curve Loop(1) = {1, 8, -4, -7};  Plane Surface(1) = {1};")
        println(io, "Curve Loop(2) = {2, 9, -5, -8};  Plane Surface(2) = {2};")
        println(io, "Curve Loop(3) = {3, 10, -6, -9}; Plane Surface(3) = {3};")
        @printf(io, "Transfinite Curve{1, 4} = %d;\n", Nx1)
        @printf(io, "Transfinite Curve{2, 5} = %d;\n", Nx2)
        @printf(io, "Transfinite Curve{3, 6} = %d;\n", Nx3)
        @printf(io, "Transfinite Curve{7, 8, 9, 10} = %d;\n", Ny)
        println(io, "Transfinite Surface{1}; Transfinite Surface{2}; Transfinite Surface{3};")
        println(io, "Recombine Surface{1, 2, 3};")
        println(io, "Physical Surface(\"domain\")        = {1, 2, 3};")
        println(io, "Physical Curve(\"contact_source\") = {7};")
        println(io, "Physical Curve(\"contact_drain\")  = {10};")
        println(io, "Physical Curve(\"probe_A\")        = {5};")
        println(io, "Physical Curve(\"probe_B\")        = {2};")
        println(io, "Physical Curve(\"walls\")          = {1, 3, 4, 6};")
    end
    return path
end

# ---------------------------------------------------------------------------
# Reusable evaluator: build mesh + semi ONCE, solve per density field
# ---------------------------------------------------------------------------

"""
    FixedEvaluator(inp_path, field, sim)

Build the semidiscretization once for the fixed mesh `inp_path`, wiring in the
base collision operator and a `BrinkmanSource` over `field`. Call
`run_f1_fixed(ev)` after mutating `field.rho` to (re)solve to steady state and
read `f_1`. Reusing the same evaluator across candidates skips mesh loading and
boundary-projector setup — only the ODE solve reruns.
"""
mutable struct FixedEvaluator
    semi::Any
    field::DensityField
    sim::SimConfig
    bc_probe_A::Any
    bc_probe_B::Any
    equations::Any
end

function FixedEvaluator(inp_path::AbstractString, field::DensityField,
                        sim::SimConfig=SimConfig(); damp_a0::Bool=false,
                        drain_bc=FermiSea.OhmicContactBC(0.0))
    equations = FermiSea.IsotropicFermiHarmonics2D(sim.n_harmonics;
                                                   v_fermi=sim.v_fermi)

    mesh = Trixi.P4estMesh{2}(inp_path; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source, :contact_drain,
                          :probe_A, :probe_B, :walls])

    bc_probe_A = FermiSea.FloatingProbeBC()
    bc_probe_B = FermiSea.FloatingProbeBC()
    boundary_conditions = (
        contact_source = FermiSea.CurrentContactBC(sim.I_source),
        contact_drain  = drain_bc,   # OhmicContactBC(0) default; CurrentContactBC(-I) for an injector pair
        probe_A        = bc_probe_A,
        probe_B        = bc_probe_B,
        walls          = FermiSea.MaxwellWallBC(1.0),
    )

    collision = FermiSea.LinearCollisionMatrix(equations;
        gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    source_terms = FermiSea.SourceTerms(collision, BrinkmanSource(field; damp_a0=damp_a0))

    initial_condition_zero(x, t, eq) =
        zero(Trixi.SVector{Trixi.nvariables(eq), Float64})

    solver = DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)
    semi = SemidiscretizationHyperbolic(mesh, equations,
        initial_condition_zero, solver;
        boundary_conditions=boundary_conditions,
        source_terms=source_terms)

    return FixedEvaluator(semi, field, sim, bc_probe_A, bc_probe_B, equations)
end

"""
    run_f1_fixed(ev; steady_abstol=1e-3, t_end=ev.sim.t_end,
                 solver_abstol=1e-5, solver_reltol=1e-5) -> (f1, info)

Solve the fixed-mesh problem to steady state for the current `ev.field.rho` and
return `f1 = (V_A - V_B) / I` plus diagnostics. Mutate `ev.field.rho` (e.g. via
`paint_blob!`) between calls to evaluate different obstacles.

`steady_abstol` is the `SteadyStateCallback` tolerance on `max|du|`; the tutorial
uses 1e-2, we default a bit tighter. `converged` in `info` is true when the
steady-state callback stopped the solve before `t_end` (otherwise we hit the
time cap and the value should be treated with suspicion). `residual` is the
final `‖du‖ / max(‖u‖, 1)`.
"""
function run_f1_fixed(ev::FixedEvaluator; steady_abstol::Real=1e-3,
                      t_end::Real=ev.sim.t_end,
                      solver_abstol::Real=1e-5, solver_reltol::Real=1e-5)
    t0 = time()
    sim = ev.sim
    semi = ev.semi

    ode = Trixi.semidiscretize(semi, (0.0, Float64(t_end)))
    steady_state = SteadyStateCallback(; abstol=steady_abstol, reltol=0.0)
    callbacks = CallbackSet(steady_state)

    sol = solve(ode, ROCK4();
        save_everystep=false, callback=callbacks,
        abstol=solver_abstol, reltol=solver_reltol)

    # Force a final RHS eval so the boundary caches (floating-probe potentials)
    # reflect the final state, then read the probe potentials.
    u_final = sol.u[end]
    du = similar(u_final)
    Trixi.rhs!(du, u_final, semi, sol.t[end])
    residual = norm(du) / max(norm(u_final), 1.0)

    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    V_A = FermiSea._current_contact_potential(ev.bc_probe_A, ev.equations, dg, cache)
    V_B = FermiSea._current_contact_potential(ev.bc_probe_B, ev.equations, dg, cache)
    f1 = (V_A - V_B) / sim.I_source

    info = (V_A=V_A, V_B=V_B, I_source=sim.I_source,
            walltime=time() - t0, steps=length(sol.t),
            t_final=sol.t[end], residual=residual,
            converged=(sol.t[end] < Float64(t_end) - 1e-9))
    return f1, info
end

"""
    run_f1_fixed_steady(ev; itmax=4000, memory=50, atol=1e-8, rtol=1e-7) -> (f1, info)

Solve the fixed-mesh steady state **directly** as a linear system instead of
time-stepping. The model is linear, so the steady state satisfies `rhs(u) = 0`,
i.e. `A u = -b` with `A v := rhs(v) - rhs(0)` (matrix-free via `rhs!`) and
`b := rhs(0)` (the pure boundary forcing; the Brinkman/collision sources vanish
at `u = 0`). We solve with restarted GMRES.

This is 50–100× faster than time-stepping for this stiff, slow-diffusive
regime. **Note on accuracy:** `f_1` is a delicate boundary quantity and only
settles once the linear residual is small (≲1e-5); the defaults target that.
`converged` reports whether GMRES hit its tolerance; even when it stops on
`itmax`, `residual` is the true steady residual to trust.
"""
function run_f1_fixed_steady(ev::FixedEvaluator; itmax::Int=4000, memory::Int=50,
                             atol::Real=1e-8, rtol::Real=1e-7)
    t0 = time()
    semi = ev.semi
    sim = ev.sim

    ode = Trixi.semidiscretize(semi, (0.0, 1.0))
    u0 = zero(ode.u0)
    N = length(u0)

    b = similar(u0)
    Trixi.rhs!(b, u0, semi, 0.0)        # affine forcing b = rhs(0)

    scratch = similar(u0)
    function matvec!(out, v)
        Trixi.rhs!(scratch, v, semi, 0.0)
        @. out = scratch - b
        return out
    end
    A = LinearMap{Float64}((out, v) -> matvec!(out, v), N; ismutating=true)

    u, stats = Krylov.gmres(A, -b; restart=true, memory=memory,
                            atol=Float64(atol), rtol=Float64(rtol), itmax=itmax)

    # Final rhs! syncs boundary caches (floating-probe potentials).
    du = similar(u)
    Trixi.rhs!(du, u, semi, 0.0)
    residual = norm(du) / max(norm(u), 1.0)

    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    V_A = FermiSea._current_contact_potential(ev.bc_probe_A, ev.equations, dg, cache)
    V_B = FermiSea._current_contact_potential(ev.bc_probe_B, ev.equations, dg, cache)
    f1 = (V_A - V_B) / sim.I_source

    info = (V_A=V_A, V_B=V_B, I_source=sim.I_source,
            walltime=time() - t0, iters=stats.niter,
            residual=residual, converged=stats.solved)
    return f1, info
end

# ---------------------------------------------------------------------------
# Point-contact vicinity geometry (narrow source/drain instead of full edges)
# ---------------------------------------------------------------------------

"""
    write_point_geo(path; L, W, xPL, xPR, wc, h)

Emit a `.geo` for the vicinity geometry with **narrow point contacts**: a 3×3
grid of transfinite blocks with a y-symmetric partition (so the mesh is exactly
mirror-symmetric). Source/drain are width-`wc` contacts centered on the
left/right walls; probe_A/probe_B are centered on the top/bottom walls; the rest
is `walls`. Full-edge contacts give ~0 vicinity signal (uniform flow); point
contacts create the spreading 2-D flow that carries a real signal.
"""
function write_point_geo(path::AbstractString; L=1.0, W=0.6, xPL=0.425, xPR=0.575,
                         wc=0.12, h=0.03)
    xs = (0.0, xPL, xPR, L)
    ys = (-W/2, -wc/2, wc/2, W/2)
    Nx = ntuple(i -> max(2, round(Int, (xs[i+1]-xs[i])/h)+1), 3)
    Ny = ntuple(j -> max(2, round(Int, (ys[j+1]-ys[j])/h)+1), 3)
    pid(i,j) = (i-1)*4 + j
    H(i,j) = (j-1)*3 + i          # horizontal line P(i,j)->P(i+1,j)
    V(i,j) = 12 + (i-1)*3 + j     # vertical line   P(i,j)->P(i,j+1)
    open(path, "w") do io
        @printf(io, "lc=%.6f;\n", h)
        for i in 1:4, j in 1:4
            @printf(io, "Point(%d)={%.10f,%.10f,0,lc};\n", pid(i,j), xs[i], ys[j])
        end
        for j in 1:4, i in 1:3
            @printf(io, "Line(%d)={%d,%d};\n", H(i,j), pid(i,j), pid(i+1,j))
        end
        for i in 1:4, j in 1:3
            @printf(io, "Line(%d)={%d,%d};\n", V(i,j), pid(i,j), pid(i,j+1))
        end
        sid = 0
        for i in 1:3, j in 1:3
            sid += 1
            @printf(io, "Curve Loop(%d)={%d,%d,%d,%d};\n", sid,
                    H(i,j), V(i+1,j), -H(i,j+1), -V(i,j))
            @printf(io, "Plane Surface(%d)={%d};\n", sid, sid)
        end
        for i in 1:3, j in 1:4; @printf(io, "Transfinite Curve{%d}=%d;\n", H(i,j), Nx[i]); end
        for i in 1:4, j in 1:3; @printf(io, "Transfinite Curve{%d}=%d;\n", V(i,j), Ny[j]); end
        for s in 1:9; @printf(io, "Transfinite Surface{%d};\n", s); end
        print(io, "Recombine Surface{"); print(io, join(1:9, ",")); println(io, "};")
        println(io, "Physical Surface(\"domain\")={", join(1:9, ","), "};")
        @printf(io, "Physical Curve(\"contact_source\")={%d};\n", V(1,2))
        @printf(io, "Physical Curve(\"contact_drain\") ={%d};\n", V(4,2))
        @printf(io, "Physical Curve(\"probe_A\")={%d};\n", H(2,4))
        @printf(io, "Physical Curve(\"probe_B\")={%d};\n", H(2,1))
        walls = [H(1,1),H(3,1),H(1,4),H(3,4),V(1,1),V(1,3),V(4,1),V(4,3)]
        println(io, "Physical Curve(\"walls\")={", join(walls, ","), "};")
    end
    return path
end

"""
    write_vicinity_geo(path; L, W, wc, h, xs, dA, dB)

Emit a `.geo` for a literature-standard **Bandurin–Levitov vicinity device** (no
obstacle): a rectangular sample with all contacts as narrow segments on the
**bottom edge** — an adjacent current **injector pair** (`contact_source` at
`[xs, xs+wc]`, `contact_drain` at `[xs+wc, xs+2wc]`) and two floating voltage
probes on the vicinity side (`probe_A` near, gap `dA`; `probe_B` far, gap `dB`).
Built as a 1-row strip of transfinite blocks (columns split at each contact
boundary, shared uniform y-partition) → structured all-quad mesh, no hole; mesh
with `geo_to_inp(...; structured=true)`. The vicinity resistance is
`f_1 = (V_A - V_B)/I`, which turns negative in the hydrodynamic (viscous) regime.
"""
function write_vicinity_geo(path::AbstractString; L=2.0, W=0.8, wc=0.1, h=0.05,
                            xs=1.25, dA=0.15, dB=0.45, xd=nothing)
    # xd=nothing → adjacent injector pair (local dipole); xd set → drain far away
    # (canonical Bandurin nonlocal layout: probes sit in the current-free vicinity).
    drain = xd === nothing ? (xs + wc, xs + 2wc) : (xd, xd + wc)
    contacts = [(xs,          xs + wc,      "contact_source"),
                (drain[1],    drain[2],     "contact_drain"),
                (xs - dA - wc, xs - dA,     "probe_A"),
                (xs - dB - wc, xs - dB,     "probe_B")]
    sort!(contacts, by=first)
    # fill [0,L] with alternating wall / contact segments (each = one column)
    segs = Tuple{Float64,Float64,String}[]; cur = 0.0
    for (a,b,name) in contacts
        a > cur + 1e-9 && push!(segs, (cur, a, "walls"))
        push!(segs, (a, b, name)); cur = b
    end
    cur < L - 1e-9 && push!(segs, (cur, L, "walls"))
    Ncol = length(segs)
    xsplit = vcat(segs[1][1], [s[2] for s in segs])      # length Ncol+1
    Ny = max(3, round(Int, W/h) + 1)
    open(path, "w") do io
        @printf(io, "lc=%.6f;\n", h)
        for i in 1:Ncol+1
            @printf(io, "Point(%d)={%.8f,0,0,lc};\n", i, xsplit[i])
            @printf(io, "Point(%d)={%.8f,%.8f,0,lc};\n", 100+i, xsplit[i], W)
        end
        for i in 1:Ncol;   @printf(io, "Line(%d)={%d,%d};\n", i, i, i+1); end          # bottom BL_i
        for i in 1:Ncol;   @printf(io, "Line(%d)={%d,%d};\n", 200+i, 100+i, 100+i+1); end # top TL_i
        for i in 1:Ncol+1; @printf(io, "Line(%d)={%d,%d};\n", 300+i, i, 100+i); end    # vert VL_i
        for i in 1:Ncol
            @printf(io, "Curve Loop(%d)={%d,%d,%d,%d};\n", i, i, 300+i+1, -(200+i), -(300+i))
            @printf(io, "Plane Surface(%d)={%d};\n", i, i)
        end
        for i in 1:Ncol
            nx = max(2, round(Int, (xsplit[i+1]-xsplit[i])/h) + 1)
            @printf(io, "Transfinite Curve{%d,%d}=%d;\n", i, 200+i, nx)
        end
        for i in 1:Ncol+1; @printf(io, "Transfinite Curve{%d}=%d;\n", 300+i, Ny); end
        for i in 1:Ncol;   @printf(io, "Transfinite Surface{%d};\n", i); end
        print(io, "Recombine Surface{"); print(io, join(1:Ncol, ",")); println(io, "};")
        println(io, "Physical Surface(\"domain\")={", join(1:Ncol, ","), "};")
        for name in ("contact_source","contact_drain","probe_A","probe_B")
            ids = [i for i in 1:Ncol if segs[i][3] == name]
            @printf(io, "Physical Curve(\"%s\")={%s};\n", name, join(ids, ","))
        end
        wallids = vcat([i for i in 1:Ncol if segs[i][3] == "walls"],   # wall bottom segs
                       [200+i for i in 1:Ncol],                          # all top segs
                       [301, 300+Ncol+1])                                # left/right ends
        println(io, "Physical Curve(\"walls\")={", join(wallids, ","), "};")
    end
    return path
end

# ---------------------------------------------------------------------------
# Exact assembled operator: reg-LU forward solve + discrete-adjoint gradient
# ---------------------------------------------------------------------------

"""
    FixedOperator

Assembled sparse operator for exact (regularized-LU) steady solves and the
discrete-adjoint gradient of `f_1` w.r.t. the density design grid. Fields:
`A0` (streaming+collision+boundary Jacobian at ρ=0), `c` (the `f_1 = cᵀu`
functional), `b1u` (Brinkman diagonal per unit density, α=1), `bvec` (`rhs(0)`
forcing), `cellof` (dof → design-cell index for gradient scatter). Built by
[`assemble_fixed_operator`](@ref).
"""
struct FixedOperator
    A0::SparseMatrixCSC{Float64,Int}
    c::Vector{Float64}
    b1u::Vector{Float64}
    bvec::Vector{Float64}
    cellof::Vector{Int}
    ev::FixedEvaluator
    nx::Int
    ny::Int
    N::Int
end

_probe_f1(ev, dg, cache) =
    (FermiSea._current_contact_potential(ev.bc_probe_A, ev.equations, dg, cache) -
     FermiSea._current_contact_potential(ev.bc_probe_B, ev.equations, dg, cache)) /
    ev.sim.I_source

"""
    assemble_fixed_operator(ev; tol=1e-12) -> FixedOperator

Assemble the sparse operator once by probing `rhs!` with unit vectors (also
recovering the `f_1` functional `c` and the Brinkman unit-diagonal `b1u`). The
operator is ρ-independent; each density then costs one sparse LU. Because the
undamped density mode makes the bare steady state singular, solves are
regularized (min-norm); see [`f1_exact`](@ref).
"""
function assemble_fixed_operator(ev::FixedEvaluator; tol::Real=1e-12)
    semi = ev.semi; field = ev.field
    _, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    nvars = Trixi.nvariables(equations)

    clear!(field)
    ode = Trixi.semidiscretize(semi, (0.0, 1.0))
    N = length(ode.u0)
    bvec = similar(ode.u0); Trixi.rhs!(bvec, zero(ode.u0), semi, 0.0)

    Ir = Int[]; Jc = Int[]; Vv = Float64[]; c = zeros(N)
    tmp = similar(bvec); ej = zeros(N)
    for j in 1:N
        ej[j] = 1.0
        Trixi.rhs!(tmp, ej, semi, 0.0)
        c[j] = _probe_f1(ev, dg, cache)
        ej[j] = 0.0
        @inbounds for i in 1:N
            v = tmp[i] - bvec[i]
            abs(v) > tol && (push!(Ir, i); push!(Jc, j); push!(Vv, v))
        end
    end
    A0 = sparse(Ir, Jc, Vv, N, N)

    # Brinkman unit diagonal at α=1: b1u = rhs_{ρ=1,α=1}(𝟙) − rhs_{ρ=0}(𝟙)
    saved_alpha = field.alpha_max; field.alpha_max = 1.0
    onev = ones(N); fill!(field.rho, 1.0)
    r1 = similar(bvec); Trixi.rhs!(r1, onev, semi, 0.0)
    clear!(field); r0 = similar(bvec); Trixi.rhs!(r0, onev, semi, 0.0)
    b1u = r1 .- r0
    field.alpha_max = saved_alpha

    # dof -> design-cell (same floor lookup as penalization)
    cellof = zeros(Int, N)
    tf = zeros(N); w = Trixi.wrap_array(tf, semi)
    nc = cache.elements.node_coordinates; nn = size(w, 2); nel = size(w, 4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x = nc[1, ii, jj, e]; y = nc[2, ii, jj, e]
        ci = clamp(floor(Int, (x - field.x0)/field.dx) + 1, 1, field.nx)
        cj = clamp(floor(Int, (y - field.y0)/field.dy) + 1, 1, field.ny)
        lin = ci + (cj - 1) * field.nx
        for v in 1:nvars; w[v, ii, jj, e] = lin; end
    end
    cellof .= round.(Int, tf)

    return FixedOperator(A0, c, b1u, bvec, cellof, ev, field.nx, field.ny, N)
end

# Fill a per-dof density vector from the design grid (same lookup as penalization).
function _rho_to_perdof!(rpd::Vector{Float64}, op::FixedOperator, rho_grid::AbstractMatrix)
    ev = op.ev; field = ev.field; semi = ev.semi
    _, equations, _, cache = Trixi.mesh_equations_solver_cache(semi)
    nvars = Trixi.nvariables(equations)
    w = Trixi.wrap_array(rpd, semi); nc = cache.elements.node_coordinates
    nn = size(w, 2); nel = size(w, 4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x = nc[1, ii, jj, e]; y = nc[2, ii, jj, e]
        ci = clamp(floor(Int, (x - field.x0)/field.dx) + 1, 1, field.nx)
        cj = clamp(floor(Int, (y - field.y0)/field.dy) + 1, 1, field.ny)
        for v in 1:nvars; w[v, ii, jj, e] = rho_grid[ci, cj]; end
    end
    return rpd
end

"""
    f1_exact(op, rho_grid; alpha, eps=1e-10) -> f1

Exact `f_1` for the density `rho_grid` via a Tikhonov-regularized sparse LU:
`(A0 + diag(α·ρ·b1u) + εI) u = -b`, `f_1 = cᵀu`. The `εI` selects the
minimum-norm solution (the undamped density mode makes the bare system
singular). `f_1` is stable as `ε→0`; the default is deep in that plateau.
"""
function f1_exact(op::FixedOperator, rho_grid::AbstractMatrix;
                  alpha::Real, eps::Real=1e-10)
    rpd = zeros(op.N); _rho_to_perdof!(rpd, op, rho_grid)
    clamp!(rpd, 0.0, 1.0)   # ρ<0 would flip damping to gain (indefinite A)
    A = op.A0 + spdiagm(0 => (alpha .* rpd) .* op.b1u) + eps * spdiagm(0 => ones(op.N))
    return dot(op.c, lu(A) \ (-op.bvec))
end

"""
    f1_adjoint_grad(op, rho_grid; alpha, eps=1e-10) -> (f1, grad)

`f_1` and its gradient w.r.t. every design cell, from one forward + one adjoint
solve (the problem is linear): `∂f_1/∂ρ_cell = -Σ_{j∈cell} λ_j·(α·b1u_j)·u_j`
with `Aᵀλ = c`. `grad` is an `op.nx × op.ny` matrix. Verified against finite
differences to ~1e-6.
"""
function f1_adjoint_grad(op::FixedOperator, rho_grid::AbstractMatrix;
                         alpha::Real, eps::Real=1e-10)
    rpd = zeros(op.N); _rho_to_perdof!(rpd, op, rho_grid)
    A = op.A0 + spdiagm(0 => (alpha .* rpd) .* op.b1u) + eps * spdiagm(0 => ones(op.N))
    F = lu(A)
    u = F \ (-op.bvec)
    f1 = dot(op.c, u)
    λ = F' \ op.c
    g = zeros(op.nx * op.ny)
    @inbounds for j in 1:op.N
        gj = -λ[j] * (alpha * op.b1u[j]) * u[j]
        gj != 0.0 && (g[op.cellof[j]] += gj)
    end
    return f1, reshape(g, op.nx, op.ny)
end

"""
    f1_exact_perdof(op, rpd; alpha, eps=1e-10) -> f1

Like [`f1_exact`](@ref) but takes a per-dof density vector directly (e.g. from
[`fourier_perdof`](@ref)), bypassing the design-grid lookup.
"""
function f1_exact_perdof(op::FixedOperator, rpd::AbstractVector;
                         alpha::Real, eps::Real=1e-10)
    rpd = clamp.(rpd, 0.0, 1.0)   # ρ<0 would flip damping to gain (indefinite A)
    A = op.A0 + spdiagm(0 => (alpha .* rpd) .* op.b1u) + eps * spdiagm(0 => ones(op.N))
    return dot(op.c, lu(A) \ (-op.bvec))
end

"""
    fourier_perdof(op; r0, yc=0.0, C=Float64[], xc=0.5, width=0.02) -> Vector

Per-dof density of a Fourier obstacle
`R(θ) = r0 + Σ_n (C[2n-1] cos nθ + C[2n] sin nθ)` (center `(xc, yc)`), evaluated
**analytically at each mesh node** — no design-grid staircase. Feed to
[`f1_exact_perdof`](@ref) for tight mesh convergence.
"""
function fourier_perdof(op::FixedOperator; r0::Real, yc::Real=0.0,
                        C::AbstractVector=Float64[], xc::Real=0.5, width::Real=0.02)
    rpd = zeros(op.N); ev = op.ev; semi = ev.semi
    _, equations, _, cache = Trixi.mesh_equations_solver_cache(semi)
    nvars = Trixi.nvariables(equations)
    w = Trixi.wrap_array(rpd, semi); nc = cache.elements.node_coordinates
    nn = size(w, 2); nel = size(w, 4); M = length(C) ÷ 2
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x = nc[1, ii, jj, e]; y = nc[2, ii, jj, e]
        dx = x - xc; dy = y - yc; d = hypot(dx, dy); th = atan(dy, dx); R = float(r0)
        @inbounds for n in 1:M
            R += C[2n-1] * cos(n*th) + C[2n] * sin(n*th)
        end
        val = 1.0 / (1.0 + exp((d - R) / width))
        for v in 1:nvars; w[v, ii, jj, e] = val; end
    end
    return rpd
end

end # module
