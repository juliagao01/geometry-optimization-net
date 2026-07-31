#!/usr/bin/env julia
# scripts/fixed_mesh_opt.jl
#
#   julia --threads=auto --project=. scripts/fixed_mesh_opt.jl
#
# Density-based ("fixed mesh") optimization of the obstacle to maximize
# f_1 = (V_A - V_B) / I, evaluated with a DIRECT matrix-free GMRES steady-state
# solve (run_f1_fixed_steady) — ~15-60 s/solve vs the ~1000 s an explicit
# time-stepper needs in this stiff, slow-diffusive regime.
#
# The quad mesh is not perfectly mirror-symmetric, so a raw f_1 carries an
# even-in-y_c discretization artifact that is signal-sized (a centered blob
# spuriously reads f_1 ~ 0.016 instead of 0). Since the true f_1 is ODD in y_c,
# we optimize the ANTISYMMETRIZED objective
#     f1_odd(y_c, r) = [ f_1(+y_c, r) - f_1(-y_c, r) ] / 2
# which cancels the even artifact exactly. Costs two solves per point.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.06)
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05, gamma_3=300.0, polydeg=1)

workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel.geo"); inp = joinpath(workdir, "channel.inp")
write_channel_geo(geo, cfg); VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false)

field = DensityField(cfg; nx=30, ny=18, alpha_max=100.0)
ev = FixedEvaluator(inp, field, sim)
xc = cfg.x_c
wall = cfg.W/2 - 0.02

raw = Dict{Tuple{Float64,Float64},NamedTuple}()
function raw_f1(yc, r)
    key = (round(yc; digits=4), round(r; digits=4))
    haskey(raw, key) && return raw[key].f1
    paint_blob!(field, xc, yc, r)
    f1, info = run_f1_fixed_steady(ev; itmax=12000, memory=80, atol=1e-9, rtol=1e-8)
    raw[key] = (f1=f1, res=info.residual, iters=info.iters, wt=info.walltime)
    @printf("    [%+.3f,%.3f] f_1=%+.5e res=%.1e it=%d %.0fs\n",
            yc, r, f1, info.residual, info.iters, info.walltime); flush(stdout)
    return f1
end

odds = NamedTuple[]
function odd_f1(yc, r)
    fp = raw_f1(+yc, r); fm = raw_f1(-yc, r)
    o = (fp - fm)/2; e = (fp + fm)/2
    push!(odds, (yc=yc, r=r, odd=o, even=e))
    @printf("  |yc|=%.3f r=%.3f -> f1_odd=%+.5e  (even artifact=%+.2e)\n",
            yc, r, o, e); flush(stdout)
    return o
end

println("=== fixed-mesh density optimization (antisymmetrized objective) ===")
println("mesh lc=$(cfg.lc)  grid $(field.nx)x$(field.ny)  alpha_max=$(field.alpha_max)")
flush(stdout)

# Phase 1: pick obstacle size r at the near-optimal offset y_c=0.08.
println("\n--- phase 1: size sweep at y_c=0.08 ---"); flush(stdout)
for r in (0.10, 0.14, 0.17, 0.20)
    0.08 + r <= wall + 1e-9 || continue
    odd_f1(0.08, r)
end
best_r = odds[argmax([x.odd for x in odds])].r
@printf("best r ≈ %.3f\n", best_r); flush(stdout)

# Phase 2: offset sweep at the best size.
println("\n--- phase 2: offset sweep at r=$(best_r) ---"); flush(stdout)
for yc in (0.04, 0.06, 0.10, 0.12)
    yc + best_r <= wall + 1e-9 || continue
    odd_f1(yc, best_r)
end

best = odds[argmax([x.odd for x in odds])]
println("\n=== RESULT (physical, antisymmetrized) ===")
@printf("BEST obstacle: y_c=%+.3f  r=%.3f  ->  f_1 = %+.6e\n", best.yc, best.r, best.odd)
@printf("(even-part artifact at this point = %+.2e; %d solves)\n", best.even, length(raw))
flush(stdout)

paint_blob!(field, xc, best.yc, best.r)
best_yc = best.yc; best_r2 = best.r; best_f1 = best.odd
@save joinpath(workdir, "result_steady.jld2") best_yc best_r2 best_f1 odds raw rho=copy(field.rho) cfg sim
println("wrote ", joinpath(workdir, "result_steady.jld2"))

println("\n--- all antisymmetrized points (sorted) ---")
for x in sort(odds; by=z->-z.odd)
    @printf("  yc=%.3f r=%.3f  f1_odd=%+.5e  even=%+.2e\n", x.yc, x.r, x.odd, x.even)
end
println("DONE"); flush(stdout)
