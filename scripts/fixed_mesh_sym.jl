#!/usr/bin/env julia
# scripts/fixed_mesh_sym.jl — symmetric (transfinite) mesh: validate + optimize
#   julia --threads=auto --project=. scripts/fixed_mesh_sym.jl
#
# STEP 1 of the plan. Build the mirror-symmetric structured mesh and verify a
# centered obstacle gives f_1 ~ 0 (the artifact that forced antisymmetrization
# on the unstructured mesh). If so, f_1 is physical from a SINGLE solve, so we
# sweep (y_c, r) directly (half the cost) to reconfirm the optimum.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.03)
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05, gamma_3=300.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel_sym.geo"); inp = joinpath(workdir, "channel_sym.inp")
write_channel_geo_symmetric(geo, cfg; h=cfg.lc)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
println("built symmetric mesh -> ", inp); flush(stdout)

field = DensityField(cfg; nx=40, ny=24, alpha_max=100.0)
ev = FixedEvaluator(inp, field, sim)
xc = cfg.x_c
solveopts = (itmax=15000, memory=80, atol=1e-9, rtol=1e-8)

function ev1(yc, r)
    paint_blob!(field, xc, yc, r)
    f1, info = run_f1_fixed_steady(ev; solveopts...)
    @printf("  yc=%+.3f r=%.3f -> f_1=%+.5e  (res=%.1e it=%d %.0fs)\n",
            yc, r, f1, info.residual, info.iters, info.walltime); flush(stdout)
    return f1
end

println("\n=== SYMMETRY VALIDATION (centered must be ~0) ==="); flush(stdout)
clear!(field);              fe = ev1(0.0, 0.0)   # empty
fc = ev1(0.0, 0.16)                              # centered blob: was +0.031 before
fp = ev1(+0.08, 0.16)
fm = ev1(-0.08, 0.16)
@printf("empty=%.2e  centered=%.2e  (+0.08)=%+.4e (-0.08)=%+.4e  sum=%.2e\n",
        abs(fe), abs(fc), fp, fm, fp+fm); flush(stdout)
sym_ok = abs(fc) < 5e-4 && abs(fp+fm) < 5e-4
println(sym_ok ? "SYMMETRY OK — f_1 is physical from single solves" :
                 "SYMMETRY STILL BROKEN — investigate mesh"); flush(stdout)

println("\n=== single-solve (y_c, r) sweep ==="); flush(stdout)
results = NamedTuple[]
wall = cfg.W/2 - 0.02
for r in (0.12, 0.16, 0.20), yc in (0.04, 0.08, 0.12)
    yc + r <= wall + 1e-9 || continue
    f1 = ev1(yc, r)
    push!(results, (yc=yc, r=r, f1=f1))
end
best = results[argmax([x.f1 for x in results])]
println("\n=== RESULT (symmetric mesh, single-solve) ===")
@printf("BEST: y_c=%+.3f r=%.3f -> f_1=%+.6e\n", best.yc, best.r, best.f1); flush(stdout)
paint_blob!(field, xc, best.yc, best.r)
best_yc=best.yc; best_r=best.r; best_f1=best.f1
@save joinpath(workdir,"result_sym.jld2") best_yc best_r best_f1 results rho=copy(field.rho) cfg sim
println("wrote result_sym.jld2")
for x in sort(results; by=z->-z.f1)
    @printf("  yc=%.3f r=%.3f  f_1=%+.5e\n", x.yc, x.r, x.f1)
end
println("DONE"); flush(stdout)
