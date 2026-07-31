#!/usr/bin/env julia
# scripts/fixed_mesh_converge.jl — is the vicinity signal REAL or truncation noise?
#   julia --threads=auto --project=. scripts/fixed_mesh_converge.jl
#
# On the symmetric mesh the true f_1 came out ~1e-7 and dropped 10x from
# n_harmonics 4->6. Before optimizing it we must know: does f_1 converge to a
# nonzero value as n_harmonics grows, or ->0 (i.e. the measurement geometry
# gives essentially no signal)? Also check obstacle solidity (alpha_max).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.03)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel_sym.geo"); inp = joinpath(workdir, "channel_sym.inp")
write_channel_geo_symmetric(geo, cfg; h=cfg.lc)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)

yc, r = 0.08, 0.16
run1(sim, alpha) = begin
    field = DensityField(cfg; nx=40, ny=24, alpha_max=alpha)
    ev = FixedEvaluator(inp, field, sim)
    paint_blob!(field, cfg.x_c, yc, r)
    f1, info = run_f1_fixed_steady(ev; itmax=25000, memory=100, atol=1e-11, rtol=1e-10)
    f1, info
end

println("=== n_harmonics convergence (gamma_mc=100, alpha=100) ==="); flush(stdout)
for M in (2, 4, 6, 8, 10, 12)
    sim = SimConfig(n_harmonics=M, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    f1, info = run1(sim, 100.0)
    @printf("  M=%2d  f_1=%+.5e  (res=%.1e it=%d %.0fs)\n",
            M, f1, info.residual, info.iters, info.walltime); flush(stdout)
end

println("\n=== obstacle solidity alpha_max (M=6, gamma_mc=100) ==="); flush(stdout)
for a in (50.0, 100.0, 500.0, 2000.0, 10000.0)
    sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    f1, info = run1(sim, a)
    @printf("  alpha=%7.0f  f_1=%+.5e  (res=%.1e it=%d %.0fs)\n",
            a, f1, info.residual, info.iters, info.walltime); flush(stdout)
end

println("\n=== strong asymmetry: big obstacle near top wall (M=6, a=2000) ==="); flush(stdout)
let sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    field = DensityField(cfg; nx=40, ny=24, alpha_max=2000.0)
    ev = FixedEvaluator(inp, field, sim)
    for (yc2,r2) in ((0.14,0.14),(0.18,0.10),(0.10,0.18))
        paint_blob!(field, cfg.x_c, yc2, r2)
        f1, info = run_f1_fixed_steady(ev; itmax=25000, memory=100, atol=1e-11, rtol=1e-10)
        @printf("  yc=%.2f r=%.2f  f_1=%+.5e  (res=%.1e)\n", yc2, r2, f1, info.residual); flush(stdout)
    end
end
println("DONE"); flush(stdout)
