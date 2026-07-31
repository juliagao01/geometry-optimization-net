#!/usr/bin/env julia
# scripts/fixed_mesh_diag.jl  — timing/convergence diagnostic (compile once)
#   julia --threads=auto --project=. scripts/fixed_mesh_diag.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.06)
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05, gamma_3=300.0,
                polydeg=1)

workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel.geo"); inp = joinpath(workdir, "channel.inp")
write_channel_geo(geo, cfg)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false)
println("meshed empty channel"); flush(stdout)

field = DensityField(cfg; nx=24, ny=16, alpha_max=100.0)
ev = FixedEvaluator(inp, field, sim)
println("evaluator built"); flush(stdout)

function trial(label; kw...)
    f1, info = run_f1_fixed(ev; kw...)
    @printf("%-30s f_1=%+.4e  t=%.1f steps=%d resid=%.1e conv=%s  %.1fs\n",
            label, f1, info.t_final, info.steps, info.residual,
            info.converged, info.walltime)
    flush(stdout)
    return f1, info
end

println("\n--- warmup (compile solve path), empty channel, tight cap ---")
clear!(field); trial("warmup empty t_end=30"; steady_abstol=1e-2, t_end=30.0)

println("\n--- empty channel: does steady trigger, how fast? ---")
clear!(field)
trial("empty tol=1e-2 t_end=200"; steady_abstol=1e-2, t_end=200.0)
trial("empty tol=1e-3 t_end=200"; steady_abstol=1e-3, t_end=200.0)

println("\n--- off-center blob alpha=100 ---")
paint_blob!(field, 0.5, 0.12, 0.12)
trial("blob+ tol=1e-2 t_end=200"; steady_abstol=1e-2, t_end=200.0)
trial("blob+ tol=1e-3 t_end=200"; steady_abstol=1e-3, t_end=200.0)

println("\n--- alpha sensitivity (blob up), tol=1e-2 t_end=200 ---")
for a in (50.0, 200.0, 500.0)
    field.alpha_max = a
    paint_blob!(field, 0.5, 0.12, 0.12)
    trial(@sprintf("blob+ alpha=%.0f", a); steady_abstol=1e-2, t_end=200.0)
end

println("\n--- symmetry check at alpha=100 tol=1e-2 ---")
field.alpha_max = 100.0
paint_blob!(field, 0.5, 0.0, 0.15);  trial("centered r=0.15")
paint_blob!(field, 0.5, +0.12, 0.12); trial("up   yc=+0.12")
paint_blob!(field, 0.5, -0.12, 0.12); trial("down yc=-0.12")
println("DONE"); flush(stdout)
