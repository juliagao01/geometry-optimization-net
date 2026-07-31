#!/usr/bin/env julia
# scripts/fixed_mesh_regime.jl — how big is the true vicinity signal vs regime?
#   julia --threads=auto --project=. scripts/fixed_mesh_regime.jl
#
# On the symmetric mesh the true f_1 for a fixed off-center obstacle is ~1e-6 at
# gamma_mc=300 (deep hydrodynamic). The README says the vicinity signal is
# strongest toward the ballistic limit (small gamma_mc). Quick check: same
# obstacle, sweep gamma_mc, see how f_1 scales. Informs where free-form
# optimization is actually worth running.
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
for gmc in (1.0, 10.0, 30.0, 100.0, 300.0)
    sim = SimConfig(n_harmonics=6, gamma_mc=gmc, gamma_mr=0.05, gamma_3=gmc, polydeg=1)
    field = DensityField(cfg; nx=40, ny=24, alpha_max=100.0)
    ev = FixedEvaluator(inp, field, sim)
    paint_blob!(field, cfg.x_c, yc, r)
    f1, info = run_f1_fixed_steady(ev; itmax=20000, memory=100, atol=1e-10, rtol=1e-9)
    @printf("gamma_mc=%6.1f  f_1=%+.5e  (res=%.1e it=%d %.0fs)\n",
            gmc, f1, info.residual, info.iters, info.walltime); flush(stdout)
end
println("DONE"); flush(stdout)
