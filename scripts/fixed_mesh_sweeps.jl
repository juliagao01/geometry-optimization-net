#!/usr/bin/env julia
# scripts/fixed_mesh_sweeps.jl — physical parameter sweeps for the optimized shape,
# using the promoted src/ API (assemble_fixed_operator + f1_exact). Characterizes
# how the vicinity signal depends on obstacle solidity α, collision rate γ_mc,
# and contact width wc.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
bestphys = load(joinpath(workdir,"result_circle.jld2"))["bestphys"]
DNX,DNY = size(bestphys)

# build evaluator + assembled operator for a given (gamma_mc, wc, h)
function build_op(; gmc=100.0, wc=0.12, h=0.05)
    sim = SimConfig(n_harmonics=6, gamma_mc=gmc, gamma_mr=0.05, gamma_3=gmc, polydeg=1)
    geo=joinpath(workdir,"sw.geo"); inp=joinpath(workdir,"sw.inp")
    write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                    xPR=cfg.x_probe+cfg.L_probe/2, wc=wc, h=h)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfg; nx=DNX, ny=DNY, alpha_max=1.0)
    ev = FixedEvaluator(inp, field, sim)
    assemble_fixed_operator(ev)
end

println("=== (A) obstacle solidity α (γ_mc=100, wc=0.12) -> hard-wall limit ===")
opA = build_op()
for a in (200.0, 500.0, 1e3, 2e3, 5e3, 2e4, 1e5)
    @printf("  α=%8.0f  f_1=%+.5e\n", a, f1_exact(opA, bestphys; alpha=a)); flush(stdout)
end

println("\n=== (B) collision rate γ_mc (α=2000, wc=0.12): ballistic->hydrodynamic ===")
for gmc in (1.0, 10.0, 30.0, 100.0, 300.0)
    op = build_op(; gmc=gmc)
    @printf("  γ_mc=%6.1f  f_1=%+.5e\n", gmc, f1_exact(op, bestphys; alpha=2000.0)); flush(stdout)
end

println("\n=== (C) contact width wc (α=2000, γ_mc=100): point vs wide ===")
for wc in (0.06, 0.09, 0.12, 0.18, 0.30)
    op = build_op(; wc=wc)
    @printf("  wc=%.2f  f_1=%+.5e\n", wc, f1_exact(op, bestphys; alpha=2000.0)); flush(stdout)
end
println("DONE"); flush(stdout)
