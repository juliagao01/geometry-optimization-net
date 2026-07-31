#!/usr/bin/env julia
# scripts/fixed_mesh_investigate.jl — revisions #1/#2/#3 via the src/ API.
#  #1 hard-wall: does f_1 converge in α if the obstacle also damps the density
#     mode (absorbing, no floating interior)? vs the momentum-only f_1 ∝ α.
#  #2 contact width toward the full-edge limit (does the signal collapse?).
#  #3 mesh convergence with the ANALYTIC Fourier density (no staircase grid).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
d = load(joinpath(workdir,"result_circle.jld2"))
Cbest = d["C"]; R0 = d["R0"]
@printf("obstacle: R0=%.3f  C=%s\n", R0, string(round.(Cbest,digits=4))); flush(stdout)

function build(; wc=0.12, h=0.045, damp_a0=false)
    geo=joinpath(workdir,"inv.geo"); inp=joinpath(workdir,"inv.inp")
    write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                    xPR=cfg.x_probe+cfg.L_probe/2, wc=wc, h=h)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfg; nx=64, ny=40, alpha_max=1.0)
    ev = FixedEvaluator(inp, field, sim; damp_a0=damp_a0)
    assemble_fixed_operator(ev)
end
circle(op) = fourier_perdof(op; r0=R0, C=Cbest, xc=cfg.x_c, width=0.02)

println("\n=== #1 hard-wall: f_1 vs α  (reflecting = momentum-only vs absorbing = +a0) ===")
op_r = build(; damp_a0=false); rp_r = circle(op_r)
op_a = build(; damp_a0=true);  rp_a = circle(op_a)
@printf("  %-10s %-16s %-16s\n","α","reflecting","absorbing(+a0)")
for a in (500.0, 2e3, 1e4, 5e4, 2e5, 1e6)
    @printf("  %-10.0f %+16.5e %+16.5e\n", a,
            f1_exact_perdof(op_r, rp_r; alpha=a),
            f1_exact_perdof(op_a, rp_a; alpha=a)); flush(stdout)
end

println("\n=== #2 contact width wc -> full-edge (α=2000) ===")
for wc in (0.12, 0.24, 0.36, 0.48, 0.54)
    op = build(; wc=wc)
    @printf("  wc=%.2f (%.0f%% of wall)  f_1=%+.5e\n", wc, 100wc/cfg.W,
            f1_exact_perdof(op, circle(op); alpha=2000.0)); flush(stdout)
end

println("\n=== #3 mesh convergence, ANALYTIC density (α=2000) ===")
for h in (0.055, 0.045, 0.038, 0.032, 0.028)
    t=time(); op=build(; h=h)
    @printf("  h=%.3f N=%6d  f_1=%+.6e  (%.0fs)\n", h, op.N,
            f1_exact_perdof(op, circle(op); alpha=2000.0), time()-t); flush(stdout)
    GC.gc()
end
println("DONE"); flush(stdout)
