#!/usr/bin/env julia
# scripts/fixed_mesh_cleanconv.jl — convergence battery on the CONTAMINATION-FREE
# (contact-excluded) optimized shape: does f_1 converge in α and h, and is it
# insensitive to the exclusion band? Decides whether ~14 is a defensible number.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
d = load(joinpath(workdir,"result_circle.jld2")); C=d["C"]; R0=d["R0"]
@printf("clean shape: R0=%.3f C=%s\n", R0, string(round.(C,digits=4))); flush(stdout)

function build(; wc=0.12, h=0.045)
    geo=joinpath(workdir,"cc.geo"); inp=joinpath(workdir,"cc.inp")
    write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                    xPR=cfg.x_probe+cfg.L_probe/2, wc=wc, h=h)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfg; nx=64, ny=40, alpha_max=1.0)
    assemble_fixed_operator(FixedEvaluator(inp, field, sim))
end
function ycoord(op)
    y=zeros(op.N); w=Trixi.wrap_array(y, op.ev.semi)
    _,eq,_,ca=Trixi.mesh_equations_solver_cache(op.ev.semi); nvars=Trixi.nvariables(eq)
    nc=ca.elements.node_coordinates; nn=size(w,2); nel=size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn, v in 1:nvars; w[v,ii,jj,e]=nc[2,ii,jj,e]; end
    y
end
function f1clean(op, yc, alpha, excl)
    r = fourier_perdof(op; r0=R0, C=C, xc=cfg.x_c, width=0.04)   # width ≈ h (resolved interface)
    r[abs.(yc) .> cfg.W/2 - excl] .= 0.0
    f1_exact_perdof(op, r; alpha=alpha)
end

op = build(h=0.045); yc = ycoord(op)
println("\n=== A) α-convergence (clean shape, h=0.045, EXCL=0.04) ===")
for a in (500.0, 2e3, 1e4, 5e4, 2e5)
    @printf("  α=%8.0f  f_1=%+.5e\n", a, f1clean(op, yc, a, 0.04)); flush(stdout)
end
println("\n=== C) exclusion-band sensitivity (α=2000, h=0.045) ===")
for excl in (0.02, 0.03, 0.04, 0.06, 0.08)
    @printf("  EXCL=%.2f (|y|<%.2f)  f_1=%+.5e\n", excl, cfg.W/2-excl,
            f1clean(op, yc, 2000.0, excl)); flush(stdout)
end
println("\n=== B) h-convergence (α=2000, EXCL=0.04) ===")
for h in (0.055, 0.045, 0.038, 0.032, 0.028)
    t=time(); oph=build(h=h); ych=ycoord(oph)
    @printf("  h=%.3f N=%6d  f_1=%+.6e  (%.0fs)\n", h, oph.N,
            f1clean(oph, ych, 2000.0, 0.04), time()-t); flush(stdout)
    GC.gc()
end
println("DONE"); flush(stdout)
