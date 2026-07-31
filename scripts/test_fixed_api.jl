#!/usr/bin/env julia
# scripts/test_fixed_api.jl — verify the promoted src/ API (write_point_geo,
# assemble_fixed_operator, f1_exact, f1_adjoint_grad) works end-to-end.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf, Random

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo=joinpath(workdir,"apitest.geo"); inp=joinpath(workdir,"apitest.inp")

# module's write_point_geo (was scripts/pointgeo.jl)
# NOTE: design grid must be COARSER than the mesh, else some design cells have no
# mesh nodes (gradient=0 there). 20x12 design on an h=0.05 mesh is safe.
NX,NY = 20,12
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.05)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)

field = DensityField(cfg; nx=NX, ny=NY, alpha_max=2000.0)
ev = FixedEvaluator(inp, field, sim)
op = assemble_fixed_operator(ev)
@printf("assembled FixedOperator: N=%d design=%dx%d nnz(A0)=%d\n",
        op.N, op.nx, op.ny, length(op.A0.nzval)); flush(stdout)

# random (partly-solid) density -> a real signal; used for both checks
Random.seed!(0); rg = clamp.(0.2 .+ 0.1randn(NX,NY), 0, 1)

# f1_exact vs run_f1_fixed_steady (GMRES)
f_lu = f1_exact(op, rg; alpha=2000.0)
field.alpha_max = 2000.0; field.rho .= rg
f_gm, info = run_f1_fixed_steady(ev; itmax=40000, memory=120, atol=1e-11, rtol=1e-10)
@printf("f1_exact=%+.5e  GMRES=%+.5e  rel=%.2e\n", f_lu, f_gm,
        abs(f_lu-f_gm)/max(abs(f_gm),1e-12)); flush(stdout)

# adjoint gradient vs finite difference
f0, g = f1_adjoint_grad(op, rg; alpha=2000.0)
maxrel = Ref(0.0)
for (ci,cj) in ((6,6),(11,6),(15,4))
    e=1e-6
    rp=copy(rg); rp[ci,cj]+=e; rm=copy(rg); rm[ci,cj]-=e
    fd=(f1_exact(op,rp;alpha=2000.0)-f1_exact(op,rm;alpha=2000.0))/(2e)
    rel=abs(fd-g[ci,cj])/max(abs(fd),1e-12); maxrel[]=max(maxrel[],rel)
    @printf("  grad(%2d,%2d) adj=%+.4e fd=%+.4e rel=%.2e\n",ci,cj,g[ci,cj],fd,rel)
end
verdict = maxrel[] < 1e-3 ? "PASS" : "FAIL"
@printf("gradient check max rel = %.2e  %s\n", maxrel[], verdict)
println("DONE"); flush(stdout)
