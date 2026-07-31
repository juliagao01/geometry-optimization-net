#!/usr/bin/env julia
# scripts/fixed_mesh_contam.jl — how much of f_1 is PROBE CONTAMINATION?
# The optimized circle's soft edge deposits density on the top/bottom walls where
# the floating probes sit. Zero the obstacle density within a band of those walls
# and see how f_1 changes: a big drop => the signal was largely a probe artifact.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
ALPHA=2000.0
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
d = load(joinpath(workdir,"result_circle.jld2")); Cbest=d["C"]; R0=d["R0"]
geo=joinpath(workdir,"ct.geo"); inp=joinpath(workdir,"ct.inp")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.045)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
field = DensityField(cfg; nx=64, ny=40, alpha_max=1.0)
ev = FixedEvaluator(inp, field, sim)
op = assemble_fixed_operator(ev)
semi = ev.semi
_, equations, _, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)
nc = cache.elements.node_coordinates

# per-dof density of the optimized circle, with rho forced to 0 where |y|>ycut
function rpd_masked(ycut)
    rpd = zeros(op.N); w = Trixi.wrap_array(rpd, semi)
    nn=size(w,2); nel=size(w,4); M=length(Cbest)÷2
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        if abs(y) > ycut
            for v in 1:nvars; w[v,ii,jj,e]=0.0; end
            continue
        end
        dx=x-cfg.x_c; dy=y; dd=hypot(dx,dy); th=atan(dy,dx); R=R0
        for n in 1:M; R+=Cbest[2n-1]*cos(n*th)+Cbest[2n]*sin(n*th); end
        val=1/(1+exp((dd-R)/0.02))
        for v in 1:nvars; w[v,ii,jj,e]=val; end
    end
    rpd
end

println("optimized circle reaches |y| ≈ ", round(R0+sum(abs,Cbest),digits=3),
        "; probes at |y|=0.3, contacts within x∈[0.425,0.575].")
println("\nf_1 vs how far the obstacle density is excluded from the ±0.3 walls:")
@printf("  %-24s %-14s\n","exclude |y|>ycut","f_1")
for ycut in (0.30, 0.29, 0.28, 0.27, 0.26, 0.24)
    f = f1_exact_perdof(op, rpd_masked(ycut); alpha=ALPHA)
    @printf("  ycut=%.2f (band %.2f)        %+.5e\n", ycut, 0.30-ycut, f); flush(stdout)
end
println("DONE"); flush(stdout)
