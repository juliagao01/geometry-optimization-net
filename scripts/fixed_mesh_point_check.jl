#!/usr/bin/env julia
# scripts/fixed_mesh_point_check.jl — is f_1 at the optimized shape PHYSICAL?
#   Load the optimized density, evaluate f_1 via reg-LU across a range of eps and
#   via a tight GMRES. If f_1 is stable as eps->0 and matches GMRES, the large
#   value is real; if it scales with eps or disagrees, it's a near-singular
#   conditioning artifact from the obstacle nearly blocking the flow.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra, Printf, JLD2
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
ALPHA=2000.0
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
d = load(joinpath(workdir, get(ENV,"RESULT","result_circle.jld2")))
bestphys = d["bestphys"]; NX,NY = size(bestphys)
savedf1 = haskey(d,"best_f1") ? d["best_f1"] : get(d,"f",NaN)
@printf("loaded optimized shape %dx%d, saved f_1=%.4e\n", NX, NY, savedf1); flush(stdout)

geo=joinpath(workdir,"cp.geo"); inp=joinpath(workdir,"cp.inp")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.05)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
field = DensityField(cfg; nx=NX, ny=NY, alpha_max=1.0)
ev = FixedEvaluator(inp, field, sim); semi=ev.semi
_, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)
extract()=(FermiSea._current_contact_potential(ev.bc_probe_A,equations,dg,cache)-
           FermiSea._current_contact_potential(ev.bc_probe_B,equations,dg,cache))/sim.I_source
clear!(field); ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0)
println("assembling..."); flush(stdout)
Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(b);ej=zeros(N)
for j in 1:N
    ej[j]=1.0; Trixi.rhs!(tmp,ej,semi,0.0); c[j]=extract(); ej[j]=0.0
    @inbounds for i in 1:N
        v=tmp[i]-b[i]; abs(v)>1e-12 && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
    end
end
A0=sparse(Ir,Jc,Vv,N,N)
fill!(field.rho,1.0); ov=ones(N); r1=similar(b); Trixi.rhs!(r1,ov,semi,0.0)
clear!(field); r0=similar(b); Trixi.rhs!(r0,ov,semi,0.0); b1u=r1.-r0
rpd=zeros(N); w=Trixi.wrap_array(rpd,semi); nc=cache.elements.node_coordinates
nn=size(w,2); nel=size(w,4)
field.rho .= bestphys
for e in 1:nel, jj in 1:nn, ii in 1:nn
    x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
    ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
    cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
    for v in 1:nvars; w[v,ii,jj,e]=bestphys[ci,cj]; end
end
D = spdiagm(0=>(ALPHA.*rpd).*b1u)
println("\nreg-LU f_1 vs eps (optimized shape):"); flush(stdout)
for eps in (1e-4,1e-6,1e-8,1e-10,1e-12,1e-14)
    @printf("   eps=%.0e  f_1=%+.6e\n", eps, dot(c, lu(A0+D+eps*spdiagm(0=>ones(N)))\(-b))); flush(stdout)
end
field.alpha_max=ALPHA; field.rho.=bestphys
fg,info = run_f1_fixed_steady(ev; itmax=60000, memory=120, atol=1e-12, rtol=1e-11)
@printf("\nGMRES f_1=%+.6e (res=%.1e conv=%s iters=%d)\n", fg, info.residual, info.converged, info.iters)
println("DONE"); flush(stdout)
