#!/usr/bin/env julia
# scripts/fixed_mesh_debug.jl — localize the LU-vs-GMRES discrepancy.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.05)   # coarser -> faster assembly for debug
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel_sym.geo"); inp = joinpath(workdir, "channel_sym.inp")
write_channel_geo_symmetric(geo, cfg; h=cfg.lc)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)

field = DensityField(cfg; nx=30, ny=18, alpha_max=100.0)
ev = FixedEvaluator(inp, field, sim); semi = ev.semi
mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)
extract_f1() = begin
    _,_,dg,ca = Trixi.mesh_equations_solver_cache(semi)
    (FermiSea._current_contact_potential(ev.bc_probe_A, equations, dg, ca) -
     FermiSea._current_contact_potential(ev.bc_probe_B, equations, dg, ca)) / sim.I_source
end

clear!(field)
ode = Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
b = similar(ode.u0); Trixi.rhs!(b, zero(ode.u0), semi, 0.0)
println("N=$N"); flush(stdout)

# assemble A0 (rho=0) + c
Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(b);ej=zeros(N);tol=1e-12
t0=time()
for j in 1:N
    ej[j]=1.0; Trixi.rhs!(tmp,ej,semi,0.0); c[j]=extract_f1(); ej[j]=0.0
    @inbounds for i in 1:N
        v=tmp[i]-b[i]; abs(v)>tol && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
    end
end
A0=sparse(Ir,Jc,Vv,N,N)
@printf("assembled nnz=%d (%.0fs)\n", nnz(A0), time()-t0); flush(stdout)

# CHECK A: assembled A0 vs matrix-free rhs at rho=0, random v
using Random; Random.seed!(0); vr=randn(N)
clear!(field); Trixi.rhs!(tmp,vr,semi,0.0); refA = tmp .- b
@printf("CHECK A  ‖A0 v - (rhs(v)-b)‖/‖.‖ = %.3e\n", norm(A0*vr-refA)/norm(refA)); flush(stdout)

# b1u at alpha=1
field.alpha_max=1.0; onev=ones(N); fill!(field.rho,1.0); r1=similar(b); Trixi.rhs!(r1,onev,semi,0.0)
clear!(field); r0=similar(b); Trixi.rhs!(r0,onev,semi,0.0); b1u=r1.-r0

rho_perdof=zeros(N)
function set_rho_perdof!(rg)
    w=Trixi.wrap_array(rho_perdof,semi); nc=cache.elements.node_coordinates
    nn=size(w,2); nel=size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; w[v,ii,jj,e]=rg[ci,cj]; end
    end
end

# obstacle
paint_blob!(field, cfg.x_c, 0.10, 0.16); rho_grid=copy(field.rho); set_rho_perdof!(rho_grid)
ALPHA=100.0
# CHECK B: full operator vs matrix-free rhs at this rho/alpha
field.alpha_max=ALPHA; field.rho.=rho_grid
Trixi.rhs!(tmp,vr,semi,0.0); refB=tmp.-b
Aop = A0 + spdiagm(0=>(ALPHA.*rho_perdof).*b1u)
@printf("CHECK B  ‖A(ρ)v - (rhs(v)-b)‖/‖.‖ = %.3e\n", norm(Aop*vr-refB)/norm(refB)); flush(stdout)

# CHECK C: LU solve is a true steady state (matrix-free residual)
u = lu(Aop)\(-b)
field.alpha_max=ALPHA; field.rho.=rho_grid
res=similar(u); Trixi.rhs!(res,u,semi,0.0)   # should be ~0 at steady state
@printf("CHECK C  ‖rhs(u_LU)‖/‖u‖ = %.3e   ‖u‖=%.3e\n", norm(res)/norm(u), norm(u)); flush(stdout)

# CHECK D: f1 three ways
f1_cu = dot(c,u)
Trixi.rhs!(res,u,semi,0.0); f1_extract = extract_f1()
field.alpha_max=ALPHA; field.rho.=rho_grid
f1_gmres,info = run_f1_fixed_steady(ev; itmax=25000, memory=100, atol=1e-11, rtol=1e-10)
@printf("CHECK D  f1: cᵀu=%+.5e  extract(u_LU)=%+.5e  GMRES=%+.5e\n",
        f1_cu, f1_extract, f1_gmres); flush(stdout)
println("DONE"); flush(stdout)
