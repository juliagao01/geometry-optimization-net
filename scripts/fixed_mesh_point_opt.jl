#!/usr/bin/env julia
# scripts/fixed_mesh_point_opt.jl — STEP 2 on a REAL signal.
#   Free-form density (topology) optimization on the point-contact geometry via
#   the discrete adjoint, with the EXACT regularized sparse-LU solver (handles
#   the stiff solid obstacle + null-mode, unlike GMRES). Maximizes f_1.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra, Printf, JLD2, Random
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
# design grid must be COARSER than the mesh node spacing (h=0.05 below), else some
# design cells contain no mesh node -> silent zero-gradient variables. NX<=L/h=20,
# NY<=W/h=12.
ALPHA=2000.0; EPS=1e-10; NX=18; NY=11; VOLFRAC=0.12; NITER=60; FILTR=2.5; MU=1.0
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
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
println("assembling (N=$N) ..."); flush(stdout); t0=time()
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
@printf("assembled nnz=%d (%.0fs)\n", nnz(A0), time()-t0); flush(stdout)

# dof<->cell maps
cellof=zeros(Int,N)
let tf=zeros(N); w=Trixi.wrap_array(tf,semi); nc=cache.elements.node_coordinates
    nn=size(w,2); nel=size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        lin=ci+(cj-1)*field.nx
        for v in 1:nvars; w[v,ii,jj,e]=lin; end
    end
    cellof.=round.(Int,tf)
end
rho_perdof=zeros(N)
function set_rpd!(rg)
    w=Trixi.wrap_array(rho_perdof,semi); nc=cache.elements.node_coordinates
    nn=size(w,2); nel=size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; w[v,ii,jj,e]=rg[ci,cj]; end
    end
end
Idn=spdiagm(0=>ones(N))
function fwd_adj(rg)
    set_rpd!(rg)
    F=lu(A0 + spdiagm(0=>(ALPHA.*rho_perdof).*b1u) + EPS*Idn)
    u=F\(-b); f1=dot(c,u); λ=F'\c
    g=zeros(field.nx*field.ny)
    @inbounds for j in 1:N
        gj=-λ[j]*(ALPHA*b1u[j])*u[j]
        gj!=0.0 && (g[cellof[j]]+=gj)
    end
    f1, reshape(g,field.nx,field.ny)
end
f1_only(rg)=(set_rpd!(rg); dot(c, lu(A0+spdiagm(0=>(ALPHA.*rho_perdof).*b1u)+EPS*Idn)\(-b)))

function _run()
# gradient check vs finite difference
Random.seed!(0); rg0=clamp.(0.2 .+ 0.1randn(NX,NY),0,1)
f0,g0=fwd_adj(rg0); maxrel=0.0
for (ci,cj) in ((6,7),(11,7),(16,4),(11,11))
    e=1e-6; rp=copy(rg0);rp[ci,cj]+=e; rm=copy(rg0);rm[ci,cj]-=e
    fd=(f1_only(rp)-f1_only(rm))/(2e); ad=g0[ci,cj]
    rel=abs(fd-ad)/max(abs(fd),1e-12); maxrel=max(maxrel,rel)
    @printf("  grad cell(%2d,%2d) adj=%+.4e fd=%+.4e rel=%.2e\n",ci,cj,ad,fd,rel)
end
@printf("gradient check max rel=%.2e  %s\n", maxrel, maxrel<1e-3 ? "PASS" : "FAIL"); flush(stdout)
maxrel<1e-3 || (println("gradient FAILED"); exit(1))

# ---- REGULARIZED volume-constrained projected gradient ascent ----
# density filter (radius FILTR -> minimum feature size, well-posed/mesh-indep) +
# perimeter (Dirichlet) penalty, to avoid the degenerate thin-wall optimum.
# Maximize J = f_1 - MU*perimeter(filtered rho).
function conefilter(M,R)
    nx,ny=size(M); O=zeros(nx,ny); Ri=ceil(Int,R)
    @inbounds for j in 1:ny, i in 1:nx
        s=0.0; wsum=0.0
        for dj in -Ri:Ri, di in -Ri:Ri
            ii=i+di; jj=j+dj; (1<=ii<=nx && 1<=jj<=ny) || continue
            w=R-sqrt(di*di+dj*dj); w>0 || continue
            s+=w*M[ii,jj]; wsum+=w
        end
        O[i,j]=s/wsum
    end; O
end
perim(M)=begin nx,ny=size(M); p=0.0
    @inbounds for j in 1:ny,i in 1:nx
        i<nx && (p+=(M[i+1,j]-M[i,j])^2); j<ny && (p+=(M[i,j+1]-M[i,j])^2)
    end; p end
gperim(M)=begin nx,ny=size(M); g=zeros(nx,ny)
    @inbounds for j in 1:ny,i in 1:nx
        if i<nx; d=2*(M[i+1,j]-M[i,j]); g[i,j]-=d; g[i+1,j]+=d end
        if j<ny; d=2*(M[i,j+1]-M[i,j]); g[i,j]-=d; g[i,j+1]+=d end
    end; g end
function proj_vol(M,V) lo,hi=-1.0,1.0
    for _ in 1:60; m=(lo+hi)/2; (sum(clamp.(M.+m,0,1))/length(M)>V) ? (hi=m) : (lo=m); end
    clamp.(M.+(lo+hi)/2,0,1)
end
Jof(rg)=begin rp=conefilter(rg,FILTR); f=f1_only(rp); (f-MU*perim(rp), f, perim(rp)) end
rho=fill(VOLFRAC,NX,NY); step=0.4; best_J=-Inf; best=copy(rho); best_f1=0.0; hist=NamedTuple[]
for it in 1:NITER
    rp=conefilter(rho,FILTR)
    f1,gf1=fwd_adj(rp); P=perim(rp); J=f1-MU*P
    gJ=conefilter(gf1 .- MU.*gperim(rp), FILTR)   # chain rule through filter
    gd=gJ./(maximum(abs,gJ)+1e-30)
    trial=proj_vol(rho.+step.*gd,VOLFRAC)
    Jt,ft,Pt=Jof(trial)
    if Jt>=J; rho=trial; else; step*=0.6; rho=trial; end
    if Jt>best_J; best_J=Jt; best=copy(rho); best_f1=ft; end
    push!(hist,(it=it,f1=ft,perim=Pt,J=Jt,vol=sum(rho)/length(rho)))
    @printf("it %2d  f_1=%+.5e  perim=%.2f  J=%+.5e  vol=%.3f step=%.3f\n",
            it,ft,Pt,Jt,sum(rho)/length(rho),step); flush(stdout)
    step<1e-3 && break
end
bestphys=conefilter(best,FILTR)
@printf("\nBEST J=%+.5e  f_1=%+.6e  (vol=%.3f, filter R=%.1f, MU=%.2g)\n",
        best_J, best_f1, sum(best)/length(best), FILTR, MU)
field.rho .= bestphys
@save joinpath(workdir,"result_point_opt.jld2") best bestphys best_f1 best_J hist cfg sim ALPHA NX NY FILTR MU
chars=" .:-=+*#%@"
println("\nOptimized obstacle, filtered density (top=+W/2; S=left mid, D=right mid, probes top/bot mid):")
for j in NY:-1:1
    print("  "); for i in 1:NX
        k=clamp(floor(Int,bestphys[i,j]*(length(chars)-1))+1,1,length(chars)); print(chars[k])
    end; println()
end
println("DONE"); flush(stdout)
end # _run
_run()
