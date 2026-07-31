#!/usr/bin/env julia
# scripts/fixed_mesh_point_conv.jl — mesh convergence of f_1 on point geometry,
# using the EXACT regularized sparse LU (no slow GMRES). Fixed obstacle.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
ALPHA = 2000.0; EPS = 1e-10

function f1_exact(h)
    geo=joinpath(workdir,"cp.geo"); inp=joinpath(workdir,"cp.inp")
    write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                    xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=h)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfg; nx=48, ny=28, alpha_max=1.0)
    ev = FixedEvaluator(inp, field, sim); semi = ev.semi
    _, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    nvars = Trixi.nvariables(equations)
    extract()=(FermiSea._current_contact_potential(ev.bc_probe_A,equations,dg,cache)-
               FermiSea._current_contact_potential(ev.bc_probe_B,equations,dg,cache))/sim.I_source
    clear!(field); ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0)
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
    paint_blob!(field,cfg.x_c,0.10,0.16); rg=copy(field.rho)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; w[v,ii,jj,e]=rg[ci,cj]; end
    end
    A=A0+spdiagm(0=>(ALPHA.*rpd).*b1u)+EPS*spdiagm(0=>ones(N))
    dot(c, lu(A)\(-b)), N
end

println("mesh convergence, obstacle yc=0.10 r=0.16 alpha=$ALPHA (exact reg-LU)")
for h in (0.06, 0.05, 0.04, 0.033, 0.028)
    t=time(); f1,N = f1_exact(h)
    @printf("  h=%.3f  N=%6d  f_1=%+.6e  (%.0fs)\n", h, N, f1, time()-t); flush(stdout)
end
println("DONE"); flush(stdout)
