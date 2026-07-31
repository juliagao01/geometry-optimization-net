#!/usr/bin/env julia
# scripts/fixed_mesh_point_lu.jl — validate the EXACT solver on the point-contact
# geometry, where a real f_1 (~0.12) exists. GMRES stalls at high alpha; the
# regularized sparse LU should give a clean f_1 that (a) converges as eps->0 to a
# NONZERO value (unlike the full-edge case where it ->0), (b) matches a long
# GMRES, (c) is stable under mesh refinement.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra
using Printf

include(joinpath(@__DIR__, "pointgeo.jl"))   # write_point_geo (shared)

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)

function assemble_and_test(h)
    geo=joinpath(workdir,"channel_point.geo"); inp=joinpath(workdir,"channel_point.inp")
    write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                    xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=h)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfg; nx=40, ny=24, alpha_max=1.0)
    ev = FixedEvaluator(inp, field, sim); semi = ev.semi
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    nvars = Trixi.nvariables(equations)
    extract() = (FermiSea._current_contact_potential(ev.bc_probe_A, equations, solver, cache) -
                 FermiSea._current_contact_potential(ev.bc_probe_B, equations, solver, cache))/sim.I_source
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
    field.alpha_max=1.0; ov=ones(N); fill!(field.rho,1.0); r1=similar(b); Trixi.rhs!(r1,ov,semi,0.0)
    clear!(field); r0=similar(b); Trixi.rhs!(r0,ov,semi,0.0); b1u=r1.-r0
    rpd=zeros(N)
    w=Trixi.wrap_array(rpd,semi); nc=cache.elements.node_coordinates
    nn=size(w,2); nel=size(w,4)
    paint_blob!(field, cfg.x_c, 0.10, 0.16); rg=copy(field.rho)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; w[v,ii,jj,e]=rg[ci,cj]; end
    end
    Idn=spdiagm(0=>ones(N))
    f1reg(alpha,eps)= dot(c, lu(A0 + spdiagm(0=>(alpha.*rpd).*b1u) + eps*Idn)\(-b))
    return ev, field, rg, f1reg, N
end

for h in (0.05, 0.035)
    println("\n########## mesh h=$h ##########"); flush(stdout)
    ev, field, rg, f1reg, N = assemble_and_test(h)
    @printf("N=%d\n", N); flush(stdout)
    # long GMRES reference at alpha=2000
    field.alpha_max=2000.0; field.rho.=rg
    fg,info = run_f1_fixed_steady(ev; itmax=80000, memory=150, atol=1e-12, rtol=1e-11)
    @printf("GMRES(alpha=2000) f_1=%+.6e  res=%.1e conv=%s\n", fg, info.residual, info.converged); flush(stdout)
    println("reg-LU alpha=2000:"); flush(stdout)
    for eps in (1e-4,1e-6,1e-8,1e-10,1e-12)
        @printf("   eps=%.0e  f_1=%+.6e\n", eps, f1reg(2000.0,eps)); flush(stdout)
    end
end
println("DONE"); flush(stdout)
