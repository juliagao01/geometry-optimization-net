#!/usr/bin/env julia
# scripts/fixed_mesh_circle_f2opt.jl — OPTIMAL f_2 over a circular obstacle (O-grid).
# Combines the circle O-grid geometry with the nonlinear f_2 pipeline: add the
# InertialStressSource (λ=1) and compute f_2 by 2nd-order perturbation theory
#   A0 u1 = -b  (f1=ΔV(u1)),  A0 u2 = -S_nl(u1)  (f2=ΔV(u2))
# via matrix-free reg-GMRES (no assembly). Maximize |f_2| over (xc,yc,R); pin the winner.
# NOTE: f_2 here is for the LOCAL inertial-stress nonlinearity (algebraic j⊗j→m2 part of
# the convective term); the full flux-level (u·∇)u is the mentor-coordinated step.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_circle_f2opt.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using VicinityOpt, VicinityOpt.Simulate
using Trixi, FermiSea
using Gmsh: gmsh
using LinearAlgebra, Krylov, LinearMaps, Printf, JLD2
using Optim
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
L,W,W2 = 1.6, 0.8, 0.4
xPL, xPR, RING = 0.35, 1.25, 0.05
SEARCH_H, EPS = 0.06, 1e-6

function mesh_it(geo,inp)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.open(geo); gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1)
    gmsh.model.mesh.generate(2)
    _,tags,_=gmsh.model.mesh.getElements(2); nc=sum(length.(tags)); gmsh.write(inp); gmsh.finalize(); (inp,nc)
end
function make_semi(inp; nonlinear::Bool)
    eqs=FermiSea.IsotropicFermiHarmonics2D(sim.n_harmonics; v_fermi=sim.v_fermi)
    mesh=Trixi.P4estMesh{2}(inp; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source,:contact_drain,:probe_A,:probe_B,:walls,:obstacle])
    bcA=FermiSea.FloatingProbeBC(); bcB=FermiSea.FloatingProbeBC()
    bcs=(contact_source=FermiSea.CurrentContactBC(sim.I_source), contact_drain=FermiSea.OhmicContactBC(0.0),
         probe_A=bcA, probe_B=bcB, walls=FermiSea.MaxwellWallBC(1.0), obstacle=FermiSea.MaxwellWallBC(1.0))
    coll=FermiSea.LinearCollisionMatrix(eqs; gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    src = nonlinear ? FermiSea.SourceTerms(coll, FermiSea.InertialStressSource(eqs,1.0)) : FermiSea.SourceTerms(coll)
    ic0(x,t,eq)=zero(Trixi.SVector{Trixi.nvariables(eq),Float64})
    solver=DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)
    semi=SemidiscretizationHyperbolic(mesh,eqs,ic0,solver; boundary_conditions=bcs, source_terms=src)
    Trixi.semidiscretize(semi,(0.0,1.0))               # inits boundary projectors
    _,_,dg,cache=Trixi.mesh_equations_solver_cache(semi)
    dV(u,scr)=(Trixi.rhs!(scr,u,semi,0.0);
               FermiSea._current_contact_potential(bcA,eqs,dg,cache)-
               FermiSea._current_contact_potential(bcB,eqs,dg,cache))
    (semi, dV)
end
# f1,f2 by perturbation (2 reg-GMRES solves). Returns (f1, f2, it1, it2, conv).
function f1f2(h; xc,yc,R, tag="opt")
    geo=joinpath(workdir,"$(tag).geo"); inp=joinpath(workdir,"$(tag).inp")
    write_ogrid_geo(geo; L=L,W=W,xc=xc,yc=yc,R=R,ring=RING,xPL=xPL,xPR=xPR,h=h); _,nc=mesh_it(geo,inp)
    sl,dVl = make_semi(inp; nonlinear=false)
    sn,_   = make_semi(inp; nonlinear=true)
    n = Trixi.ndofs(sl)*Trixi.nvariables(sl.equations)
    u0=zeros(n); b=similar(u0); Trixi.rhs!(b,u0,sl,0.0); scr=similar(b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,sl,0.0); @. o=scr-b+EPS*v; o), length(b); ismutating=true)
    u1,s1=Krylov.gmres(A,-b;  restart=true,memory=100,atol=1e-11,rtol=1e-10,itmax=40000)
    rn=similar(b); rl=similar(b); Trixi.rhs!(rn,u1,sn,0.0); Trixi.rhs!(rl,u1,sl,0.0)
    S1 = rn .- rl
    u2,s2=Krylov.gmres(A,-S1; restart=true,memory=100,atol=1e-11,rtol=1e-10,itmax=40000)
    f1=dVl(u1,scr); f2=dVl(u2,scr)
    (f1,f2,s1.niter,s2.niter, s1.solved&&s2.solved, length(b))
end
valid(xc,yc,R)= (s=R+RING; R>0.08 && yc+s<W2-0.03 && yc-s>-(W2-0.03) && xc-s>xPL+0.02 && xc+s<xPR-0.02)
const HIST=Vector{NTuple{5,Float64}}(); neval=Ref(0)
function objective(p)
    xc,yc,R=p; valid(xc,yc,R) || return 0.0
    try
        f1,f2,_,_,cv,_ = f1f2(SEARCH_H; xc=xc,yc=yc,R=R); cv || return 0.0
        neval[]+=1; push!(HIST,(xc,yc,R,f1,f2))
        (neval[]%5==0)&&(@printf("  [%3d] xc=%.3f yc=%+.3f R=%.3f  f1=%+.5e f2=%+.5e\n",neval[],xc,yc,R,f1,f2);flush(stdout))
        return -abs(f2)
    catch; return 0.0; end
end
println("=== OPTIMIZE |f_2| over CIRCLE obstacle (xc,yc,R), O-grid perturbation @ h=$(SEARCH_H) ===")
lb=[0.55,-0.25,0.08]; ub=[1.05,0.25,0.28]; x0=[0.72,0.08,0.22]
res=Optim.optimize(objective, x0, ParticleSwarm(lower=lb,upper=ub,n_particles=12), Optim.Options(iterations=6))
xb=Optim.minimizer(res); fb=-Optim.minimum(res)
@printf("\nBEST (search h=%.3f, %d evals): xc=%.4f yc=%+.4f R=%.4f  |f2|=%.6e\n", SEARCH_H,neval[],xb...,fb); flush(stdout)
function pin(xb)
    println("\n=== PIN winner: h-refine f_2 (perturbation) ==="); println("   h       N       f_1           f_2           Δf2")
    prev=NaN
    for h in (0.05,0.04,0.032,0.026)
        t=time(); f1,f2,i1,i2,cv,N=f1f2(h; xc=xb[1],yc=xb[2],R=xb[3], tag="pin")
        d=isnan(prev) ? NaN : f2-prev
        @printf("  %.3f  %6d  %+.6e  %+.6e  %+.2e  (it %d/%d %s %.0fs)\n",h,N,f1,f2,d,i1,i2,cv,time()-t)
        flush(stdout); prev=f2; GC.gc()
    end
end
pin(xb)
@save joinpath(workdir,"result_circle_f2.jld2") xbest=xb f2_search=fb history=HIST SEARCH_H sim L W xPL xPR RING
println("saved runs/fixed_mesh/result_circle_f2.jld2"); println("DONE"); flush(stdout)
