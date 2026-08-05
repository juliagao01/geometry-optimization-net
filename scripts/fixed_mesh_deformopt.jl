#!/usr/bin/env julia
# scripts/fixed_mesh_deformopt.jl — find the (circular-ish) MOST OPTIMAL obstacle for f_1.
# Deformable boundary r(θ)=R+a2 cos2θ + b2 sin2θ + b3 sin3θ on the O-grid, in a TALL,
# roomy channel (full-perimeter contacts: source=whole left wall, drain=whole right wall).
# Optimize (xc,yc,R,a2,b2,b3) over |f_1| (linear reg-GMRES solve), deformation bounded so
# the shape stays round (no corners/spikes); then PIN the winner by h-refinement.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_deformopt.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using VicinityOpt, VicinityOpt.Simulate
using Trixi, FermiSea
using Gmsh: gmsh
using LinearAlgebra, Krylov, LinearMaps, Printf, JLD2
using Optim
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
L,W,W2 = 1.8, 1.4, 0.7
xPL, xPR, RING = 0.25, 1.55, 0.06
SEARCH_H, EPS = 0.07, 1e-6

mesh_it(geo,inp)=(gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0); gmsh.open(geo);
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1); gmsh.model.mesh.generate(2); gmsh.write(inp); gmsh.finalize(); inp)
function f1_of(h; xc,yc,R,a2,b2,b3, tag="opt")
    geo=joinpath(workdir,"$(tag).geo"); inp=joinpath(workdir,"$(tag).inp")
    write_ogrid_deformed(geo; L=L,W=W,xc=xc,yc=yc,R=R, A=[0.0,a2], B=[0.0,b2,b3],
                         ring=RING, xPL=xPL, xPR=xPR, h=h); _,nc=mesh_it(geo,inp), 0
    eqs=FermiSea.IsotropicFermiHarmonics2D(sim.n_harmonics; v_fermi=sim.v_fermi)
    mesh=Trixi.P4estMesh{2}(inp; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source,:contact_drain,:probe_A,:probe_B,:walls,:obstacle])
    bcA=FermiSea.FloatingProbeBC(); bcB=FermiSea.FloatingProbeBC()
    bcs=(contact_source=FermiSea.CurrentContactBC(1.0), contact_drain=FermiSea.OhmicContactBC(0.0),
         probe_A=bcA, probe_B=bcB, walls=FermiSea.MaxwellWallBC(1.0), obstacle=FermiSea.MaxwellWallBC(1.0))
    coll=FermiSea.LinearCollisionMatrix(eqs; gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    ic0(x,t,eq)=zero(Trixi.SVector{Trixi.nvariables(eq),Float64})
    solver=DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)
    semi=SemidiscretizationHyperbolic(mesh,eqs,ic0,solver; boundary_conditions=bcs, source_terms=FermiSea.SourceTerms(coll))
    _,_,dg,cache=Trixi.mesh_equations_solver_cache(semi)
    dV(u)=(scr=similar(u); Trixi.rhs!(scr,u,semi,0.0);
           FermiSea._current_contact_potential(bcA,eqs,dg,cache)-FermiSea._current_contact_potential(bcB,eqs,dg,cache))
    ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0); scr=similar(b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,semi,0.0); @. o=scr-b+EPS*v; o), N; ismutating=true)
    u,st=Krylov.gmres(A,-b; restart=true,memory=100,atol=1e-11,rtol=1e-10,itmax=40000)
    (dV(u), N, st.niter, st.solved)
end
function shape_ok(xc,yc,R,a2,b2,b3)
    θ=range(0,2π,length=181); r=@. R + a2*cos(2θ)+b2*sin(2θ)+b3*sin(3θ)
    rmax=maximum(r); rmin=minimum(r); s=rmax+RING
    rmin > 0.6*R &&                                    # stays round (no deep dents)
        (yc+s < W2-0.04) && (yc-s > -(W2-0.04)) &&
        (xc-s > xPL+0.03) && (xc+s < xPR-0.03) && R>0.15
end
const HIST=NTuple{7,Float64}[]; neval=Ref(0)
function objective(p)
    xc,yc,R,a2,b2,b3 = p; shape_ok(xc,yc,R,a2,b2,b3) || return 0.0
    try
        f1,N,it,cv = f1_of(SEARCH_H; xc=xc,yc=yc,R=R,a2=a2,b2=b2,b3=b3); cv || return 0.0
        neval[]+=1; push!(HIST,(xc,yc,R,a2,b2,b3,f1))
        (neval[]%5==0)&&(@printf("  [%3d] xc=%.2f yc=%+.2f R=%.2f a2=%+.2f b2=%+.2f b3=%+.2f  f1=%+.5e\n",
                                 neval[],xc,yc,R,a2,b2,b3,f1); flush(stdout))
        return -abs(f1)
    catch; return 0.0; end
end
println("=== OPTIMIZE |f_1| over deformable circular-ish obstacle (tall channel W=$W) ===")
lb=[0.60,-0.45,0.15,-0.12,-0.12,-0.12]; ub=[1.20,0.45,0.42,0.12,0.12,0.12]
x0=[0.90,0.15,0.28,0.02,0.05,0.02]
res=Optim.optimize(objective, x0, ParticleSwarm(lower=lb,upper=ub,n_particles=14), Optim.Options(iterations=6))
xb=Optim.minimizer(res); fb=-Optim.minimum(res)
@printf("\nBEST (h=%.3f, %d evals): xc=%.3f yc=%+.3f R=%.3f a2=%+.3f b2=%+.3f b3=%+.3f  |f1|=%.6e\n",
        SEARCH_H, neval[], xb..., fb); flush(stdout)
function pin(xb)
    println("\n=== PIN winner: h-refine f_1 ==="); println("   h       N        f_1           Δ")
    prev=NaN
    for h in (0.06,0.048,0.038,0.030)
        t=time(); f1,N,it,cv=f1_of(h; xc=xb[1],yc=xb[2],R=xb[3],a2=xb[4],b2=xb[5],b3=xb[6], tag="pin")
        d=isnan(prev) ? NaN : f1-prev
        @printf("  %.3f  %6d  %+.6e  %+.2e  (it=%d %s %.0fs)\n", h,N,f1,d,it,cv,time()-t); flush(stdout); prev=f1; GC.gc()
    end
end
pin(xb)
@save joinpath(workdir,"result_deformed.jld2") xbest=xb f1_search=fb history=HIST L W xPL xPR RING sim
println("saved runs/fixed_mesh/result_deformed.jld2"); println("DONE"); flush(stdout)
