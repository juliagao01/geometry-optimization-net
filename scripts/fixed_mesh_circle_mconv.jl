#!/usr/bin/env julia
# scripts/fixed_mesh_circle_mconv.jl — n_harmonics (M) convergence of f_1 and f_2 on the
# OPTIMAL circle, at FIXED mesh h, to see whether the velocity-angle truncation (m<=M)
# shifts the coefficients. f_1,f_2 via the 2-solve perturbation on the O-grid.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_circle_mconv.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using VicinityOpt, VicinityOpt.Simulate
using Trixi, FermiSea
using Gmsh: gmsh
using LinearAlgebra, Krylov, LinearMaps, Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W = 1.6, 0.8
xc,yc,R = 0.694, 0.083, 0.235     # the f_2-optimal circle
H, EPS = 0.045, 1e-6              # fixed mesh (isolates the M effect)

function mesh_it(geo,inp)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.open(geo); gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1)
    gmsh.model.mesh.generate(2); gmsh.write(inp); gmsh.finalize(); inp
end
function make_semi(inp, sim; nonlinear::Bool)
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
    Trixi.semidiscretize(semi,(0.0,1.0))
    _,_,dg,cache=Trixi.mesh_equations_solver_cache(semi)
    dV(u,scr)=(Trixi.rhs!(scr,u,semi,0.0);
               FermiSea._current_contact_potential(bcA,eqs,dg,cache)-
               FermiSea._current_contact_potential(bcB,eqs,dg,cache))
    (semi,dV)
end
function f1f2_M(M)
    sim=SimConfig(n_harmonics=M, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    geo=joinpath(workdir,"mc.geo"); inp=joinpath(workdir,"mc.inp")
    write_ogrid_geo(geo; L=L,W=W,xc=xc,yc=yc,R=R,ring=0.05,xPL=0.35,xPR=1.25,h=H); mesh_it(geo,inp)
    sl,dVl=make_semi(inp,sim; nonlinear=false); sn,_=make_semi(inp,sim; nonlinear=true)
    n=Trixi.ndofs(sl)*Trixi.nvariables(sl.equations)
    b=zeros(n); Trixi.rhs!(b,zeros(n),sl,0.0); scr=similar(b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,sl,0.0); @. o=scr-b+EPS*v; o), n; ismutating=true)
    u1,s1=Krylov.gmres(A,-b;  restart=true,memory=100,atol=1e-11,rtol=1e-10,itmax=60000)
    rn=similar(b); rl=similar(b); Trixi.rhs!(rn,u1,sn,0.0); Trixi.rhs!(rl,u1,sl,0.0)
    u2,s2=Krylov.gmres(A,-(rn.-rl); restart=true,memory=100,atol=1e-11,rtol=1e-10,itmax=60000)
    (dVl(u1,scr), dVl(u2,scr), n, s1.niter, s2.niter, s1.solved&&s2.solved)
end
function main()
    @printf("M-convergence on optimal circle R=%.3f @ (%.3f,%+.3f), fixed h=%.3f\n", R,xc,yc,H)
    println("   M   NVARS    N        f_1            f_2           (GMRES it1/it2 conv)")
    for M in (4,6,8)
        t=time(); f1,f2,N,i1,i2,cv=f1f2_M(M)
        @printf("  %2d   %3d   %6d  %+.6e  %+.6e  (%d/%d %s %.0fs)\n", M, 2M+1, N, f1, f2, i1, i2, cv, time()-t)
        flush(stdout); GC.gc()
    end
end
main()
println("DONE"); flush(stdout)
