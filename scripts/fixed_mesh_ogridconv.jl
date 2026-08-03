#!/usr/bin/env julia
# scripts/fixed_mesh_ogridconv.jl — TRUSTWORTHINESS CHECK on the STRUCTURED square-
# obstacle mesh (hard-wall MaxwellWallBC). This is the same hard-wall physics that
# scattered on the unstructured hole; the question is whether the structured block mesh
# fixes it. A) one mesh: does matrix-free reg-GMRES converge now? + reg-LU truth.
# B) h-convergence via reg-GMRES (matrix-free ⇒ reaches fine h).
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_ogridconv.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "blockstruct_geo.jl"))
using VicinityOpt, VicinityOpt.Simulate
using Trixi, FermiSea
using Gmsh: gmsh
using SparseArrays, LinearAlgebra, Krylov, LinearMaps, Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)

function mesh_it(geo,inp)   # transfinite ⇒ structured; do NOT force Algorithm 8/11
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.open(geo); gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1)
    gmsh.model.mesh.generate(2)
    _,tags,_=gmsh.model.mesh.getElements(2); nc=sum(length.(tags)); gmsh.write(inp); gmsh.finalize(); (inp,nc)
end
function build(h)
    geo=joinpath(workdir,"ogc.geo"); inp=joinpath(workdir,"ogc.inp")
    # obstacle offset in BOTH x and y to break the source↔drain (M_x) and up-down (M_y)
    # mirror symmetries — both are required for a nonzero f_1.
    write_blockstruct_geo(geo; h=h, xc=0.65, yc=0.20, s=0.10, xPL=0.45, xPR=0.85); _,nc=mesh_it(geo,inp)
    eqs=FermiSea.IsotropicFermiHarmonics2D(sim.n_harmonics; v_fermi=sim.v_fermi)
    mesh=Trixi.P4estMesh{2}(inp; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source,:contact_drain,:probe_A,:probe_B,:walls,:obstacle])
    bcA=FermiSea.FloatingProbeBC(); bcB=FermiSea.FloatingProbeBC()
    bcs=(contact_source=FermiSea.CurrentContactBC(sim.I_source), contact_drain=FermiSea.OhmicContactBC(0.0),
         probe_A=bcA, probe_B=bcB, walls=FermiSea.MaxwellWallBC(1.0), obstacle=FermiSea.MaxwellWallBC(1.0))
    coll=FermiSea.LinearCollisionMatrix(eqs; gamma_mr=sim.gamma_mr, gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    ic0(x,t,eq)=zero(Trixi.SVector{Trixi.nvariables(eq),Float64})
    solver=DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)
    semi=SemidiscretizationHyperbolic(mesh,eqs,ic0,solver; boundary_conditions=bcs, source_terms=coll)
    _,_,dg,cache=Trixi.mesh_equations_solver_cache(semi)
    ext()=(FermiSea._current_contact_potential(bcA,eqs,dg,cache)-
           FermiSea._current_contact_potential(bcB,eqs,dg,cache))/sim.I_source
    ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0)
    (; semi, ext, N, b, nc)
end
function f1_lu(s; eps=1e-10)
    N=s.N; Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(s.b);ej=zeros(N)
    for j in 1:N
        ej[j]=1.0; Trixi.rhs!(tmp,ej,s.semi,0.0); c[j]=s.ext(); ej[j]=0.0
        @inbounds for i in 1:N; v=tmp[i]-s.b[i]; abs(v)>1e-12 && (push!(Ir,i);push!(Jc,j);push!(Vv,v)); end
    end
    A=sparse(Ir,Jc,Vv,N,N)+eps*spdiagm(0=>ones(N)); dot(c, lu(A)\(-s.b))
end
function f1_gmres(s; eps=1e-6)
    scr=similar(s.b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,s.semi,0.0); @. o=scr-s.b+eps*v; o), s.N; ismutating=true)
    u,st=Krylov.gmres(A,-s.b; restart=true, memory=100, atol=1e-11, rtol=1e-10, itmax=30000)
    Trixi.rhs!(scr,u,s.semi,0.0); (s.ext(), st.niter, norm(scr .+ eps.*u)/max(norm(u),1), st.solved)
end
function main()
    println("=== A) coarse h=0.055: reg-GMRES convergence test + reg-LU truth ===")
    s=build(0.055); @printf("  N=%d cells=%d\n", s.N, s.nc); flush(stdout)
    t=time(); f1g,it,res,cv=f1_gmres(s; eps=1e-6)
    @printf("  reg-GMRES f_1=%+.6e  (it=%d res=%.0e conv=%s %.0fs)\n", f1g,it,res,cv,time()-t); flush(stdout)
    t=time(); f1l=f1_lu(s; eps=1e-10)
    @printf("  reg-LU    f_1=%+.6e  Δgmres=%+.1e  (%.0fs)\n", f1l, f1g-f1l, time()-t); flush(stdout)
    println("\n=== B) h-convergence via reg-GMRES ===")
    println("   h       N       f_1            Δ          ratio   (it,res)")
    prev=NaN; prevd=NaN
    for h in (0.06, 0.048, 0.038, 0.030, 0.024)
        t=time(); s2=build(h); f1,it,res,cv=f1_gmres(s2; eps=1e-6)
        d=isnan(prev) ? NaN : f1-prev; r=(isnan(prevd)||prevd==0) ? NaN : d/prevd
        @printf("  %.3f  %6d  %+.6e  %+.2e  %+.3f  (%d,%.0e,%s %.0fs)\n", h,s2.N,f1,d,r,it,res,cv,time()-t)
        flush(stdout); prevd=d; prev=f1; GC.gc()
    end
end
main()
println("DONE"); flush(stdout)
