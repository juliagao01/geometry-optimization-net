#!/usr/bin/env julia
# scripts/fixed_mesh_ogridopt.jl — OPTIMIZE f_1 over a floating square obstacle on the
# STRUCTURED block mesh (where reg-GMRES converges & matches reg-LU). Free parameters:
# obstacle center (xc, yc) and half-size s. Objective = maximize |f_1|=(V_A-V_B)/I,
# subject to the obstacle staying clear of every wall and contact. Search on a fast
# coarse mesh (reg-LU, exact); then PIN the winning shape's f_1 by h-refinement.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_ogridopt.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "blockstruct_geo.jl"))
using VicinityOpt, VicinityOpt.Simulate
using Trixi, FermiSea
using Gmsh: gmsh
using SparseArrays, LinearAlgebra, Krylov, LinearMaps, Printf, JLD2
using Optim
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
L,W,W2,wc = 1.6, 0.8, 0.4, 0.12
xPL, xPR = 0.45, 1.15                 # fixed probe span (top & bottom)
SEARCH_H  = 0.075                      # fast coarse search mesh
MARGIN    = 0.05                       # min clearance to walls; keep edges off contacts

function mesh_it(geo,inp)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.open(geo); gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1)
    gmsh.model.mesh.generate(2)
    _,tags,_=gmsh.model.mesh.getElements(2); nc=sum(length.(tags)); gmsh.write(inp); gmsh.finalize(); (inp,nc)
end
function build(h; xc, yc, s, tag="opt")
    geo=joinpath(workdir,"$(tag).geo"); inp=joinpath(workdir,"$(tag).inp")
    write_blockstruct_geo(geo; L=L, W=W, xc=xc, yc=yc, s=s, wc=wc, xPL=xPL, xPR=xPR, h=h)
    _,nc=mesh_it(geo,inp)
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
function f1_gmres(s; eps=1e-6, itmax=60000)
    scr=similar(s.b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,s.semi,0.0); @. o=scr-s.b+eps*v; o), s.N; ismutating=true)
    u,st=Krylov.gmres(A,-s.b; restart=true, memory=100, atol=1e-11, rtol=1e-10, itmax=itmax)
    Trixi.rhs!(scr,u,s.semi,0.0); (s.ext(), st.niter, norm(scr .+ eps.*u)/max(norm(u),1), st.solved)
end

valid(xc,yc,s) = (s>0.06) && (yc+s < W2-MARGIN) && (yc-s > -(W2-MARGIN)) &&
                 (xc-s > 0.10) && (xc+s < L-0.10) &&
                 all(abs(e-wc/2)>0.02 && abs(e+wc/2)>0.02 for e in (yc-s,yc+s)) &&  # off source/drain rows
                 all(abs(e-xPL)>0.02 && abs(e-xPR)>0.02 for e in (xc-s,xc+s))       # off probe cols

const HIST = Vector{NTuple{4,Float64}}()   # (xc,yc,s,f1)
neval = Ref(0)
function objective(p)
    xc,yc,s = p
    valid(xc,yc,s) || return 0.0
    try
        st = build(SEARCH_H; xc=xc, yc=yc, s=s)
        f1 = f1_lu(st)
        neval[] += 1; push!(HIST,(xc,yc,s,f1))
        (neval[] % 5 == 0) && (@printf("  [%3d] xc=%.3f yc=%+.3f s=%.3f  f1=%+.6e\n",neval[],xc,yc,s,f1); flush(stdout))
        return -abs(f1)                    # minimize -|f1| ⇒ maximize |f1|
    catch
        return 0.0
    end
end

println("=== OPTIMIZE |f_1| over square obstacle (xc,yc,s), reg-LU @ h=$(SEARCH_H) ===")
lb=[0.30,-0.28,0.06]; ub=[1.30,0.28,0.26]; x0=[0.6154,0.2161,0.1244]  # warm-start from prior best
res = Optim.optimize(objective, x0, ParticleSwarm(lower=lb, upper=ub, n_particles=12),
                     Optim.Options(iterations=7, show_trace=false))
xb = Optim.minimizer(res); fb = -Optim.minimum(res)
@printf("\nBEST (search h=%.3f, %d evals): xc=%.4f yc=%+.4f s=%.4f  |f1|=%.6e\n",
        SEARCH_H, neval[], xb[1], xb[2], xb[3], fb); flush(stdout)

function pin_winner(xb)
    println("\n=== PIN the winner: h-refine f_1 (reg-GMRES) ===")
    println("   h       N       f_1            Δ")
    prev=NaN
    for h in (0.06, 0.045, 0.035, 0.028)
        t=time(); st=build(h; xc=xb[1], yc=xb[2], s=xb[3], tag="pin")
        f1,it,rr,cv=f1_gmres(st; eps=1e-6)
        d=isnan(prev) ? NaN : f1-prev
        @printf("  %.3f  %6d  %+.6e  %+.2e  (it=%d res=%.0e conv=%s %.0fs)\n", h,st.N,f1,d,it,rr,cv,time()-t)
        flush(stdout); prev=f1; GC.gc()
    end
end
pin_winner(xb)
@save joinpath(workdir,"result_ogrid.jld2") xbest=xb f1_search=fb history=HIST SEARCH_H sim L W wc xPL xPR
println("saved runs/fixed_mesh/result_ogrid.jld2"); println("DONE"); flush(stdout)
