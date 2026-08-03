#!/usr/bin/env julia
# scripts/fixed_mesh_holeconv.jl — (b) TRUSTWORTHINESS CHECK for the clean-gap,
# floating hard-wall obstacle (offset circle hole, MaxwellWallBC) on the exact geometry
# shown by show_gmsh_obstacle.jl. The earlier hard-wall hole (tight gaps) was
# h-NON-convergent; here the gaps are fat/clean. Solve f_1 = (V_A-V_B)/I:
#   A) one mesh: reg-LU ground truth + test whether matrix-free reg-GMRES converges now.
#   B) h-convergence via reg-LU (the solver that works on this singular reflecting operator).
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_holeconv.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt, VicinityOpt.Simulate
using Trixi, FermiSea
using Gmsh: gmsh
using SparseArrays, LinearAlgebra, Krylov, LinearMaps, Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W = 1.6, 0.8; W2=W/2; xc,yc,R = 0.8, 0.10, 0.18; wc=0.12; xPL,xPR = 0.60,1.00
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)

function write_geo(path; h, nob=80)
    open(path,"w") do io
        @printf(io,"lc=%.5f;\n",h)
        P=[(0.0,-W2),(0.0,-wc/2),(0.0,wc/2),(0.0,W2),(xPL,W2),(xPR,W2),(L,W2),
           (L,wc/2),(L,-wc/2),(L,-W2),(xPR,-W2),(xPL,-W2)]
        for (i,(x,y)) in enumerate(P); @printf(io,"Point(%d)={%.6f,%.6f,0,lc};\n",i,x,y); end
        for i in 1:12; @printf(io,"Line(%d)={%d,%d};\n", i, i, i==12 ? 1 : i+1); end
        println(io,"Curve Loop(1)={",join(1:12,","),"};")
        th=range(0,2pi,length=nob+1)[1:end-1]
        for (k,t) in enumerate(th); @printf(io,"Point(%d)={%.6f,%.6f,0,lc};\n",100+k, xc+R*cos(t), yc+R*sin(t)); end
        for k in 1:nob; @printf(io,"Line(%d)={%d,%d};\n", 200+k, 100+k, 100+(k%nob)+1); end
        println(io,"Curve Loop(2)={",join(201:200+nob,","),"};")
        println(io,"Plane Surface(1)={1,2};")
        println(io,"Physical Surface(\"domain\")={1};")
        println(io,"Physical Curve(\"contact_source\")={2};")
        println(io,"Physical Curve(\"contact_drain\")={8};")
        println(io,"Physical Curve(\"probe_A\")={5};")
        println(io,"Physical Curve(\"probe_B\")={11};")
        println(io,"Physical Curve(\"walls\")={1,3,4,6,7,9,10,12};")
        println(io,"Physical Curve(\"obstacle\")={",join(201:200+nob,","),"};")
    end; path
end
function mesh_it(geo,inp)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.open(geo); gmsh.option.setNumber("Mesh.Algorithm",8)
    gmsh.option.setNumber("Mesh.RecombineAll",1); gmsh.option.setNumber("Mesh.RecombinationAlgorithm",1)
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1); gmsh.model.mesh.generate(2)
    _,tags,_=gmsh.model.mesh.getElements(2); nc=sum(length.(tags)); gmsh.write(inp); gmsh.finalize(); (inp,nc)
end
function build(h)
    geo=joinpath(workdir,"hole.geo"); inp=joinpath(workdir,"hole.inp")
    write_geo(geo; h=h); _,nc=mesh_it(geo,inp)
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
    (; semi, ext, N, b, u0=ode.u0, nc)
end
function f1_lu(s; eps=1e-10)
    N=s.N; Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(s.b);ej=zeros(N)
    for j in 1:N
        ej[j]=1.0; Trixi.rhs!(tmp,ej,s.semi,0.0); c[j]=s.ext(); ej[j]=0.0
        @inbounds for i in 1:N
            v=tmp[i]-s.b[i]; abs(v)>1e-12 && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
        end
    end
    A=sparse(Ir,Jc,Vv,N,N)+eps*spdiagm(0=>ones(N)); dot(c, lu(A)\(-s.b))
end
function f1_gmres(s; eps=1e-6)
    scr=similar(s.b)
    A=LinearMap{Float64}((o,v)->(Trixi.rhs!(scr,v,s.semi,0.0); @. o=scr-s.b+eps*v; o), s.N; ismutating=true)
    u,st=Krylov.gmres(A,-s.b; restart=true, memory=100, atol=1e-11, rtol=1e-10, itmax=20000)
    Trixi.rhs!(scr,u,s.semi,0.0); (s.ext(), st.niter, norm(scr .+ eps.*u)/max(norm(u),1), st.solved)
end

function main()
    @printf("clean-gap hard-wall hole: L=%.1f W=%.1f circle R=%.2f @ (%.2f,%+.2f)\n", L,W,R,xc,yc)
    println("\n=== A) coarse mesh h=0.06: reg-LU truth + reg-GMRES convergence test ===")
    s = build(0.06); @printf("  N=%d (cells=%d)\n", s.N, s.nc); flush(stdout)
    t=time(); f1lu=f1_lu(s; eps=1e-10); @printf("  reg-LU   f_1=%+.6e  [truth]  (%.0fs)\n", f1lu, time()-t); flush(stdout)
    t=time(); f1g,it,res,cv=f1_gmres(s; eps=1e-6)
    @printf("  reg-GMRES f_1=%+.6e  Δlu=%+.1e  (it=%d res=%.0e conv=%s %.0fs)\n", f1g,f1g-f1lu,it,res,cv,time()-t); flush(stdout)
    println("\n=== B) h-convergence via reg-LU ===")
    println("   h       N       f_1            Δ          ratio")
    prev=NaN; prevd=NaN
    for h in (0.07, 0.058, 0.048, 0.040, 0.034)
        t=time(); s2=build(h); f1=f1_lu(s2; eps=1e-10)
        d = isnan(prev) ? NaN : f1-prev; r=(isnan(prevd)||prevd==0) ? NaN : d/prevd
        @printf("  %.3f  %6d  %+.6e  %+.2e  %+.3f  (%.0fs)\n", h, s2.N, f1, d, r, time()-t)
        flush(stdout); prevd=d; prev=f1; GC.gc()
    end
end
main()
println("DONE"); flush(stdout)
