#!/usr/bin/env julia
# scripts/fixed_mesh_wallobstacle.jl — TRUE hard-wall obstacle (no Brinkman).
# The optimized Fourier shape is carved as an interior HOLE with MaxwellWallBC on
# its boundary — a genuine reflecting wall, no penalization layer, no α. This is
# the physically trustworthy f_1 for the shape; check h-convergence.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_wallobstacle.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate
using Trixi, FermiSea
using Gmsh: gmsh
using SparseArrays, LinearAlgebra, Krylov, LinearMaps, Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6)
# n_harmonics=4 (NVARS=9) to keep N small enough for direct reg-LU on the
# unstructured hole mesh (GMRES fails on this singular operator; assembly needs N≲25k).
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
EPS=1e-10
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
d = load(joinpath(workdir,"result_circle.jld2")); C=d["C"]
# The optimized R0≈0.20 shape nearly fills the channel (|y|→0.26) → thin 0.04 wall
# gaps that hang unstructured hole-meshing. For the hard-wall CONVERGENCE demo use
# a moderate obstacle (fat gaps, cleanly meshable); it keeps the same b2/b4 dimple
# family so f_1≠0. (Re-optimize under the wall-BC objective separately.)
R0 = 0.12
xc=cfg.x_c; W2=cfg.W/2; xPL=cfg.x_probe-cfg.L_probe/2; xPR=cfg.x_probe+cfg.L_probe/2; wc=0.12
M=length(C)÷2
Robs(th) = R0 + sum(C[2n-1]*cos(n*th)+C[2n]*sin(n*th) for n in 1:M)
@printf("obstacle R0=%.3f, reaches |y|≈%.3f\n", R0, R0+sum(abs,C)); flush(stdout)

# --- .geo: point-contact channel with the Fourier obstacle as a HOLE ---
function write_wall_geo(path; h, nobs=200)
    open(path,"w") do io
        @printf(io,"lc=%.5f;\n",h)   # built-in kernel (OpenCASCADE+spline hole hangs Alg-11)
        P=[(0.0,-W2),(0.0,-wc/2),(0.0,wc/2),(0.0,W2),(xPL,W2),(xPR,W2),(cfg.L_x,W2),
           (cfg.L_x,wc/2),(cfg.L_x,-wc/2),(cfg.L_x,-W2),(xPR,-W2),(xPL,-W2)]
        for (i,(x,y)) in enumerate(P); @printf(io,"Point(%d)={%.8f,%.8f,0,lc};\n",i,x,y); end
        for i in 1:12; @printf(io,"Line(%d)={%d,%d};\n", i, i, i==12 ? 1 : i+1); end
        println(io,"Curve Loop(1)={",join(1:12,","),"};")
        # obstacle boundary as a closed polygon (line-loop)
        pid=100; th=range(0,2pi,length=nobs+1)[1:end-1]
        for t in th; @printf(io,"Point(%d)={%.8f,%.8f,0,lc};\n",pid, xc+Robs(t)*cos(t), Robs(t)*sin(t)); pid+=1; end
        for k in 0:nobs-1; @printf(io,"Line(%d)={%d,%d};\n", 200+k, 100+k, 100+(k+1)%nobs); end
        println(io,"Curve Loop(2)={",join(200:200+nobs-1,","),"};")
        println(io,"Plane Surface(1)={1,2};")
        println(io,"Physical Surface(\"domain\")={1};")
        println(io,"Physical Curve(\"contact_source\")={2};")
        println(io,"Physical Curve(\"contact_drain\")={8};")
        println(io,"Physical Curve(\"probe_A\")={5};")
        println(io,"Physical Curve(\"probe_B\")={11};")
        println(io,"Physical Curve(\"walls\")={1,3,4,6,7,9,10,12};")
        println(io,"Physical Curve(\"obstacle\")={",join(200:200+nobs-1,","),"};")
    end
    path
end
function mesh_it(geo,inp)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
    gmsh.open(geo); gmsh.option.setNumber("Mesh.Algorithm",8)   # 11 hangs on holed domains here
    gmsh.option.setNumber("Mesh.RecombineAll",1); gmsh.option.setNumber("Mesh.RecombinationAlgorithm",1)
    gmsh.option.setNumber("Mesh.SaveGroupsOfNodes",1)
    gmsh.model.mesh.generate(2)
    _,tags,_ = gmsh.model.mesh.getElements(2); nc=sum(length.(tags))
    gmsh.write(inp); gmsh.finalize()
    @printf("    meshed (%d elements)\n", nc); flush(stdout)
    inp
end

# --- common setup: mesh the holed domain and build the semidiscretization ---
function build_semi(h)
    geo=joinpath(workdir,"wall.geo"); inp=joinpath(workdir,"wall.inp")
    write_wall_geo(geo; h=h, nobs=48); mesh_it(geo,inp)
    equations = FermiSea.IsotropicFermiHarmonics2D(sim.n_harmonics; v_fermi=sim.v_fermi)
    mesh = Trixi.P4estMesh{2}(inp; polydeg=sim.polydeg,
        boundary_symbols=[:contact_source,:contact_drain,:probe_A,:probe_B,:walls,:obstacle])
    bcA=FermiSea.FloatingProbeBC(); bcB=FermiSea.FloatingProbeBC()
    bcs=(contact_source=FermiSea.CurrentContactBC(sim.I_source),
         contact_drain=FermiSea.OhmicContactBC(0.0), probe_A=bcA, probe_B=bcB,
         walls=FermiSea.MaxwellWallBC(1.0), obstacle=FermiSea.MaxwellWallBC(1.0))
    collision=FermiSea.LinearCollisionMatrix(equations; gamma_mr=sim.gamma_mr,
        gamma_mc=sim.gamma_mc, gamma_3=sim.gamma_3)
    ic0(x,t,eq)=zero(Trixi.SVector{Trixi.nvariables(eq),Float64})
    solver=DGSEM(polydeg=sim.polydeg, surface_flux=flux_lax_friedrichs)
    semi=SemidiscretizationHyperbolic(mesh,equations,ic0,solver;
        boundary_conditions=bcs, source_terms=collision)
    _,_,dg,cache=Trixi.mesh_equations_solver_cache(semi)
    extract()=(FermiSea._current_contact_potential(bcA,equations,dg,cache)-
               FermiSea._current_contact_potential(bcB,equations,dg,cache))/sim.I_source
    ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0)
    return (; semi, extract, N, b, u0=ode.u0)
end

# --- matrix-free Tikhonov-reg GMRES: matvec rhs(v)-b+εv (the unblocking solver;
# ε cures the singular a0 nullspace AND conditions the op so GMRES converges) ---
function f1_gmres(s; eps=1e-8)
    scr=similar(s.b)
    A=LinearMap{Float64}((out,v)->(Trixi.rhs!(scr,v,s.semi,0.0); @. out=scr-s.b+eps*v; out),
                         s.N; ismutating=true)
    u,st=Krylov.gmres(A,-s.b; restart=true, memory=100, atol=1e-11, rtol=1e-10, itmax=30000)
    du=similar(u); Trixi.rhs!(du,u,s.semi,0.0)
    (f1=s.extract(), it=st.niter, res=norm(du .+ eps.*u)/max(norm(u),1), conv=st.solved)
end

# --- probing assembly + reg-LU: ground-truth cross-check (feasible at small N) ---
function f1_lu(s; eps=1e-10)
    N=s.N; Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(s.b);ej=zeros(N)
    for j in 1:N
        ej[j]=1.0; Trixi.rhs!(tmp,ej,s.semi,0.0); c[j]=s.extract(); ej[j]=0.0
        @inbounds for i in 1:N
            v=tmp[i]-s.b[i]; abs(v)>1e-12 && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
        end
    end
    A=sparse(Ir,Jc,Vv,N,N)+eps*spdiagm(0=>ones(N))
    dot(c, lu(A)\(-s.b))
end

# === A) coarse mesh: reg-LU ε-insensitivity (ground truth), and CONFIRM matrix-free
#        reg-GMRES stalls on this unstructured holed + reflecting-boundary operator
#        (it converged for the clean STRUCTURED vicinity mesh, but not here — needs a
#        preconditioner). reg-LU is therefore the solver of record for the hard wall. ===
println("\n=== A) reg-LU ε-check + GMRES-convergence probe @ coarse h=0.09 ===")
let s = build_semi(0.09)
    @printf("  N=%d\n", s.N); flush(stdout)
    for eps in (1e-8,1e-10,1e-12)
        t=time(); f1 = f1_lu(s; eps=eps)
        @printf("  reg-LU  ε=%.0e  f_1=%+.8e  (%.0fs)\n", eps, f1, time()-t); flush(stdout)
    end
    t=time(); r = f1_gmres(s; eps=1e-8)
    @printf("  GMRES   ε=1e-8   f_1=%+.6e  (it=%d res=%.0e conv=%s %.0fs)  <- expected NON-conv\n",
            r.f1, r.it, r.res, r.conv, time()-t); flush(stdout)
end

# === B) h-convergence via reg-LU (feasible: moderate obstacle keeps N≲35k).
#        No α to sweep — this is a REAL reflecting wall, so the whole α-divergence
#        class is gone by construction, as is the soft-ρ probe contamination. ===
EPS=1e-10
println("\n=== B) TRUE hard-wall (MaxwellWallBC) f_1, h-convergence (reg-LU, ε=$(EPS)) ===")
for h in (0.09, 0.075, 0.062, 0.052, 0.042, 0.033)
    t=time(); s = build_semi(h); f1 = f1_lu(s; eps=EPS)
    @printf("  h=%.3f  N=%6d  f_1=%+.6e  (%.0fs)\n", h, s.N, f1, time()-t); flush(stdout); GC.gc()
end
println("DONE"); flush(stdout)
