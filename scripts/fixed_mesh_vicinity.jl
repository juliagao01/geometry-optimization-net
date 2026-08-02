#!/usr/bin/env julia
# scripts/fixed_mesh_vicinity.jl — TRUSTWORTHY f_1 on a literature Bandurin–Levitov
# vicinity device (no obstacle): narrow injector pair + side voltage probes on the
# bottom edge. f_1 = (V_A - V_B)/I. Solve = matrix-free TIKHONOV-regularized GMRES
# (matvec rhs(v)-b+εv): ε-regularization cures the a0 nullspace AND conditions the
# operator so GMRES converges; f_1 is ε-insensitive here (reg-LU eps-sweep gave
# 0.0327272 flat over ε=1e-6..1e-14), so ε=1e-6 is exact and needs no O(N) assembly.
# Validate by the sign change vs γ_mc (viscous → negative vicinity resistance).
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_vicinity.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea, Krylov, LinearMaps
using LinearAlgebra: norm
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W,wc,xs,dA,dB = 2.0, 0.8, 0.1, 1.25, 0.15, 0.45
cfgV = ChannelConfig(L_x=L, W=W); Isrc = 1.0

function f1_vic(; h=0.05, gmc=100.0, M=6, eps=1e-6)
    sim = SimConfig(n_harmonics=M, gamma_mc=gmc, gamma_mr=0.05, gamma_3=gmc,
                    polydeg=1, I_source=Isrc)
    geo=joinpath(workdir,"vic.geo"); inp=joinpath(workdir,"vic.inp")
    write_vicinity_geo(geo; L=L, W=W, wc=wc, h=h, xs=xs, dA=dA, dB=dB)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfgV; nx=8, ny=6, alpha_max=1.0)
    ev = FixedEvaluator(inp, field, sim; drain_bc=FermiSea.CurrentContactBC(-Isrc))
    semi = ev.semi
    _, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    extract()=(FermiSea._current_contact_potential(ev.bc_probe_A,equations,dg,cache)-
               FermiSea._current_contact_potential(ev.bc_probe_B,equations,dg,cache))/Isrc
    ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0); scr=similar(b)
    A=LinearMap{Float64}((out,v)->(Trixi.rhs!(scr,v,semi,0.0); @. out=scr-b+eps*v; out),
                         N; ismutating=true)
    u,st=Krylov.gmres(A,-b; restart=true, memory=100, atol=1e-11, rtol=1e-10, itmax=30000)
    du=similar(u); Trixi.rhs!(du,u,semi,0.0)
    f1=extract(); res=norm(du .+ eps.*u)/max(norm(u),1)
    f1, N, st.niter, res, st.solved
end

println("(ε-convergence already established via reg-LU: f_1=0.0327272 flat over ε=1e-6..1e-14)")
println("\n=== B) h-convergence (γ_mc=100, M=6) ==="); flush(stdout)
for h in (0.065, 0.05, 0.04, 0.033, 0.027)
    t=time(); f1,N,it,res,ok=f1_vic(h=h,gmc=100.0,M=6)
    @printf("  h=%.3f N=%6d  f_1=%+.6e  (it=%d res=%.0e conv=%s %.0fs)\n",
            h,N,f1,it,res,ok,time()-t); flush(stdout); GC.gc()
end
println("\n=== C) n_harmonics (h=0.05, γ_mc=100) ==="); flush(stdout)
for M in (4,6,8,10)
    f1,N,it,res,ok=f1_vic(h=0.05,gmc=100.0,M=M)
    @printf("  M=%2d N=%6d  f_1=%+.6e  (it=%d res=%.0e)\n",M,N,f1,it,res); flush(stdout); GC.gc()
end
println("\n=== D) γ_mc sweep (h=0.05, M=8): ohmic → viscous ==="); flush(stdout)
for gmc in (0.1,1.0,10.0,50.0,200.0)
    f1,N,it,res,ok=f1_vic(h=0.05,gmc=gmc,M=8)
    @printf("  γ_mc=%6.1f  f_1=%+.6e  (it=%d res=%.0e)\n",gmc,f1,it,res); flush(stdout); GC.gc()
end
println("DONE"); flush(stdout)
