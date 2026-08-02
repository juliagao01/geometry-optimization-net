#!/usr/bin/env julia
# scripts/fixed_mesh_vicinity_neg.jl — canonical Bandurin–Levitov NEGATIVE-vicinity
# test. Single narrow injector (source) with the drain placed FAR away (xd near the
# right end); both voltage probes sit to the LEFT of the injector in the current-free
# "vicinity" region. f_1=(V_A-V_B)/I should turn NEGATIVE in the hydrodynamic regime
# (viscous backflow), unlike the adjacent-injector-pair layout (local dipole, always +).
# Matrix-free Tikhonov-reg GMRES (ε=1e-6, ε-insensitive). Sweep γ_mc: ballistic→viscous→ohmic.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_vicinity_neg.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea, Krylov, LinearMaps
using LinearAlgebra: norm
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W,wc = 2.0, 0.8, 0.1
xs, dA, dB, xd = 0.6, 0.15, 0.45, 1.7   # injector left-of-center; probes further left; drain far right
cfgV = ChannelConfig(L_x=L, W=W); Isrc = 1.0

function f1_vic(; h=0.05, gmc=100.0, gmr=0.05, M=8, eps=1e-6)
    sim = SimConfig(n_harmonics=M, gamma_mc=gmc, gamma_mr=gmr, gamma_3=gmc,
                    polydeg=1, I_source=Isrc)
    geo=joinpath(workdir,"vicn.geo"); inp=joinpath(workdir,"vicn.inp")
    write_vicinity_geo(geo; L=L, W=W, wc=wc, h=h, xs=xs, dA=dA, dB=dB, xd=xd)
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

@printf("device: L=%.1f W=%.1f  source@[%.2f,%.2f] drain@[%.2f,%.2f]  probeA@[%.2f,%.2f] probeB@[%.2f,%.2f]\n",
        L,W, xs,xs+wc, xd,xd+wc, xs-dA-wc,xs-dA, xs-dB-wc,xs-dB); flush(stdout)
println("\n=== γ_mc sweep (h=0.05, M=8, γ_mr=0.05): ballistic → viscous → ohmic ==="); flush(stdout)
for gmc in (0.1,0.5,2.0,10.0,50.0,200.0)
    t=time(); f1,N,it,res,ok=f1_vic(h=0.05,gmc=gmc,M=8)
    @printf("  γ_mc=%6.1f  f_1=%+.6e  (N=%d it=%d res=%.0e %.0fs)\n",gmc,f1,N,it,res,time()-t)
    flush(stdout); GC.gc()
end
println("DONE"); flush(stdout)
