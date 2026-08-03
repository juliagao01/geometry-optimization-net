#!/usr/bin/env julia
# scripts/fixed_mesh_f2conv.jl — MESH-CONVERGENCE of the second-order coefficient f_2.
# Fix the nonlinearity strength λ (leading-order regime, so f_2/λ is the λ-independent
# target) and refine h. f_2 is trustworthy iff f_2/λ settles with a shrinking
# successive change (the same metric that pinned f_1 ≈ 0.020). At each h we solve the
# nonlinear steady state at 5 currents and fit ΔV(I)=f1·I+f2·I²+f3·I³.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_f2conv.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using LinearAlgebra: dot
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W,wc,xs,dA,dB = 2.0, 0.8, 0.1, 1.25, 0.15, 0.45
cfgV = ChannelConfig(L_x=L, W=W)
GMC, LAMBDA, MH = 100.0, 1e-2, 4
Ivals = [0.25, 0.5, 1.0, 2.0, 4.0]

function f2_at(h)
    sim = SimConfig(n_harmonics=MH, gamma_mc=GMC, gamma_mr=0.05, gamma_3=GMC,
                    polydeg=1, I_source=1.0)
    geo=joinpath(workdir,"f2c.geo"); inp=joinpath(workdir,"f2c.inp")
    write_vicinity_geo(geo; L=L, W=W, wc=wc, h=h, xs=xs, dA=dA, dB=dB)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfgV; nx=8, ny=6, alpha_max=1.0)
    drain = FermiSea.CurrentContactBC(-1.0)
    op = assemble_fixed_operator(FixedEvaluator(inp, field, sim; drain_bc=drain))
    eqs = FermiSea.IsotropicFermiHarmonics2D(MH)
    ev_nl = FixedEvaluator(inp, field, sim; drain_bc=drain,
                           extra_sources=(FermiSea.InertialStressSource(eqs, LAMBDA),))
    dVs = Float64[]; mit=0; mres=0.0
    for I in Ivals
        u,res,it,cv = solve_nonlinear_steady(op, ev_nl; Iscale=I, eps=1e-8, maxit=300, tol=1e-10)
        push!(dVs, dot(op.c, u)); mit=max(mit,it); mres=max(mres,res)
    end
    Vm = hcat(Ivals, Ivals.^2, Ivals.^3)
    f1,f2,f3 = Vm \ dVs
    (; f1, f2, f2λ=f2/LAMBDA, N=op.N, mit, mres)
end

function run_conv()
    @printf("f_2 mesh-convergence: vicinity device, λ=%.0e, M=%d, γ_mc=%.0f\n", LAMBDA, MH, GMC)
    println("   h       N       f_1           f_2/λ          Δ(f2/λ)     ratio   chord(it,res)")
    prev = NaN; prevd = NaN
    for h in (0.08, 0.065, 0.052, 0.042, 0.034, 0.028)
        t=time(); r = f2_at(h)
        d = isnan(prev) ? NaN : r.f2λ - prev
        ratio = (isnan(prevd) || prevd==0) ? NaN : d/prevd
        @printf("  %.3f  %6d  %+.6e  %+.6e  %+.3e  %+.3f  (%d, %.0e)  %.0fs\n",
                h, r.N, r.f1, r.f2λ, d, ratio, r.mit, r.mres, time()-t)
        flush(stdout); prevd = d; prev = r.f2λ; GC.gc()
    end
end
run_conv()
println("DONE"); flush(stdout)
