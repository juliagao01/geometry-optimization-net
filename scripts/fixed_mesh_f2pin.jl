#!/usr/bin/env julia
# scripts/fixed_mesh_f2pin.jl — PIN DOWN f_2 via matrix-free perturbation theory.
# f_2 = ΔV(u2) with A0 u2 = -S_nl(u1), A0 u1 = -b  (two reg-GMRES solves, no assembly,
# no O(N²) bottleneck) — so we can refine the mesh to nail f_2 to few %. Reports
# f_2/λ (λ=1 in the source ⇒ f_2 is already λ-normalized). Section A: ε-insensitivity;
# Section B: h-convergence to fine meshes.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_f2pin.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W,wc,xs,dA,dB = 2.0, 0.8, 0.1, 1.25, 0.15, 0.45
cfgV = ChannelConfig(L_x=L, W=W)
GMC, MH = 100.0, 4

function evals_at(h)
    sim = SimConfig(n_harmonics=MH, gamma_mc=GMC, gamma_mr=0.05, gamma_3=GMC,
                    polydeg=1, I_source=1.0)
    geo=joinpath(workdir,"f2p.geo"); inp=joinpath(workdir,"f2p.inp")
    write_vicinity_geo(geo; L=L, W=W, wc=wc, h=h, xs=xs, dA=dA, dB=dB)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfgV; nx=8, ny=6, alpha_max=1.0)
    drain = FermiSea.CurrentContactBC(-1.0)
    eqs = FermiSea.IsotropicFermiHarmonics2D(MH)
    ev_lin = FixedEvaluator(inp, field, sim; drain_bc=drain)
    ev_nl  = FixedEvaluator(inp, field, sim; drain_bc=drain,
                            extra_sources=(FermiSea.InertialStressSource(eqs, 1.0),))  # λ=1 ⇒ f2/λ
    return ev_lin, ev_nl
end

function run_pin()
    println("=== A) ε-insensitivity of f_2 (h=0.05) ===")
    el, en = evals_at(0.05)
    for eps in (1e-4, 1e-6, 1e-8)
        f1,f2,info = f2_perturbation(el, en; eps=eps)
        @printf("  ε=%.0e  f_1=%+.6e  f_2/λ=%+.6e  (N=%d, GMRES it %d/%d, conv %s/%s)\n",
                eps, f1, f2, info.N, info.it1, info.it2, info.conv1, info.conv2); flush(stdout)
    end
    println("\n=== B) h-convergence of f_2/λ (ε=1e-6) ===")
    println("   h       N       f_1           f_2/λ          Δ(f2/λ)    ratio")
    prev = NaN; prevd = NaN
    for h in (0.065, 0.052, 0.042, 0.033, 0.027, 0.022)
        t=time(); el2,en2 = evals_at(h); f1,f2,info = f2_perturbation(el2, en2; eps=1e-6)
        d = isnan(prev) ? NaN : f2 - prev
        r = (isnan(prevd) || prevd==0) ? NaN : d/prevd
        @printf("  %.3f  %6d  %+.6e  %+.6e  %+.3e  %+.3f  (it %d/%d %.0fs)\n",
                h, info.N, f1, f2, d, r, info.it1, info.it2, time()-t)
        flush(stdout); prevd=d; prev=f2; GC.gc()
    end
end
run_pin()
println("DONE"); flush(stdout)
