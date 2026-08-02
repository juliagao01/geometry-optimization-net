#!/usr/bin/env julia
# scripts/fixed_mesh_nonlinear.jl — END-TO-END NONLINEAR PIPELINE + f_2 extraction.
# Adds the InertialStressSource (local quadratic j⊗j → stress coupling, the algebraic
# part of the convective (u·∇)u term) to the vicinity device, solves the resulting
# NONLINEAR steady state with a modified-Newton (chord) iteration reusing one reg-LU
# factorization, and extracts the response coefficients ΔV(I)=f1·I+f2·I²+f3·I³.
#
# Validation (the whole point):
#   (A) λ=0 reproduces the linear null-test: f_2 ≈ machine noise.
#   (B) λ>0 gives a NONZERO, chord-converged f_2.
#   (C) f_2 ∝ λ for small λ (f_2/λ → const): the nonlinearity IS the source of f_2.
#   (D) f_2 is mesh-convergent (small h-sweep).
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_nonlinear.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using LinearAlgebra: dot
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W,wc,xs,dA,dB = 2.0, 0.8, 0.1, 1.25, 0.15, 0.45
cfgV = ChannelConfig(L_x=L, W=W)
GMC = 100.0
Ivals = [0.25, 0.5, 1.0, 2.0, 4.0]

# Build linear operator (once per mesh) + a nonlinear evaluator at strength λ.
function build(h, M)
    sim = SimConfig(n_harmonics=M, gamma_mc=GMC, gamma_mr=0.05, gamma_3=GMC,
                    polydeg=1, I_source=1.0)
    geo=joinpath(workdir,"nl.geo"); inp=joinpath(workdir,"nl.inp")
    write_vicinity_geo(geo; L=L, W=W, wc=wc, h=h, xs=xs, dA=dA, dB=dB)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfgV; nx=8, ny=6, alpha_max=1.0)         # ρ≡0 (no obstacle)
    drain = FermiSea.CurrentContactBC(-1.0)
    ev_lin = FixedEvaluator(inp, field, sim; drain_bc=drain)      # linear: for A0, b, c
    op = assemble_fixed_operator(ev_lin)
    eqs = FermiSea.IsotropicFermiHarmonics2D(M)
    return sim, inp, field, drain, op, eqs
end

# Fit ΔV(I)=f1·I+f2·I²+f3·I³ (through origin) at nonlinearity strength λ.
function coeffs_at(op, inp, field, sim, drain, eqs, λ)
    ev_nl = FixedEvaluator(inp, field, sim; drain_bc=drain,
                           extra_sources=(FermiSea.InertialStressSource(eqs, λ),))
    dVs = Float64[]; maxit = 0; maxres = 0.0; allconv = true
    for I in Ivals
        u,res,it,conv = solve_nonlinear_steady(op, ev_nl; Iscale=I, eps=1e-8,
                                               maxit=300, tol=1e-10)
        push!(dVs, dot(op.c, u))                 # raw ΔV (op.c is the raw functional, I_source=1)
        maxit = max(maxit,it); maxres = max(maxres,res); allconv &= conv
    end
    Vm = hcat(Ivals, Ivals.^2, Ivals.^3)
    f1,f2,f3 = Vm \ dVs
    resid = maximum(abs, Vm*[f1,f2,f3] .- dVs)
    (; f1,f2,f3, fitresid=resid, maxit, maxres, allconv)
end

println("=== NONLINEAR f_2 pipeline (vicinity device + InertialStressSource) ===")
@printf("mesh h=0.07, M=4, γ_mc=%.0f ; currents I=%s\n", GMC, Ivals); flush(stdout)
sim, inp, field, drain, op, eqs = build(0.07, 4)
@printf("linear operator assembled: N=%d\n\n", op.N); flush(stdout)

println("=== λ sweep: f_2 vs nonlinearity strength ===")
println("   λ          f_1            f_2            f_2/λ         |f2|/|f1|   fit_res   chord(it,res,conv)")
for λ in (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    r = coeffs_at(op, inp, field, sim, drain, eqs, λ)
    f2overλ = λ == 0 ? NaN : r.f2/λ
    @printf("  %.1e   %+.6e   %+.6e   %+.4e   %.2e   %.1e   (%d, %.0e, %s)\n",
            λ, r.f1, r.f2, f2overλ, abs(r.f2)/abs(r.f1), r.fitresid, r.maxit, r.maxres, r.allconv)
    flush(stdout); GC.gc()
end

println("\n=== (D) h-convergence of f_2 at fixed λ=1e-2 ===")
for h in (0.09, 0.07, 0.055)
    s,i2,fld,dr,op2,eq2 = build(h, 4)
    r = coeffs_at(op2, i2, fld, s, dr, eq2, 1e-2)
    @printf("  h=%.3f N=%6d  f_2=%+.6e  (f2/λ=%+.4e, chord it=%d res=%.0e)\n",
            h, op2.N, r.f2, r.f2/1e-2, r.maxit, r.maxres); flush(stdout); GC.gc()
end
println("DONE"); flush(stdout)
