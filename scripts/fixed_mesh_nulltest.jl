#!/usr/bin/env julia
# scripts/fixed_mesh_nulltest.jl — LINEAR-NULL-TEST for higher-order coefficients.
# Solve the (obstacle-free) vicinity device at several injected-current amplitudes I,
# read the RAW probe drop ΔV(I) = V_A − V_B (NOT divided by I), and fit
#     ΔV(I) = f_1·I + f_2·I² + f_3·I³ + …
# Because FermiSea's model is strictly linear (rhs(u)=A·u+b, b∝I ⇒ u∝I), this MUST
# return f_2, f_3 ≈ machine/solver-residual noise and f_1 = const. Purposes:
#   (1) empirically confirm the current code cannot produce a nonzero f_2;
#   (2) validate the multi-amplitude extraction+fit harness against a known-zero,
#       so any nonzero curvature seen AFTER a nonlinearity is added is real physics.
# Deliberately coarse mesh — resolution is irrelevant to an exact-zero result.
#   julia --threads=2 --gcthreads=1 --heap-size-hint=6G --project=. scripts/fixed_mesh_nulltest.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea, Krylov, LinearMaps
using LinearAlgebra: norm
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
L,W,wc,xs,dA,dB = 2.0, 0.8, 0.1, 1.25, 0.15, 0.45
cfgV = ChannelConfig(L_x=L, W=W)
H, MHARM, GMC, EPS = 0.07, 4, 100.0, 1e-6   # coarse on purpose

# RAW probe drop ΔV = V_A − V_B (NOT normalized by I) at injected current Isrc.
function dV_raw(Isrc)
    sim = SimConfig(n_harmonics=MHARM, gamma_mc=GMC, gamma_mr=0.05, gamma_3=GMC,
                    polydeg=1, I_source=Isrc)
    geo=joinpath(workdir,"null.geo"); inp=joinpath(workdir,"null.inp")
    write_vicinity_geo(geo; L=L, W=W, wc=wc, h=H, xs=xs, dA=dA, dB=dB)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
    field = DensityField(cfgV; nx=8, ny=6, alpha_max=1.0)
    ev = FixedEvaluator(inp, field, sim; drain_bc=FermiSea.CurrentContactBC(-Isrc))
    semi = ev.semi
    _, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    # RAW potentials — no division by Isrc
    dVof()=FermiSea._current_contact_potential(ev.bc_probe_A,equations,dg,cache)-
           FermiSea._current_contact_potential(ev.bc_probe_B,equations,dg,cache)
    ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
    b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0); scr=similar(b)
    A=LinearMap{Float64}((out,v)->(Trixi.rhs!(scr,v,semi,0.0); @. out=scr-b+EPS*v; out),
                         N; ismutating=true)
    u,st=Krylov.gmres(A,-b; restart=true, memory=100, atol=1e-11, rtol=1e-10, itmax=30000)
    du=similar(u); Trixi.rhs!(du,u,semi,0.0)
    dVof(), N, st.niter, norm(du .+ EPS.*u)/max(norm(u),1), st.solved
end

Ivals = [0.25, 0.5, 1.0, 2.0, 4.0]
@printf("Linear-null-test on vicinity device (h=%.3f, M=%d, γ_mc=%.0f, ε=%.0e)\n", H,MHARM,GMC,EPS)
println("  I        ΔV(raw)          f_1=ΔV/I         (GMRES it/res/conv)"); flush(stdout)
dVs = Float64[]
for I in Ivals
    dv,N,it,res,ok = dV_raw(I); push!(dVs, dv)
    @printf("  %5.2f   %+.10e   %+.10e   (it=%d res=%.0e %s)\n", I, dv, dv/I, it, res, ok)
    flush(stdout); GC.gc()
end

# Fit ΔV = f1·I + f2·I² + f3·I³ through the origin (least squares, 5 pts / 3 coeffs).
Vm = hcat(Ivals, Ivals.^2, Ivals.^3)           # 5×3 Vandermonde (no constant term)
coef = Vm \ dVs                                 # [f1, f2, f3]
resid = Vm*coef .- dVs
f1,f2,f3 = coef
@printf("\n=== FIT  ΔV(I) = f1·I + f2·I² + f3·I³  (through origin) ===\n")
@printf("  f_1 = %+.10e\n", f1)
@printf("  f_2 = %+.10e   |f_2|/|f_1| = %.2e\n", f2, abs(f2)/max(abs(f1),eps()))
@printf("  f_3 = %+.10e   |f_3|/|f_1| = %.2e\n", f3, abs(f3)/max(abs(f1),eps()))
@printf("  fit residual (max) = %.2e ;  spread of ΔV/I = %.2e\n",
        maximum(abs,resid), maximum(dVs./Ivals)-minimum(dVs./Ivals))
verdict = abs(f2)/max(abs(f1),eps()) < 1e-4 ? "PASS — f_2 at noise floor: model is LINEAR, f_2≡0 as predicted" :
                                              "UNEXPECTED — nonzero curvature (investigate)"
println("  VERDICT: ", verdict)
println("DONE"); flush(stdout)
