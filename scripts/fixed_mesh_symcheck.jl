#!/usr/bin/env julia
# scripts/fixed_mesh_symcheck.jl
#   julia --threads=auto --project=. scripts/fixed_mesh_symcheck.jl
#
# Diagnose + defeat the discrete symmetry-breaking. The true f_1 is ODD in y_c
# (f_1(-y_c) = -f_1(y_c)); any nonzero f_1 at y_c=0 is discretization noise
# (the quad mesh isn't perfectly mirror-symmetric). We sweep y_c symmetrically
# at fixed r with a TIGHT solve, then split f_1 into:
#     odd  (physical)  = [f_1(+y_c) - f_1(-y_c)] / 2
#     even (spurious)  = [f_1(+y_c) + f_1(-y_c)] / 2
# If the odd part is clean and monotone while the even part ~ the y_c=0 value,
# antisymmetrizing recovers the real signal on this fixed mesh.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.06)
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05, gamma_3=300.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel.geo"); inp = joinpath(workdir, "channel.inp")
write_channel_geo(geo, cfg); VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false)

field = DensityField(cfg; nx=30, ny=18, alpha_max=100.0)
ev = FixedEvaluator(inp, field, sim)

f1s = Dict{Tuple{Float64,Float64},Float64}()
function ev1(yc, r)
    paint_blob!(field, cfg.x_c, yc, r)
    f1, info = run_f1_fixed_steady(ev; itmax=8000, memory=60, atol=1e-9, rtol=1e-8)
    f1s[(yc,r)] = f1
    @printf("  yc=%+.3f r=%.3f  f_1=%+.5e  res=%.1e iters=%d %.0fs\n",
            yc, r, f1, info.residual, info.iters, info.walltime); flush(stdout)
    f1
end

for r in (0.12, 0.16)
    println("\n=== r=$r : symmetric y_c sweep (tight solve) ==="); flush(stdout)
    for yc in (0.0, 0.04, 0.08, 0.12, 0.16, -0.04, -0.08, -0.12, -0.16)
        yc + r <= cfg.W/2 - 0.02 + 1e-9 || continue
        abs(yc) - r >= -(cfg.W/2 - 0.02) - 1e-9 || continue
        ev1(yc, r)
    end
    println("  --- odd (physical) vs even (spurious) ---")
    for yc in (0.04, 0.08, 0.12, 0.16)
        haskey(f1s,(yc,r)) && haskey(f1s,(-yc,r)) || continue
        fp = f1s[(yc,r)]; fm = f1s[(-yc,r)]
        @printf("  |yc|=%.3f  odd=%+.5e  even=%+.5e  (f0=%+.5e)\n",
                yc, (fp-fm)/2, (fp+fm)/2, get(f1s,(0.0,r),NaN)); flush(stdout)
    end
end
println("DONE"); flush(stdout)
