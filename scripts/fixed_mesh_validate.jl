#!/usr/bin/env julia
# scripts/fixed_mesh_validate.jl
#
#   julia --threads=auto --project=. scripts/fixed_mesh_validate.jl
#
# Validates the fixed-mesh / density-based obstacle before optimizing:
#   - empty channel (ρ = 0)     -> f_1 ≈ 0 (no obstacle, up-down symmetric)
#   - centered circular blob     -> f_1 ≈ 0 (obstacle is y-symmetric)
#   - off-center blob (up)       -> f_1 > 0 (breaks symmetry toward probe A)
#   - off-center blob (down)     -> f_1 < 0 (mirror image)
# If the last two don't come out with opposite signs and the centered ones
# near zero, the Brinkman coupling or the coordinate lookup is wrong.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using VicinityOpt
using VicinityOpt.Geometry
using VicinityOpt.Simulate
using VicinityOpt.FixedMesh
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.05)

# Advisor's stabilizing params. polydeg=1 for speed.
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05,
                gamma_3=300.0, polydeg=1, t_end=2000.0)

workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
mkpath(workdir)

inp = joinpath(workdir, "channel.inp")
geo = joinpath(workdir, "channel.geo")
write_channel_geo(geo, cfg)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false)
println("meshed empty channel -> ", inp); flush(stdout)

field = DensityField(cfg; nx=30, ny=18, alpha_max=500.0)
ev = FixedEvaluator(inp, field, sim)
println("built evaluator (n_harmonics=$(sim.n_harmonics), polydeg=$(sim.polydeg))")

function report(label)
    f1, info = run_f1_fixed(ev)
    @printf("%-22s f_1 = %+.5e   (V_A=%+.4e V_B=%+.4e  %.1fs, %d steps, t=%.1f)\n",
            label, f1, info.V_A, info.V_B, info.walltime, info.steps, info.t_final)
    flush(stdout)
    return f1
end

println("\n=== validation ===")
clear!(field)
f_empty = report("empty channel")

paint_blob!(field, 0.5, 0.0, 0.15)
f_center = report("centered r=0.15")

paint_blob!(field, 0.5, +0.12, 0.12)
f_up = report("up   yc=+0.12 r=0.12")

paint_blob!(field, 0.5, -0.12, 0.12)
f_down = report("down yc=-0.12 r=0.12")

println("\n=== checks ===")
@printf("empty  |f_1| = %.2e  (want ~0)\n", abs(f_empty))
@printf("center |f_1| = %.2e  (want ~0)\n", abs(f_center))
@printf("up/down antisymmetry: f_up=%+.3e f_down=%+.3e  (want opposite signs)\n",
        f_up, f_down)
ok = abs(f_empty) < 1e-3 && abs(f_center) < 1e-3 &&
     f_up > 1e-4 && f_down < -1e-4
println(ok ? "PASS" : "CHECK: physics did not match expectation")
