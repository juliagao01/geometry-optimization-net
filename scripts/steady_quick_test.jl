using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Mesh, VicinityOpt.Simulate

# Tiny problem: coarse mesh, few harmonics
cfg = ChannelConfig(n_modes=0, lc=0.10, margin=0.02)
o = ObstacleParams(0.05, 0.23, Float64[], Float64[])
write_geo("sq.geo", o, cfg)
geo_to_inp("sq.geo", "sq.inp"; verbose=false)

sim = SimConfig(n_harmonics=4, gamma_mc=100.0, polydeg=2, residual_tol=1e-6)

println("GMRES steady-state solve (tiny problem):")
@time f1, info = run_f1_steady("sq.inp", sim)
println("f_1 = $f1")
println("converged = $(info.converged), iters = $(info.gmres_iters), residual = $(info.residual)")
