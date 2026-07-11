using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Mesh, VicinityOpt.Simulate

# Small/cheap config so the time-stepped reference is affordable
cfg = ChannelConfig(n_modes=0, lc=0.08, margin=0.02)
o = ObstacleParams(0.05, 0.23, Float64[], Float64[])
write_geo("sv.geo", o, cfg)
geo_to_inp("sv.geo", "sv.inp"; verbose=false)

sim = SimConfig(n_harmonics=4, gamma_mc=50.0, polydeg=2,
                t_end=200.0, residual_tol=1e-6)

println("=== direct steady-state (GMRES) ===")
@time f1_direct, info_d = run_f1_steady("sv.inp", sim)
println("f_1 = $f1_direct")
println("gmres iters = $(info_d.gmres_iters), converged = $(info_d.converged), residual = $(info_d.residual)")

println("\n=== time-stepped reference (t_end=200, may take a while) ===")
@time f1_ts, info_t = run_f1("sv.inp", sim)
println("f_1 = $f1_ts")

println("\nagreement: ", isapprox(f1_direct, f1_ts; rtol=1e-2) ? "YES" : "NO",
        "  (direct=$f1_direct vs time-stepped=$f1_ts)")
