using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Mesh, VicinityOpt.Simulate

cfg = ChannelConfig(n_modes=0, lc=0.08, margin=0.02)
o = ObstacleParams(0.05, 0.20, Float64[], Float64[])
write_geo("st.geo", o, cfg); geo_to_inp("st.geo", "st.inp"; verbose=false)

# His suggested stabilizing params, verbose so we see if the callback fires
sim = SimConfig(n_harmonics=4, gamma_mc=300.0, gamma_mr=0.05,
                polydeg=2, t_end=2000.0, verbose=true)

println("solving to steady state...")
@time f1, info = run_f1("st.inp", sim)
@show f1
@show info.steps
@show info.walltime
