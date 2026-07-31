#!/usr/bin/env julia
# Print the optimized obstacle (density field) as an ASCII map + summary.
#   julia --project=. scripts/fixed_mesh_show.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Printf
f = joinpath(@__DIR__, "..", "runs", "fixed_mesh", "result_steady.jld2")
d = load(f)
rho = d["rho"]; cfg = d["cfg"]
@printf("BEST: y_c=%+.3f  r=%.3f  f_1(physical)=%+.5e\n",
        d["best_yc"], d["best_r2"], d["best_f1"])
nx, ny = size(rho)
chars = " .:-=+*#%@"
println("\nObstacle density (top = +W/2, channel is $(cfg.L_x) x $(cfg.W)):")
for j in ny:-1:1
    print("  ")
    for i in 1:nx
        k = clamp(floor(Int, rho[i,j]*(length(chars)-1))+1, 1, length(chars))
        print(chars[k])
    end
    println()
end
println("  (source=left edge, drain=right edge, probe_A=top, probe_B=bottom)")
