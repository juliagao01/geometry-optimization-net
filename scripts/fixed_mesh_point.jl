#!/usr/bin/env julia
# scripts/fixed_mesh_point.jl — vicinity geometry with NARROW point contacts.
#   julia --threads=auto --project=. scripts/fixed_mesh_point.jl
#
# The full-edge source/drain drove uniform 1D flow -> no vicinity signal (f_1
# ill-posed ~0). Here source/drain are narrow contacts centered on the left/right
# walls, so current spreads as a genuine 2D flow that an obstacle can deflect.
# Built as a 3x3 grid of transfinite blocks with a y-symmetric partition, so the
# mesh is exactly mirror-symmetric (centered obstacle -> f_1=0 to machine eps).
# Decisive test: does an off-center obstacle give a ROBUST (mesh- and
# n_harmonics-converged, sign-antisymmetric) nonzero f_1?
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Printf

# ---- 3x3 transfinite block mesh with narrow contacts -------------------------
function write_point_geo(path; L=1.0, W=0.6, xPL=0.425, xPR=0.575,
                          wc=0.12, h=0.03)
    xs = (0.0, xPL, xPR, L)
    ys = (-W/2, -wc/2, wc/2, W/2)
    Nx = ntuple(i -> max(2, round(Int,(xs[i+1]-xs[i])/h)+1), 3)
    Ny = ntuple(j -> max(2, round(Int,(ys[j+1]-ys[j])/h)+1), 3)
    pid(i,j) = (i-1)*4 + j
    H(i,j) = (j-1)*3 + i          # horizontal line P(i,j)->P(i+1,j), i∈1..3 j∈1..4
    V(i,j) = 12 + (i-1)*3 + j     # vertical line   P(i,j)->P(i,j+1), i∈1..4 j∈1..3
    open(path,"w") do io
        @printf(io,"lc=%.6f;\n", h)
        for i in 1:4, j in 1:4
            @printf(io,"Point(%d)={%.10f,%.10f,0,lc};\n", pid(i,j), xs[i], ys[j])
        end
        for j in 1:4, i in 1:3
            @printf(io,"Line(%d)={%d,%d};\n", H(i,j), pid(i,j), pid(i+1,j))
        end
        for i in 1:4, j in 1:3
            @printf(io,"Line(%d)={%d,%d};\n", V(i,j), pid(i,j), pid(i,j+1))
        end
        sid=0
        for i in 1:3, j in 1:3
            sid+=1
            @printf(io,"Curve Loop(%d)={%d,%d,%d,%d};\n", sid,
                    H(i,j), V(i+1,j), -H(i,j+1), -V(i,j))
            @printf(io,"Plane Surface(%d)={%d};\n", sid, sid)
        end
        # transfinite: horizontal seg i -> Nx[i]; vertical seg j -> Ny[j]
        for i in 1:3, j in 1:4; @printf(io,"Transfinite Curve{%d}=%d;\n",H(i,j),Nx[i]); end
        for i in 1:4, j in 1:3; @printf(io,"Transfinite Curve{%d}=%d;\n",V(i,j),Ny[j]); end
        for s in 1:9; @printf(io,"Transfinite Surface{%d};\n", s); end
        print(io,"Recombine Surface{"); print(io, join(1:9,",")); println(io,"};")
        # physical groups
        println(io,"Physical Surface(\"domain\")={",join(1:9,","),"};")
        @printf(io,"Physical Curve(\"contact_source\")={%d};\n", V(1,2))  # left, centered
        @printf(io,"Physical Curve(\"contact_drain\") ={%d};\n", V(4,2))  # right, centered
        @printf(io,"Physical Curve(\"probe_A\")={%d};\n", H(2,4))         # top, centered x
        @printf(io,"Physical Curve(\"probe_B\")={%d};\n", H(2,1))         # bottom
        walls = [H(1,1),H(3,1),H(1,4),H(3,4),V(1,1),V(1,3),V(4,1),V(4,3)]
        println(io,"Physical Curve(\"walls\")={",join(walls,","),"};")
    end
    return path
end

cfg = ChannelConfig(L_x=1.0, W=0.6)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir,"channel_point.geo"); inp = joinpath(workdir,"channel_point.inp")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.03)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
println("built point-contact mesh"); flush(stdout)

testone(M, alpha, yc, r) = begin
    sim = SimConfig(n_harmonics=M, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    field = DensityField(cfg; nx=40, ny=24, alpha_max=alpha)
    ev = FixedEvaluator(inp, field, sim)
    r > 0 ? paint_blob!(field, cfg.x_c, yc, r) : clear!(field)
    f1, info = run_f1_fixed_steady(ev; itmax=25000, memory=100, atol=1e-12, rtol=1e-11)
    f1, info
end

println("\n=== sanity: empty & centered (expect ~0) ==="); flush(stdout)
for (lab,yc,r) in (("empty",0.0,0.0),("centered r=0.16",0.0,0.16))
    f1,info = testone(6, 2000.0, yc, r)
    @printf("  %-16s f_1=%+.5e (res=%.1e conv=%s)\n", lab, f1, info.residual, info.converged); flush(stdout)
end

println("\n=== off-center obstacle, n_harmonics convergence (alpha=2000) ==="); flush(stdout)
for M in (2,4,6,8,10)
    f1,info = testone(M, 2000.0, 0.10, 0.16)
    @printf("  M=%2d  f_1=%+.5e (res=%.1e conv=%s)\n", M, f1, info.residual, info.converged); flush(stdout)
end

println("\n=== antisymmetry & solidity (M=6) ==="); flush(stdout)
for a in (200.0, 2000.0)
    fp,_ = testone(6, a, +0.10, 0.16)
    fm,_ = testone(6, a, -0.10, 0.16)
    @printf("  alpha=%5.0f  f(+)= %+.5e  f(-)= %+.5e  odd=%+.5e sum=%.1e\n",
            a, fp, fm, (fp-fm)/2, fp+fm); flush(stdout)
end
println("DONE"); flush(stdout)
