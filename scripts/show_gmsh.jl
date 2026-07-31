#!/usr/bin/env julia
# scripts/show_gmsh.jl — pop up the optimized obstacle in the Gmsh GUI.
#   julia --project=. scripts/show_gmsh.jl
#
# The fixed-mesh method's "shape" is the optimized density field rho(x,y) on the
# point-contact channel mesh. We build the structured mesh (Gmsh), overlay the
# density as a post-processing View (.pos), and open the interactive Gmsh window
# (gmsh.fltk.run(), the same GUI FermiSea's mesh tutorial pops up).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt.Geometry
using JLD2, Printf
using Gmsh: gmsh
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
bestphys = load(joinpath(workdir,"result_point_opt.jld2"))["bestphys"]
nx, ny = size(bestphys)
x0, y0 = 0.0, -cfg.W/2
dx, dy = cfg.L_x/nx, cfg.W/ny

# --- write the density field as a Gmsh post-processing view (.pos) ---
pos = joinpath(workdir, "obstacle.pos")
open(pos, "w") do io
    println(io, "View \"obstacle density\" {")
    for j in 1:ny, i in 1:nx
        xa=x0+(i-1)*dx; xb=x0+i*dx; ya=y0+(j-1)*dy; yb=y0+j*dy
        v=bestphys[i,j]
        # SQ: scalar quad, 4 corners CCW, one value per corner
        @printf(io, "SQ(%.6f,%.6f,0,%.6f,%.6f,0,%.6f,%.6f,0,%.6f,%.6f,0){%.5f,%.5f,%.5f,%.5f};\n",
                xa,ya, xb,ya, xb,yb, xa,yb, v,v,v,v)
    end
    println(io, "};")
end

# --- build the structured point-contact mesh + show ---
geo = joinpath(workdir, "cp_show.geo")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.03)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.open(geo)
gmsh.model.mesh.generate(2)
gmsh.merge(pos)
# nice display: show the density view, thin mesh lines
gmsh.option.setNumber("Mesh.SurfaceEdges", 1)
gmsh.option.setNumber("Mesh.SurfaceFaces", 0)
gmsh.option.setNumber("View[0].ColormapNumber", 2)   # viridis-ish
gmsh.option.setNumber("View[0].ShowScale", 1)
println("Opening Gmsh window — close it to end. (obstacle density on the point-contact channel)")
gmsh.fltk.run()
gmsh.finalize()
