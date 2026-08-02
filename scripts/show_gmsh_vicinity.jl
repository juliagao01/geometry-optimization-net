#!/usr/bin/env julia
# scripts/show_gmsh_vicinity.jl — GUI of the Bandurin–Levitov vicinity device:
# structured mesh with the injector pair + side voltage probes on the bottom edge
# (contacts colored by physical group). No obstacle.
#   julia --project=. scripts/show_gmsh_vicinity.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt.Geometry, VicinityOpt.FixedMesh
using Gmsh: gmsh
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo=joinpath(workdir,"vic_show.geo")
write_vicinity_geo(geo; L=2.0, W=0.8, wc=0.1, h=0.04, xs=1.25, dA=0.15, dB=0.45)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo)
gmsh.model.mesh.generate(2)                       # structured (transfinite in .geo)
gmsh.option.setNumber("Mesh.SurfaceEdges",1)      # show the grid
gmsh.option.setNumber("Mesh.SurfaceFaces",0)
gmsh.option.setNumber("Mesh.ColorCarousel",2)     # color 1D elements by physical group
gmsh.option.setNumber("Mesh.LineWidth",5)         # fat contact lines
gmsh.option.setNumber("Geometry.Curves",0)
println("Opening Gmsh — vicinity device: injector pair (source+drain adjacent) +")
println("probes A (near) / B (far) on the bottom edge; colored by physical group. Close to end.")
flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
