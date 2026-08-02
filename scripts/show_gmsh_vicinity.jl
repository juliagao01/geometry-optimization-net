#!/usr/bin/env julia
# scripts/show_gmsh_vicinity.jl — GUI of the Bandurin–Levitov vicinity device.
# There is NO obstacle in the middle by design: the device geometry IS the
# arrangement of narrow contacts on the bottom edge — an adjacent current
# injector pair (source+drain) and two floating voltage probes (A near, B far).
# We hide the surface fill and draw the mesh grid + the contact segments colored
# by physical group (fat lines), so the device reads clearly.
#   julia --project=. scripts/show_gmsh_vicinity.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt.Geometry, VicinityOpt.FixedMesh
using Gmsh: gmsh
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo=joinpath(workdir,"vic_show.geo")
write_vicinity_geo(geo; L=2.0, W=0.8, wc=0.1, h=0.04, xs=1.25, dA=0.15, dB=0.45)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo)
gmsh.model.mesh.generate(2)                        # structured (transfinite in .geo)
# --- visibility: grid on, green face fill OFF, contacts fat + colored by group ---
gmsh.option.setNumber("Mesh.SurfaceFaces",0)       # <- kills the neon-green fill
gmsh.option.setNumber("Mesh.SurfaceEdges",1)       # show the mesh grid
gmsh.option.setNumber("Mesh.Lines",1)              # draw 1D (boundary) mesh elements
gmsh.option.setNumber("Mesh.LineWidth",7)          # fat contact segments
gmsh.option.setNumber("Mesh.ColorCarousel",2)      # color 1D elements by physical group
gmsh.option.setNumber("Mesh.Points",0)
gmsh.option.setNumber("Geometry.Curves",0)
gmsh.option.setNumber("Geometry.Points",0)
println("""
Opening Gmsh — vicinity device (NO central obstacle, that is intended).
  Bottom edge, left→right:  probe_B (far) | probe_A (near) | SOURCE | DRAIN
  Each contact is a fat colored segment (colored by physical group); the rest of
  the perimeter + interior grid are the sample walls / mesh.
If you still see a solid green rectangle, toggle Tools > Options > Mesh >
  Visibility: uncheck 'Surface faces', check 'Surface edges' and 'Line elements'.
Close the window to end.""")
flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
