#!/usr/bin/env julia
# scripts/show_gmsh_circle.jl — build + verify + show the O-GRID structured mesh with a
# real CIRCULAR obstacle. Headlessly confirms all-quad, then opens the GUI.
#   julia --project=. scripts/show_gmsh_circle.jl            (window)
#   NOGUI=1 julia --project=. scripts/show_gmsh_circle.jl    (verify only)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using Gmsh: gmsh
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
xc=parse(Float64,get(ENV,"XC","0.72")); yc=parse(Float64,get(ENV,"YC","0.08"))
R =parse(Float64,get(ENV,"R","0.22"))
geo = joinpath(workdir, "circle_show.geo")
# Full channel: source=entire left wall, drain=entire right wall ⇒ the circle can be BIG
# and anywhere. Box half-side s=R+ring only needs to clear the top/bottom walls.
_, m = write_ogrid_geo(geo; W=0.8, xc=xc, yc=yc, R=R, ring=0.05, xPL=0.35, xPR=1.25, h=0.04)
@printf("O-grid: %dx%d blocks; circle R=%.3f box-halfside=%.3f @ (%.2f,%+.2f)\n",
        m.nx, m.ny, m.R, m.s, xc, yc)
@printf("clearances: top=%.2f bottom=%.2f sides=%.2f\n", m.gap_top, m.gap_bot, m.gap_x)
(m.gap_top>0 && m.gap_bot>0 && m.gap_x>0) || error("obstacle touches a perimeter")
flush(stdout)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo); gmsh.model.mesh.generate(2)
etypes, etags, _ = gmsh.model.mesh.getElements(2)
nq=0; nt=0
for (et,tg) in zip(etypes,etags); et==3 && (global nq+=length(tg)); et==2 && (global nt+=length(tg)); end
@printf("MESHED: %d quads, %d tris  (%s)\n", nq, nt, nt==0 ? "all-quad ✓ STRUCTURED" : "has tris ✗"); flush(stdout)
if get(ENV,"NOGUI","")=="1"; gmsh.finalize(); println("DONE (headless)"); exit(); end
# CLEAN view: just the structured quad grid (which conforms to the circle via the O-grid)
# + the obstacle circle outlined. No thick colored 1D clutter from the internal spokes/arcs.
gmsh.option.setNumber("Mesh.SurfaceEdges",1)     # the quad grid
gmsh.option.setNumber("Mesh.SurfaceFaces",0)     # no fill
gmsh.option.setNumber("Mesh.Lines",0)            # <- drop the busy colored boundary lines
gmsh.option.setNumber("Mesh.Points",0)
gmsh.option.setNumber("Mesh.ColorCarousel",0)    # single neutral mesh color
gmsh.option.setNumber("Geometry.Curves",1)       # show CAD curves (thin) so the circle+contacts read
gmsh.option.setNumber("Geometry.Points",0)
gmsh.option.setNumber("Geometry.CurveWidth",2)
println("Opening Gmsh — STRUCTURED channel with a floating CIRCULAR obstacle (O-grid")
println("rings wrap the circle). source=left, drain=right, probe_A=top, probe_B=bottom.")
println("The smooth circle in the middle is the obstacle. Close to end."); flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
