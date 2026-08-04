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
# --- faint, muted mesh grid (thin, soft gray — no neon) ---
gmsh.option.setNumber("Mesh.SurfaceEdges",1)     # the quad grid
gmsh.option.setNumber("Mesh.SurfaceFaces",0)     # no fill
gmsh.option.setNumber("Mesh.Lines",0)            # no 1D-element clutter
gmsh.option.setNumber("Mesh.Points",0)
gmsh.option.setNumber("Mesh.ColorCarousel",0)    # single color, not by-group neon
gmsh.option.setNumber("Mesh.LineWidth",0.7)      # thin grid
gmsh.option.setColor("Mesh.Lines",       165,170,180,255)  # soft gray-blue grid
gmsh.option.setColor("Mesh.Quadrangles", 165,170,180,255)
gmsh.option.setColor("Mesh.Triangles",   165,170,180,255)
# --- bold ONLY the obstacle circle over the faint grid ---
gmsh.option.setNumber("Geometry.Curves",1)       # CAD curves channel (independent width)
gmsh.option.setNumber("Geometry.CurveWidth",5)   # bold
gmsh.option.setNumber("Geometry.Points",0)
allc = gmsh.model.getEntities(1)                  # hide every CAD curve...
gmsh.model.setVisibility(allc, 0)
obs = [(1,601),(1,602),(1,603),(1,604)]           # ...except the 4 circle arcs (the obstacle)
gmsh.model.setVisibility(obs, 1)
gmsh.model.setColor(obs, 200, 40, 40)             # bold red obstacle outline
println("Opening Gmsh — STRUCTURED channel with a floating CIRCULAR obstacle (O-grid")
println("rings wrap the circle). source=left, drain=right, probe_A=top, probe_B=bottom.")
println("The smooth circle in the middle is the obstacle. Close to end."); flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
