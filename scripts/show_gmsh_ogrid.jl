#!/usr/bin/env julia
# scripts/show_gmsh_ogrid.jl — build + verify + show the STRUCTURED square-obstacle mesh.
# Headlessly checks it meshes as all-quads (no hang, structured), then opens the GUI.
#   julia --project=. scripts/show_gmsh_ogrid.jl            (opens window)
#   NOGUI=1 julia --project=. scripts/show_gmsh_ogrid.jl    (headless verify only)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "blockstruct_geo.jl"))
using Gmsh: gmsh
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "ogrid_show.geo")
_, m = write_blockstruct_geo(geo; h=0.05)
@printf("blocks: %d cols x %d rows; obstacle block=(%d,%d); source row=%d; probe cols=%s\n",
        m.nx, m.ny, m.ci, m.cj, m.jsrc, m.cprobe)
@printf("clearances: top=%.2f bottom=%.2f sides=%.2f source-gap=%.2f\n",
        m.gap_top, m.gap_bot, m.gap_x, m.gap_src)
(m.gap_top>0 && m.gap_bot>0 && m.gap_x>0 && m.gap_src>0) || error("obstacle touches a perimeter/contact")
flush(stdout)

gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo)
gmsh.model.mesh.generate(2)                                   # pure transfinite ⇒ structured, no hang
etypes, etags, _ = gmsh.model.mesh.getElements(2)
nquad = 0; ntri = 0
for (et, tg) in zip(etypes, etags)
    et == 3 && (global nquad += length(tg))                  # 3 = 4-node quad
    et == 2 && (global ntri  += length(tg))                  # 2 = 3-node tri
end
@printf("MESHED: %d quads, %d tris  (%s)\n", nquad, ntri, ntri==0 ? "all-quad ✓ STRUCTURED" : "has tris ✗")
flush(stdout)

if get(ENV,"NOGUI","")=="1"
    gmsh.finalize(); println("DONE (headless)"); exit()
end
gmsh.option.setNumber("Mesh.SurfaceEdges",1); gmsh.option.setNumber("Mesh.SurfaceFaces",0)
gmsh.option.setNumber("Mesh.Lines",1); gmsh.option.setNumber("Mesh.LineWidth",6)
gmsh.option.setNumber("Mesh.ColorCarousel",2)
gmsh.option.setNumber("Geometry.Curves",0); gmsh.option.setNumber("Geometry.Points",0)
println("Opening Gmsh — STRUCTURED channel with a floating SQUARE obstacle (a clean")
println("block hole). source=left, drain=right, probe_A=top, probe_B=bottom; the box in")
println("the middle is the obstacle. All-quad transfinite grid. Close to end.")
flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
