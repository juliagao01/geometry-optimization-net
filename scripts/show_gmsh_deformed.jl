#!/usr/bin/env julia
# scripts/show_gmsh_deformed.jl — build/verify/show the O-grid with a smooth DEFORMED
# circular obstacle in a taller, roomier channel. Faint grid + bold obstacle.
#   NOGUI=1 julia --project=. scripts/show_gmsh_deformed.jl   (verify only)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using Gmsh: gmsh
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
# defaults: taller channel, big circular-ish deformed obstacle (b2,b3 small ⇒ round)
L=1.8; W=1.4
xc=parse(Float64,get(ENV,"XC","0.90")); yc=parse(Float64,get(ENV,"YC","0.15"))
R =parse(Float64,get(ENV,"R","0.28"))
b2=parse(Float64,get(ENV,"B2","0.05")); b3=parse(Float64,get(ENV,"B3","0.02"))
a2=parse(Float64,get(ENV,"A2","0.02"))
geo=joinpath(workdir,"deformed_show.geo")
# probe span (xPL,xPR) must CONTAIN the obstacle box; keep it wide.
_, m = write_ogrid_deformed(geo; L=L,W=W,xc=xc,yc=yc,R=R, A=[0.0,a2], B=[0.0,b2,b3],
                            ring=0.07, xPL=0.25, xPR=L-0.25, h=0.05)
@printf("deformed obstacle: R=%.3f rmax=%.3f box=%.3f @ (%.2f,%+.2f)  A=[0,%.2f] B=[0,%.2f,%.2f]\n",
        m.R, m.rmax, m.s, xc, yc, a2, b2, b3)
@printf("clearances (to rmax): top=%.2f bottom=%.2f sides=%.2f\n", m.gap_top, m.gap_bot, m.gap_x)
(m.gap_top>0 && m.gap_bot>0 && m.gap_x>0) || error("obstacle touches a perimeter")
flush(stdout)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo); gmsh.model.mesh.generate(2)
et,tg,_=gmsh.model.mesh.getElements(2); nq=0; nt=0
for (e,t) in zip(et,tg); e==3 && (global nq+=length(t)); e==2 && (global nt+=length(t)); end
@printf("MESHED: %d quads, %d tris  (%s)\n", nq, nt, nt==0 ? "all-quad ✓" : "has tris ✗"); flush(stdout)
if get(ENV,"NOGUI","")=="1"; gmsh.finalize(); println("DONE (headless)"); exit(); end
gmsh.option.setNumber("Mesh.SurfaceEdges",1); gmsh.option.setNumber("Mesh.SurfaceFaces",0)
gmsh.option.setNumber("Mesh.Lines",0); gmsh.option.setNumber("Mesh.Points",0)
gmsh.option.setNumber("Mesh.ColorCarousel",0); gmsh.option.setNumber("Mesh.LineWidth",0.7)
gmsh.option.setColor("Mesh.Lines",165,170,180,255); gmsh.option.setColor("Mesh.Quadrangles",165,170,180,255)
gmsh.option.setNumber("Geometry.Curves",1); gmsh.option.setNumber("Geometry.CurveWidth",5); gmsh.option.setNumber("Geometry.Points",0)
allc=gmsh.model.getEntities(1); gmsh.model.setVisibility(allc,0)
obs=[(1,601),(1,602),(1,603),(1,604)]; gmsh.model.setVisibility(obs,1); gmsh.model.setColor(obs,200,40,40)
println("Opening Gmsh — deformed circular-ish obstacle (bold red) in a taller channel."); flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
