#!/usr/bin/env julia
# scripts/render_optimal_shape.jl — render the optimal deformed obstacle to docs/optimal_shape.png
# (faint O-grid + bold red obstacle). Uses Gmsh's off-screen/graphics image export.
#   julia --project=. scripts/render_optimal_shape.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "ogrid_geo.jl"))
using Gmsh: gmsh
using JLD2, Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
docs    = joinpath(@__DIR__, "..", "docs"); mkpath(docs)
# optimal params (result_deformed.jld2): xc,yc,R,a2,b2,b3
d = load(joinpath(workdir,"result_deformed.jld2")); xb = d["xbest"]
xc,yc,R,a2,b2,b3 = xb
@printf("rendering optimal shape: xc=%.3f yc=%+.3f R=%.3f a2=%+.3f b2=%+.3f b3=%+.3f\n", xc,yc,R,a2,b2,b3)
geo = joinpath(workdir,"opt_render.geo")
write_ogrid_deformed(geo; L=1.8, W=1.4, xc=xc, yc=yc, R=R, A=[0.0,a2], B=[0.0,b2,b3],
                     ring=0.06, xPL=0.25, xPR=1.55, h=0.045)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo); gmsh.model.mesh.generate(2)
# styling: faint grid, bold red obstacle, clean white background
gmsh.option.setNumber("Mesh.SurfaceEdges",1); gmsh.option.setNumber("Mesh.SurfaceFaces",0)
gmsh.option.setNumber("Mesh.Lines",0); gmsh.option.setNumber("Mesh.Points",0)
gmsh.option.setNumber("Mesh.ColorCarousel",0); gmsh.option.setNumber("Mesh.LineWidth",1.0)
gmsh.option.setColor("Mesh.Lines",170,175,185,255); gmsh.option.setColor("Mesh.Quadrangles",170,175,185,255)
gmsh.option.setNumber("Geometry.Curves",1); gmsh.option.setNumber("Geometry.CurveWidth",6); gmsh.option.setNumber("Geometry.Points",0)
gmsh.option.setNumber("General.SmallAxes",0); gmsh.option.setNumber("General.Axes",0)
gmsh.option.setColor("General.Background",255,255,255,255)
gmsh.option.setColor("General.Foreground",60,60,60,255); gmsh.option.setColor("General.Text",20,20,20,255)
gmsh.option.setNumber("General.GraphicsWidth",1400); gmsh.option.setNumber("General.GraphicsHeight",1100)
try
    gmsh.fltk.initialize()
    allc=gmsh.model.getEntities(1); gmsh.model.setVisibility(allc,0)
    obs=[(1,601),(1,602),(1,603),(1,604)]; gmsh.model.setVisibility(obs,1); gmsh.model.setColor(obs,205,35,35)
    gmsh.graphics.draw()
    out=joinpath(docs,"optimal_shape.png"); gmsh.write(out)
    println("wrote ", out)
catch e
    println("image export failed: ", e)
end
gmsh.finalize(); println("DONE")
