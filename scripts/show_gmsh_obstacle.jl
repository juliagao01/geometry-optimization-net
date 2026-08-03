#!/usr/bin/env julia
# scripts/show_gmsh_obstacle.jl — GUI of a literature channel-with-obstacle device:
# a CLOSED obstacle (circle or square) floating in the interior as a HOLE, with a
# guaranteed clear gap to every wall and every contact (no touching → no probe
# contamination). Current enters at source (left wall), exits at drain (right wall),
# and floating voltage probes sit on the top/bottom walls; the obstacle deflects the
# flow. Shows the mesh grid + the obstacle boundary + contacts colored by group.
#   SHAPE=circle julia --project=. scripts/show_gmsh_obstacle.jl     (or SHAPE=square)
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using Gmsh: gmsh
using Printf
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)

# --- device parameters ---
L, W  = 1.6, 0.8                 # channel length, width
W2    = W/2
xc    = L/2
yc    = parse(Float64, get(ENV, "YC", "0.10"))   # OFFSET breaks up-down symmetry ⇒ f1≠0
Robs  = parse(Float64, get(ENV, "R",  "0.18"))   # smaller than centered case to keep fat gaps
wc    = 0.12                    # contact width
xPL, xPR = xc-0.20, xc+0.20     # probe segments on top/bottom, spanning the obstacle
h     = 0.05
SHAPE = Symbol(get(ENV, "SHAPE", "circle"))

# obstacle boundary polygon (closed, floating)
function obstacle_pts()
    if SHAPE === :square
        s = Robs*0.85
        # 4 sides, subdivided for a clean transfinite-ish boundary
        pts = Tuple{Float64,Float64}[]
        corners = [(-s,-s),(s,-s),(s,s),(-s,s)]
        for k in 1:4
            (ax,ay)=corners[k]; (bx,by)=corners[k%4+1]
            for t in range(0,1,length=13)[1:end-1]
                push!(pts,(xc+ax+(bx-ax)*t, yc+ay+(by-ay)*t))
            end
        end
        return pts
    else
        n=64; th=range(0,2pi,length=n+1)[1:end-1]
        return [(xc+Robs*cos(t), yc+Robs*sin(t)) for t in th]
    end
end

geo = joinpath(workdir, "obstacle_show.geo")
open(geo, "w") do io
    @printf(io, "lc=%.5f;\n", h)
    # outer channel boundary with contact segments (source left, drain right, probes top/bottom)
    P = [(0.0,-W2),(0.0,-wc/2),(0.0,wc/2),(0.0,W2),(xPL,W2),(xPR,W2),(L,W2),
         (L,wc/2),(L,-wc/2),(L,-W2),(xPR,-W2),(xPL,-W2)]
    for (i,(x,y)) in enumerate(P); @printf(io,"Point(%d)={%.6f,%.6f,0,lc};\n",i,x,y); end
    for i in 1:12; @printf(io,"Line(%d)={%d,%d};\n", i, i, i==12 ? 1 : i+1); end
    println(io,"Curve Loop(1)={",join(1:12,","),"};")
    ob = obstacle_pts(); nob = length(ob)
    for (k,(x,y)) in enumerate(ob); @printf(io,"Point(%d)={%.6f,%.6f,0,lc};\n",100+k,x,y); end
    for k in 1:nob; @printf(io,"Line(%d)={%d,%d};\n", 200+k, 100+k, 100+(k%nob)+1); end
    println(io,"Curve Loop(2)={",join(201:200+nob,","),"};")
    println(io,"Plane Surface(1)={1,2};")              # channel MINUS the obstacle hole
    println(io,"Physical Surface(\"domain\")={1};")
    println(io,"Physical Curve(\"contact_source\")={2};")
    println(io,"Physical Curve(\"contact_drain\")={8};")
    println(io,"Physical Curve(\"probe_A\")={5};")
    println(io,"Physical Curve(\"probe_B\")={11};")
    println(io,"Physical Curve(\"walls\")={1,3,4,6,7,9,10,12};")
    println(io,"Physical Curve(\"obstacle\")={",join(201:200+nob,","),"};")
end

# clearances (should all be positive; top≠bottom once offset)
gap_top = W2 - (yc + Robs); gap_bot = W2 + (yc - Robs); gap_x = xc - Robs
@printf("obstacle=%s  R=%.2f  center=(%.2f,%+.2f)  [offset yc=%+.2f breaks up-down symmetry]\n",
        SHAPE, Robs, xc, yc, yc)
@printf("clearances: top wall/probe_A=%.2f, bottom wall/probe_B=%.2f, source/drain=%.2f\n",
        gap_top, gap_bot, gap_x)
(gap_top>0 && gap_bot>0 && gap_x>0) || error("obstacle touches a perimeter — reduce R or yc")
flush(stdout)

gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo)
gmsh.option.setNumber("Mesh.Algorithm",8)              # 11 hangs on holed domains; 8 = Frontal-Delaunay
gmsh.option.setNumber("Mesh.RecombineAll",1); gmsh.option.setNumber("Mesh.RecombinationAlgorithm",1)
gmsh.model.mesh.generate(2)
gmsh.option.setNumber("Mesh.SurfaceEdges",1)           # show the grid
gmsh.option.setNumber("Mesh.SurfaceFaces",0)           # no green fill
gmsh.option.setNumber("Mesh.Lines",1)                  # draw 1D (boundary) elements
gmsh.option.setNumber("Mesh.LineWidth",6)              # fat contact + obstacle lines
gmsh.option.setNumber("Mesh.ColorCarousel",2)          # color by physical group
gmsh.option.setNumber("Geometry.Curves",0); gmsh.option.setNumber("Geometry.Points",0)
println("Opening Gmsh — channel with a floating closed obstacle (a HOLE, clear of all")
println("perimeters). source=left wall, drain=right wall, probe_A=top, probe_B=bottom;")
println("the closed loop in the middle is the obstacle boundary. Close to end.")
flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
