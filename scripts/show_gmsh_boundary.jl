#!/usr/bin/env julia
# scripts/show_gmsh_boundary.jl — show the optimized obstacle as a 2D BOUNDARY
# curve inside the channel perimeter (FermiSea-style .geo geometry), in Gmsh.
#   julia --project=. scripts/show_gmsh_boundary.jl
#
# The fixed-mesh optimum is a density rho(x,y). We extract its rho=0.5 contour,
# turn it into a closed boundary curve, and write a .geo: channel rectangle
# (perimeter) with the obstacle as a hole. Then open the Gmsh GUI.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt.Geometry
using JLD2, Printf
using Gmsh: gmsh

cfg = ChannelConfig(L_x=1.0, W=0.6)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
resfile = get(ENV,"RESULT","result_circle.jld2")
bestphys = load(joinpath(workdir,resfile))["bestphys"]
nx, ny = size(bestphys)
dx, dy = cfg.L_x/nx, cfg.W/ny
xc = [ (i-0.5)*dx for i in 1:nx ]                 # cell-center coords
yc = [ -cfg.W/2 + (j-0.5)*dy for j in 1:ny ]
LEVEL = 0.5

# ---- bilinear upsample for a smoother contour ----
function upsample(V, xc, yc, F)
    Nx,Ny=size(V)
    ux=range(xc[1],xc[end],length=(Nx-1)*F+1); uy=range(yc[1],yc[end],length=(Ny-1)*F+1)
    U=zeros(length(ux),length(uy))
    for (bj,y) in enumerate(uy), (bi,x) in enumerate(ux)
        fi=(x-xc[1])/(xc[2]-xc[1]); i=clamp(floor(Int,fi)+1,1,Nx-1); tx=clamp(fi-(i-1),0,1)
        fj=(y-yc[1])/(yc[2]-yc[1]); j=clamp(floor(Int,fj)+1,1,Ny-1); ty=clamp(fj-(j-1),0,1)
        U[bi,bj]=(1-tx)*(1-ty)*V[i,j]+tx*(1-ty)*V[i+1,j]+(1-tx)*ty*V[i,j+1]+tx*ty*V[i+1,j+1]
    end
    U, collect(ux), collect(uy)
end
U, ux, uy = upsample(bestphys, xc, yc, 5)

# ---- marching squares -> line segments ----
function segments(V, xs, ys, lv)
    Nx,Ny=size(V); segs=NTuple{4,Float64}[]
    for j in 1:Ny-1, i in 1:Nx-1
        va=V[i,j]; vb=V[i+1,j]; vc=V[i+1,j+1]; vd=V[i,j+1]
        x0=xs[i]; x1=xs[i+1]; y0=ys[j]; y1=ys[j+1]; p=Tuple{Float64,Float64}[]
        (va-lv)*(vb-lv)<0 && push!(p,(x0+(lv-va)/(vb-va)*(x1-x0), y0))
        (vb-lv)*(vc-lv)<0 && push!(p,(x1, y0+(lv-vb)/(vc-vb)*(y1-y0)))
        (vc-lv)*(vd-lv)<0 && push!(p,(x1+(lv-vc)/(vd-vc)*(x0-x1), y1))
        (vd-lv)*(va-lv)<0 && push!(p,(x0, y1+(lv-vd)/(va-vd)*(y0-y1)))
        if length(p)==2; push!(segs,(p[1][1],p[1][2],p[2][1],p[2][2]))
        elseif length(p)==4
            push!(segs,(p[1][1],p[1][2],p[2][1],p[2][2])); push!(segs,(p[3][1],p[3][2],p[4][1],p[4][2]))
        end
    end
    segs
end
segs = segments(U, ux, uy, LEVEL)

# ---- chain segments into closed loops ----
function chain(segs; tol=1e-7)
    near(p,q)=hypot(p[1]-q[1],p[2]-q[2])<tol
    used=falses(length(segs)); loops=Vector{Vector{Tuple{Float64,Float64}}}()
    for s in eachindex(segs)
        used[s] && continue; used[s]=true
        loop=[(segs[s][1],segs[s][2]),(segs[s][3],segs[s][4])]
        ext=true
        while ext; ext=false; tail=loop[end]
            for k in eachindex(segs)
                used[k] && continue
                a=(segs[k][1],segs[k][2]); b=(segs[k][3],segs[k][4])
                if near(tail,a); push!(loop,b); used[k]=true; ext=true; break
                elseif near(tail,b); push!(loop,a); used[k]=true; ext=true; break end
            end
        end
        push!(loops,loop)
    end
    loops
end
loops = chain(segs)
polyarea(L)=abs(sum((L[k][1]*L[k%length(L)+1][2]-L[k%length(L)+1][1]*L[k][2]) for k in 1:length(L)))/2
loops = [L for L in loops if length(L)>=4]
isempty(loops) && error("no contour found")
# keep only the largest loop (clean single boundary)
main = loops[argmax(polyarea.(loops))]
main[1]==main[end] && (main=main[1:end-1])          # open (no duplicate closing pt)
@printf("extracted %d loop(s); using largest (%d pts, area %.4f)\n",
        length(loops), length(main), polyarea(main)); flush(stdout)
# smooth the closed polygon (few passes of [.25 .5 .25])
function smoothclosed(P, passes)
    n=length(P)
    for _ in 1:passes
        Q=similar(P)
        for k in 1:n
            a=P[mod1(k-1,n)]; b=P[k]; c=P[mod1(k+1,n)]
            Q[k]=(0.25*a[1]+0.5*b[1]+0.25*c[1], 0.25*a[2]+0.5*b[2]+0.25*c[2])
        end
        P=Q
    end
    P
end
main = smoothclosed(main, 3)

# ---- write .geo: channel rectangle (perimeter) + obstacle hole ----
geo = joinpath(workdir, "obstacle_boundary.geo")
open(geo,"w") do io
    L=cfg.L_x; W2=cfg.W/2; h=0.03
    println(io,"SetFactory(\"OpenCASCADE\");"); @printf(io,"lc=%.4f;\n",h)
    @printf(io,"Point(1)={0,%.6f,0,lc};\nPoint(2)={%.6f,%.6f,0,lc};\n",-W2,L,-W2)
    @printf(io,"Point(3)={%.6f,%.6f,0,lc};\nPoint(4)={0,%.6f,0,lc};\n",L,W2,W2)
    println(io,"Line(1)={1,2};Line(2)={2,3};Line(3)={3,4};Line(4)={4,1};")
    println(io,"Curve Loop(1)={1,2,3,4};")
    pid=100; start=pid
    for (x,y) in main; @printf(io,"Point(%d)={%.6f,%.6f,0,lc};\n",pid,x,y); pid+=1; end
    @printf(io,"Spline(10)={%s,%d};\n", join(start:pid-1, ","), start)   # closed spline
    println(io,"Curve Loop(2)={10};")
    println(io,"Plane Surface(1)={1,2};")    # channel with obstacle as a hole
end

# ---- open in Gmsh GUI (CAD geometry only; no meshing -> no recovery hang) ----
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo)
gmsh.option.setNumber("Geometry.Curves",1)
gmsh.option.setNumber("Geometry.CurveWidth",4)
gmsh.option.setNumber("Geometry.Points",0)
gmsh.option.setNumber("Geometry.SurfaceType",0)
println("Opening Gmsh — obstacle boundary within the channel perimeter. Close window to end.")
flush(stdout)
gmsh.fltk.run()
gmsh.finalize()
