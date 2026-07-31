#!/usr/bin/env julia
# scripts/show_gmsh_grid.jl — show the FIXED mesh grid with the optimized obstacle
# boundary overlaid on it (the fixed-mesh idea: obstacle = density on a fixed
# structured mesh, NOT a body-fitted hole).
#   RESULT=result_circle.jld2 julia --project=. scripts/show_gmsh_grid.jl
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt.Geometry
using JLD2, Printf
using Gmsh: gmsh
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh")
bestphys = load(joinpath(workdir, get(ENV,"RESULT","result_circle.jld2")))["bestphys"]
nx, ny = size(bestphys); dx, dy = cfg.L_x/nx, cfg.W/ny
xc=[(i-0.5)*dx for i in 1:nx]; yc=[-cfg.W/2+(j-0.5)*dy for j in 1:ny]

# --- upsample + marching squares -> largest smoothed contour (as in boundary view) ---
function upsample(V,xc,yc,F)
    Nx,Ny=size(V); ux=range(xc[1],xc[end],length=(Nx-1)*F+1); uy=range(yc[1],yc[end],length=(Ny-1)*F+1)
    U=zeros(length(ux),length(uy))
    for (bj,y) in enumerate(uy),(bi,x) in enumerate(ux)
        fi=(x-xc[1])/(xc[2]-xc[1]); i=clamp(floor(Int,fi)+1,1,Nx-1); tx=clamp(fi-(i-1),0,1)
        fj=(y-yc[1])/(yc[2]-yc[1]); j=clamp(floor(Int,fj)+1,1,Ny-1); ty=clamp(fj-(j-1),0,1)
        U[bi,bj]=(1-tx)*(1-ty)*V[i,j]+tx*(1-ty)*V[i+1,j]+(1-tx)*ty*V[i,j+1]+tx*ty*V[i+1,j+1]
    end
    U,collect(ux),collect(uy)
end
U,ux,uy=upsample(bestphys,xc,yc,5)
function segments(V,xs,ys,lv)
    Nx,Ny=size(V); s=NTuple{4,Float64}[]
    for j in 1:Ny-1,i in 1:Nx-1
        va=V[i,j];vb=V[i+1,j];vc=V[i+1,j+1];vd=V[i,j+1]
        x0=xs[i];x1=xs[i+1];y0=ys[j];y1=ys[j+1];p=Tuple{Float64,Float64}[]
        (va-lv)*(vb-lv)<0&&push!(p,(x0+(lv-va)/(vb-va)*(x1-x0),y0))
        (vb-lv)*(vc-lv)<0&&push!(p,(x1,y0+(lv-vb)/(vc-vb)*(y1-y0)))
        (vc-lv)*(vd-lv)<0&&push!(p,(x1+(lv-vc)/(vd-vc)*(x0-x1),y1))
        (vd-lv)*(va-lv)<0&&push!(p,(x0,y1+(lv-vd)/(va-vd)*(y0-y1)))
        length(p)==2&&push!(s,(p[1][1],p[1][2],p[2][1],p[2][2]))
        length(p)==4&&(push!(s,(p[1][1],p[1][2],p[2][1],p[2][2]));push!(s,(p[3][1],p[3][2],p[4][1],p[4][2])))
    end
    s
end
segs=segments(U,ux,uy,0.5)
function chain(segs;tol=1e-7)
    near(p,q)=hypot(p[1]-q[1],p[2]-q[2])<tol; used=falses(length(segs)); loops=[]
    for s in eachindex(segs)
        used[s]&&continue; used[s]=true; loop=[(segs[s][1],segs[s][2]),(segs[s][3],segs[s][4])]; ext=true
        while ext; ext=false; tail=loop[end]
            for k in eachindex(segs)
                used[k]&&continue; a=(segs[k][1],segs[k][2]);b=(segs[k][3],segs[k][4])
                if near(tail,a);push!(loop,b);used[k]=true;ext=true;break
                elseif near(tail,b);push!(loop,a);used[k]=true;ext=true;break end
            end
        end
        push!(loops,loop)
    end
    loops
end
polyarea(L)=abs(sum((L[k][1]*L[k%length(L)+1][2]-L[k%length(L)+1][1]*L[k][2]) for k in 1:length(L)))/2
loops=[L for L in chain(segs) if length(L)>=4]; main=loops[argmax(polyarea.(loops))]
main[1]==main[end]&&(main=main[1:end-1])
function smoothclosed(P,passes); n=length(P)
    for _ in 1:passes; Q=similar(P)
        for k in 1:n; a=P[mod1(k-1,n)];b=P[k];c=P[mod1(k+1,n)]
            Q[k]=(0.25a[1]+0.5b[1]+0.25c[1],0.25a[2]+0.5b[2]+0.25c[2]) end
        P=Q end
    P
end
main=smoothclosed(main,3)
@printf("obstacle boundary: %d pts, area %.3f\n",length(main),polyarea(main)); flush(stdout)

# --- obstacle boundary as a Gmsh post-view (.pos), thick line ---
pos=joinpath(workdir,"obstacle_line.pos")
open(pos,"w") do io
    println(io,"View \"obstacle boundary\" {")
    n=length(main)
    for k in 1:n
        a=main[k]; b=main[mod1(k+1,n)]
        @printf(io,"SL(%.6f,%.6f,0,%.6f,%.6f,0){1,1};\n",a[1],a[2],b[1],b[2])
    end
    println(io,"};")
end

# --- FIXED structured mesh (grid) + obstacle boundary overlay ---
geo=joinpath(workdir,"cp_grid.geo")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.035)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",1)
gmsh.open(geo)
gmsh.model.mesh.generate(2)      # structured -> fast/clean, no hole to hang on
gmsh.merge(pos)
gmsh.option.setNumber("Mesh.SurfaceEdges",1)   # show the grid
gmsh.option.setNumber("Mesh.SurfaceFaces",0)
gmsh.option.setNumber("Mesh.ColorCarousel",0)
gmsh.option.setNumber("View[0].LineWidth",4)
gmsh.option.setNumber("View[0].ColormapNumber",1)
gmsh.option.setNumber("View[0].ShowScale",0)
println("Opening Gmsh — fixed mesh grid with obstacle boundary overlaid. Close to end.")
flush(stdout)
gmsh.fltk.run(); gmsh.finalize()
