#!/usr/bin/env julia
# scripts/fixed_mesh_circle.jl — LOOP iter: optimize a LARGE circular obstacle
# (density on the fixed point-contact mesh) to maximize f_1, radius as big as
# possible before touching the perimeter. Exact reg-LU solve (convergent f_1).
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra, Printf, JLD2
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
ALPHA=2000.0; EPS=1e-10
DNX,DNY = 64, 40          # finer design grid so a big circle is smooth
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo=joinpath(workdir,"cpc.geo"); inp=joinpath(workdir,"cpc.inp")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.045)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
field = DensityField(cfg; nx=DNX, ny=DNY, alpha_max=1.0)
ev = FixedEvaluator(inp, field, sim); semi=ev.semi
_, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)
extract()=(FermiSea._current_contact_potential(ev.bc_probe_A,equations,dg,cache)-
           FermiSea._current_contact_potential(ev.bc_probe_B,equations,dg,cache))/sim.I_source
clear!(field); ode=Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
b=similar(ode.u0); Trixi.rhs!(b,zero(ode.u0),semi,0.0)
println("assembling (N=$N)..."); flush(stdout); t0=time()
Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(b);ej=zeros(N)
for j in 1:N
    ej[j]=1.0; Trixi.rhs!(tmp,ej,semi,0.0); c[j]=extract(); ej[j]=0.0
    @inbounds for i in 1:N
        v=tmp[i]-b[i]; abs(v)>1e-12 && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
    end
end
A0=sparse(Ir,Jc,Vv,N,N)
fill!(field.rho,1.0); ov=ones(N); r1=similar(b); Trixi.rhs!(r1,ov,semi,0.0)
clear!(field); r0v=similar(b); Trixi.rhs!(r0v,ov,semi,0.0); b1u=r1.-r0v
@printf("assembled (%.0fs)\n", time()-t0); flush(stdout)
rpd=zeros(N); wA=Trixi.wrap_array(rpd,semi); nc=cache.elements.node_coordinates
nn=size(wA,2); nel=size(wA,4)
function f1_of(rgrid)
    wr=Trixi.wrap_array(rpd,semi)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; wr[v,ii,jj,e]=rgrid[ci,cj]; end
    end
    dot(c, lu(A0+spdiagm(0=>(ALPHA.*rpd).*b1u)+EPS*spdiagm(0=>ones(N)))\(-b))
end
# smooth Fourier-circle painter -> density grid.
# boundary R(theta)=r0 + a1 cos + b1 sin + a2 cos2 + b2 sin2 (dimples)
function fcircle(yc,r0,a1,b1,a2,b2; xc=cfg.x_c, wdt=0.02)
    g=zeros(DNX,DNY)
    for j in 1:DNY, i in 1:DNX
        x=(i-0.5)*field.dx; y=field.y0+(j-0.5)*field.dy
        dx=x-xc; dy=y-yc; d=hypot(dx,dy); th=atan(dy,dx)
        R=r0+a1*cos(th)+b1*sin(th)+a2*cos(2th)+b2*sin(2th)
        g[i,j]=1/(1+exp((d-R)/wdt))
    end
    g
end
circle(yc,r)=fcircle(yc,r,0,0,0,0)

function main()
    println("\n=== (A) plain circles: f_1 over (yc,r), tangent cases included ===")
    for r in (0.12,0.16,0.20,0.24,0.27)
        line=""
        for yc in (0.0,0.04,0.08,0.12,0.16)
            yc+r <= cfg.W/2-0.012+1e-9 || continue
            f=f1_of(circle(yc,r))
            line*=@sprintf("  yc=%.2f:%+.3e",yc,f)
        end
        @printf(" r=%.3f%s\n",r,line); flush(stdout)
    end
    println("\n=== (B) BIG centred circle + sin dimples (break symmetry) ===")
    best=(f1=-Inf,p=(0.0,0.20,0.0,0.0,0.0,0.0))
    for r0 in (0.20,0.24,0.26), b1 in (0.0,0.03,0.06), b2 in (0.0,0.03)
        p=(0.0,r0,0.0,b1,0.0,b2)
        f=f1_of(fcircle(p...))
        @printf("  r0=%.2f b1=%.2f b2=%.2f -> f_1=%+.5e\n",r0,b1,b2,f); flush(stdout)
        f>best.f1 && (best=(f1=f,p=p))
    end
    @printf("\nBEST: r0=%.3f b1=%.3f b2=%.3f  f_1=%+.6e\n",
            best.p[2],best.p[4],best.p[6],best.f1); flush(stdout)
    bestphys=fcircle(best.p...)
    @save joinpath(workdir,"result_circle.jld2") bestphys best cfg sim ALPHA DNX DNY
    println("wrote result_circle.jld2  DONE"); flush(stdout)
end
main()
