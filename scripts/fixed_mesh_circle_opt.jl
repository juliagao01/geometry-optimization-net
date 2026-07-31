#!/usr/bin/env julia
# scripts/fixed_mesh_circle_opt.jl — LOOP iter 3: big centred circle, optimize the
# Fourier dimples to maximize |f_1|. Density on fixed point-contact mesh, reg-LU.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra, Printf, JLD2
include(joinpath(@__DIR__, "pointgeo.jl"))

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
ALPHA=2000.0; EPS=1e-10; DNX,DNY=64,40; M=4; R0=0.24
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
@printf("assembled (%.0fs)\n",time()-t0); flush(stdout)
rpd=zeros(N); nc=cache.elements.node_coordinates
nn=size(Trixi.wrap_array(rpd,semi),2); nel=size(Trixi.wrap_array(rpd,semi),4)

# Fourier-circle density: R(θ)=R0 + Σ_n a_n cos nθ + b_n sin nθ  (C=[a1,b1,...])
function paintC(C; r0=R0, yc=0.0, xc=cfg.x_c, wdt=0.02)
    g=zeros(DNX,DNY)
    for j in 1:DNY, i in 1:DNX
        x=(i-0.5)*field.dx; y=field.y0+(j-0.5)*field.dy
        dx=x-xc; dy=y-yc; d=hypot(dx,dy); th=atan(dy,dx); R=r0
        for n in 1:M; R+=C[2n-1]*cos(n*th)+C[2n]*sin(n*th); end
        g[i,j]=1/(1+exp((d-R)/wdt))
    end
    g
end
function f1C(C)
    g=paintC(C); wr=Trixi.wrap_array(rpd,semi)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; wr[v,ii,jj,e]=g[ci,cj]; end
    end
    dot(c, lu(A0+spdiagm(0=>(ALPHA.*rpd).*b1u)+EPS*spdiagm(0=>ones(N)))\(-b))
end

function main()
    # maximize |f_1| via FD gradient ascent on the harmonic vector (yc=0, r0=R0)
    C=zeros(2M); C[4]=-0.02          # seed b2<0 (positive-f_1 branch)
    obj(C)=f1C(C)
    f=obj(C); step=0.03; h=0.004
    @printf("start f_1=%+.4e\n",f); flush(stdout)
    for it in 1:14
        g=zeros(2M)
        for k in 1:2M
            Cp=copy(C);Cp[k]+=h; Cm=copy(C);Cm[k]-=h
            g[k]=(obj(Cp)-obj(Cm))/(2h)
        end
        gn=norm(g); gn<1e-9 && break
        # amplitude cap so bumps don't cross the wall: R0+Σ|C|<=0.275
        accepted=false
        for _ in 1:5
            Cn=C .+ step.*g./gn
            s=sum(abs,Cn); s>0.275-R0 && (Cn .*= (0.275-R0)/s)
            fn=obj(Cn)
            if fn>f
                C=Cn; f=fn; accepted=true; break
            else
                step*=0.5
            end
        end
        @printf("it %2d  f_1=%+.5e  |C|1=%.3f b2=%+.3f  step=%.3f\n",
                it,f,sum(abs,C),C[4],step); flush(stdout)
        (!accepted || step<2e-3) && break
    end
    @printf("\nBEST big-circle+dimples: f_1=%+.6e  R0=%.3f\n",f,R0)
    @printf("coeffs [a1,b1,a2,b2,...]=%s\n", string(round.(C,digits=4))); flush(stdout)
    bestphys=paintC(C)
    @save joinpath(workdir,"result_circle.jld2") bestphys C R0 f cfg sim ALPHA DNX DNY
    println("wrote result_circle.jld2  DONE"); flush(stdout)
end
main()
