#!/usr/bin/env julia
# scripts/fixed_mesh_circle_opt.jl — big centred circle, optimize the Fourier
# dimples to maximize |f_1|, WITH a contact-exclusion band so the obstacle never
# touches the floating probes (the audit found probe contamination was ~90% of
# the old, inflated f_1). Uses the src/ API + analytic density.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, Printf, JLD2

cfg = ChannelConfig(L_x=1.0, W=0.6)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
ALPHA=2000.0; M=4; R0=0.20; H=0.04
EXCL=0.04                        # keep obstacle density ≥ EXCL off the ±W/2 walls
YCUT=cfg.W/2 - EXCL              # zero ρ for |y| > YCUT (no probe contamination)
CAP=YCUT - R0                    # so r0 + Σ|C| ≤ YCUT (body stays inside the band)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo=joinpath(workdir,"cpc.geo"); inp=joinpath(workdir,"cpc.inp")
write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=H)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)
field = DensityField(cfg; nx=64, ny=40, alpha_max=ALPHA)
ev = FixedEvaluator(inp, field, sim)
println("assembling..."); flush(stdout); t0=time()
op = assemble_fixed_operator(ev)
@printf("assembled: N=%d (%.0fs)\n", op.N, time()-t0); flush(stdout)

# per-dof y coordinate, for the contact-exclusion mask
nvars = Trixi.nvariables(ev.equations)
ycoord = zeros(op.N)
let w=Trixi.wrap_array(ycoord, op.ev.semi)
    _,_,_,cache = Trixi.mesh_equations_solver_cache(op.ev.semi)
    nc=cache.elements.node_coordinates; nn=size(w,2); nel=size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn, v in 1:nvars
        w[v,ii,jj,e]=nc[2,ii,jj,e]
    end
end
mask = abs.(ycoord) .> YCUT

function f1C(C)
    r = fourier_perdof(op; r0=R0, C=C, xc=cfg.x_c, width=0.02)
    r[mask] .= 0.0                       # contact exclusion
    f1_exact_perdof(op, r; alpha=ALPHA)
end

function main()
    C = zeros(2M); C[4] = -0.02
    f = f1C(C); step = 0.03; h = 0.004
    @printf("start f_1=%+.4e  (R0=%.2f, exclude |y|>%.2f)\n", f, R0, YCUT); flush(stdout)
    for it in 1:16
        g = zeros(2M)
        for k in 1:2M
            Cp=copy(C); Cp[k]+=h; Cm=copy(C); Cm[k]-=h
            g[k]=(f1C(Cp)-f1C(Cm))/(2h)
        end
        gn = sqrt(sum(abs2,g)); gn<1e-9 && break
        accepted=false
        for _ in 1:5
            Cn = C .+ step.*g./gn
            s=sum(abs,Cn); s>CAP && (Cn .*= CAP/s)
            fn = f1C(Cn)
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
    @printf("\nBEST (contamination-free) big-circle+dimples: f_1=%+.6e  R0=%.3f\n", f, R0)
    @printf("coeffs [a1,b1,a2,b2,...]=%s\n", string(round.(C,digits=4))); flush(stdout)
    nx,ny=field.nx,field.ny; bestphys=zeros(nx,ny)
    for j in 1:ny, i in 1:nx
        x=(i-0.5)*field.dx; y=field.y0+(j-0.5)*field.dy
        abs(y)>YCUT && continue
        dx=x-cfg.x_c; dy=y; dd=hypot(dx,dy); th=atan(dy,dx); R=R0
        for n in 1:M; R+=C[2n-1]*cos(n*th)+C[2n]*sin(n*th); end
        bestphys[i,j]=1/(1+exp((dd-R)/0.02))
    end
    @save joinpath(workdir,"result_circle.jld2") bestphys C R0 f cfg sim ALPHA nx ny EXCL YCUT
    println("wrote result_circle.jld2  DONE"); flush(stdout)
end
main()
