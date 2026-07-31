#!/usr/bin/env julia
# scripts/fixed_mesh_reg.jl — fix the non-unique steady state via regularization.
#   The operator has a null space (undamped density mode a0), so A u = -b is
#   non-unique; LU picks an arbitrary representative, GMRES-from-0 picks the
#   minimum-norm one. Tikhonov (A + εI) selects min-norm as ε->0. Verify
#   regularized LU reproduces GMRES at soft AND solid obstacles, then take α up.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.05)
sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel_sym.geo"); inp = joinpath(workdir, "channel_sym.inp")
write_channel_geo_symmetric(geo, cfg; h=cfg.lc)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)

field = DensityField(cfg; nx=30, ny=18, alpha_max=1.0)
ev = FixedEvaluator(inp, field, sim); semi = ev.semi
mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)
extract_f1() = begin
    _,_,dg,ca = Trixi.mesh_equations_solver_cache(semi)
    (FermiSea._current_contact_potential(ev.bc_probe_A, equations, dg, ca) -
     FermiSea._current_contact_potential(ev.bc_probe_B, equations, dg, ca)) / sim.I_source
end
clear!(field)
ode = Trixi.semidiscretize(semi,(0.0,1.0)); N=length(ode.u0)
b = similar(ode.u0); Trixi.rhs!(b, zero(ode.u0), semi, 0.0)

# assemble A0 + c
Ir=Int[];Jc=Int[];Vv=Float64[];c=zeros(N);tmp=similar(b);ej=zeros(N);tol=1e-12
for j in 1:N
    ej[j]=1.0; Trixi.rhs!(tmp,ej,semi,0.0); c[j]=extract_f1(); ej[j]=0.0
    @inbounds for i in 1:N
        v=tmp[i]-b[i]; abs(v)>tol && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
    end
end
A0=sparse(Ir,Jc,Vv,N,N)
field.alpha_max=1.0; onev=ones(N); fill!(field.rho,1.0); r1=similar(b); Trixi.rhs!(r1,onev,semi,0.0)
clear!(field); r0=similar(b); Trixi.rhs!(r0,onev,semi,0.0); b1u=r1.-r0
rho_perdof=zeros(N)
function set_rho_perdof!(rg)
    w=Trixi.wrap_array(rho_perdof,semi); nc=cache.elements.node_coordinates
    nn=size(w,2); nel=size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x=nc[1,ii,jj,e]; y=nc[2,ii,jj,e]
        ci=clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj=clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; w[v,ii,jj,e]=rg[ci,cj]; end
    end
end
paint_blob!(field, cfg.x_c, 0.10, 0.16); rho_grid=copy(field.rho); set_rho_perdof!(rho_grid)
println("assembled N=$N nnz=$(nnz(A0))"); flush(stdout)
Idn = spdiagm(0=>ones(N))

f1_reg(alpha, eps) = begin
    A = A0 + spdiagm(0=>(alpha.*rho_perdof).*b1u) + eps*Idn
    dot(c, lu(A)\(-b))
end

for alpha in (100.0, 2000.0, 10000.0)
    field.alpha_max=alpha; field.rho.=rho_grid
    f1g, info = run_f1_fixed_steady(ev; itmax=25000, memory=100, atol=1e-12, rtol=1e-11)
    @printf("\nalpha=%.0f  GMRES f_1=%+.6e (res=%.1e, conv=%s)\n",
            alpha, f1g, info.residual, info.converged); flush(stdout)
    for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
        @printf("   eps=%.0e  reg-LU f_1=%+.6e\n", eps, f1_reg(alpha, eps)); flush(stdout)
    end
end
println("DONE"); flush(stdout)
