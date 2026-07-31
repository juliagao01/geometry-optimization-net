#!/usr/bin/env julia
# scripts/fixed_mesh_alpha.jl — EXACT (sparse-LU) f_1 vs obstacle solidity.
#   julia --threads=auto --project=. scripts/fixed_mesh_alpha.jl
#
# GMRES stalls for a solid (high-alpha) Brinkman obstacle. Assemble the linear
# operator once and solve EXACTLY with a sparse LU. Validate LU==GMRES at low
# alpha (where GMRES is reliable), then sweep alpha to the hard-wall limit to
# find the true converged vicinity signal for a fixed obstacle.
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra
using Printf

cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.04)
sim = SimConfig(n_harmonics=6, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel_sym.geo"); inp = joinpath(workdir, "channel_sym.inp")
write_channel_geo_symmetric(geo, cfg; h=cfg.lc)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)

field = DensityField(cfg; nx=40, ny=24, alpha_max=1.0)   # alpha handled explicitly below
ev = FixedEvaluator(inp, field, sim)
semi = ev.semi
mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)

extract_f1() = begin
    _,_,dg,ca = Trixi.mesh_equations_solver_cache(semi)
    (FermiSea._current_contact_potential(ev.bc_probe_A, equations, dg, ca) -
     FermiSea._current_contact_potential(ev.bc_probe_B, equations, dg, ca)) / sim.I_source
end

clear!(field)
ode = Trixi.semidiscretize(semi, (0.0,1.0)); N = length(ode.u0)
b = similar(ode.u0); Trixi.rhs!(b, zero(ode.u0), semi, 0.0)

# assemble A0 (rho=0) and c in one pass
println("assembling A0 + c (N=$N) ..."); flush(stdout); t0=time()
Ir=Int[]; Jc=Int[]; Vv=Float64[]; c=zeros(N); tmp=similar(b); ej=zeros(N); tol=1e-11
for j in 1:N
    ej[j]=1.0; Trixi.rhs!(tmp, ej, semi, 0.0); c[j]=extract_f1(); ej[j]=0.0
    @inbounds for i in 1:N
        v=tmp[i]-b[i]; abs(v)>tol && (push!(Ir,i);push!(Jc,j);push!(Vv,v))
    end
end
A0 = sparse(Ir,Jc,Vv,N,N)
@printf("  nnz=%d ‖c‖=%.3e (%.0fs)\n", nnz(A0), norm(c), time()-t0); flush(stdout)

# unit Brinkman diagonal at alpha=1: b1u = rhs_{rho=1,alpha=1}(1) - rhs_{rho=0}(1)
field.alpha_max = 1.0
onev=ones(N); fill!(field.rho,1.0); r1=similar(b); Trixi.rhs!(r1,onev,semi,0.0)
clear!(field); r0=similar(b); Trixi.rhs!(r0,onev,semi,0.0)
b1u = r1 .- r0

# per-dof density for a painted obstacle
rho_perdof = zeros(N)
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

# fixed obstacle (the strong case): yc=0.10, r=0.18
paint_blob!(field, cfg.x_c, 0.10, 0.18)   # fills field.rho grid
rho_grid = copy(field.rho)
set_rho_perdof!(rho_grid)

f1_lu(alpha) = begin
    A = A0 + spdiagm(0 => (alpha .* rho_perdof) .* b1u)
    u = lu(A) \ (-b)
    dot(c, u)
end

# --- validate LU vs GMRES at alpha=100 (GMRES reliable there) ---
field.alpha_max = 100.0; field.rho .= rho_grid
f1_g, info_g = run_f1_fixed_steady(ev; itmax=25000, memory=100, atol=1e-11, rtol=1e-10)
f1_l = f1_lu(100.0)
rel = abs(f1_l-f1_g)/max(abs(f1_g),1e-14)
verdict = rel < 1e-3 ? "PASS" : "FAIL"
@printf("\nVALIDATE alpha=100: LU=%+.5e  GMRES=%+.5e  rel=%.2e  %s\n",
        f1_l, f1_g, rel, verdict); flush(stdout)

println("\n=== EXACT f_1 vs alpha (obstacle yc=0.10 r=0.18, hard-wall limit) ===")
for a in (100.0, 500.0, 2e3, 1e4, 5e4, 2e5, 1e6)
    @printf("  alpha=%9.0f  f_1=%+.6e\n", a, f1_lu(a)); flush(stdout)
end
println("DONE"); flush(stdout)
