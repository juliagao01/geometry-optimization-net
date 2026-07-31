#!/usr/bin/env julia
# scripts/fixed_mesh_adjoint.jl  — STEP 2: free-form topology optimization
#   julia --threads=auto --project=. scripts/fixed_mesh_adjoint.jl
#
# Free-form per-cell density optimization of the obstacle via the DISCRETE
# ADJOINT. Everything is linear, which makes this exact and cheap:
#
#   forward:   A(ρ) u = -b,   A(ρ) = A0 + diag(d),  d_j = ρ(cell_j) * b1_j
#   objective: f1 = cᵀ u                 (probe potentials are linear in u)
#   adjoint:   Aᵀ λ = c
#   gradient:  ∂f1/∂ρ_cell = -Σ_{j∈cell} λ_j b1_j u_j   (ALL cells, 2 solves)
#
# A0 (streaming+collision+boundary Jacobian at ρ=0) and c are assembled ONCE by
# probing rhs! with unit vectors; b1 (Brinkman diagonal at unit density) is two
# matvecs. Then each descent step is a sparse LU + two solves — exact, no GMRES
# tolerance noise (crucial: the true signal here is ~1e-6).
#
# Four validation gates must pass before any optimization runs.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Trixi, FermiSea
using SparseArrays, LinearAlgebra
using Printf, JLD2, Random

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
GMC = get(ENV, "GMC", "30.0") |> x->parse(Float64, x)   # regime (set from regime sweep)
cfg = ChannelConfig(L_x=1.0, W=0.6, lc=0.04)
sim = SimConfig(n_harmonics=6, gamma_mc=GMC, gamma_mr=0.05, gamma_3=GMC, polydeg=1)
NX, NY = 24, 14            # design grid
VOLFRAC = 0.16            # target material fraction (volume constraint)
NITER   = 40

workdir = joinpath(@__DIR__, "..", "runs", "fixed_mesh"); mkpath(workdir)
geo = joinpath(workdir, "channel_sym.geo"); inp = joinpath(workdir, "channel_sym.inp")
write_channel_geo_symmetric(geo, cfg; h=cfg.lc)
VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)

field = DensityField(cfg; nx=NX, ny=NY, alpha_max=100.0)
ev = FixedEvaluator(inp, field, sim)
semi = ev.semi
mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
nvars = Trixi.nvariables(equations)
println("regime gamma_mc=$GMC  design $(NX)x$(NY)"); flush(stdout)

extract_f1() = begin
    _,_,dg,ca = Trixi.mesh_equations_solver_cache(semi)
    (FermiSea._current_contact_potential(ev.bc_probe_A, equations, dg, ca) -
     FermiSea._current_contact_potential(ev.bc_probe_B, equations, dg, ca)) / sim.I_source
end

clear!(field)
ode = Trixi.semidiscretize(semi, (0.0, 1.0))
N = length(ode.u0)
b = similar(ode.u0); Trixi.rhs!(b, zero(ode.u0), semi, 0.0)

# --------------------------------------------------------------------------
# assemble A0 (ρ=0) and c (f1 functional) in one probing pass
# --------------------------------------------------------------------------
println("assembling A0 + c  (N=$N) ..."); flush(stdout)
t0 = time()
Irow=Int[]; Jcol=Int[]; Val=Float64[]; c=zeros(N)
tmp=similar(b); ej=zeros(N); tol=1e-11
for j in 1:N
    ej[j]=1.0
    Trixi.rhs!(tmp, ej, semi, 0.0)      # = A0 ej + b
    c[j] = extract_f1()                 # f1(ej)=c_j  (linear, f1(0)=0)
    ej[j]=0.0
    @inbounds for i in 1:N
        v = tmp[i]-b[i]
        abs(v) > tol && (push!(Irow,i); push!(Jcol,j); push!(Val,v))
    end
end
A0 = sparse(Irow,Jcol,Val,N,N)
@printf("  A0 nnz=%d  ‖c‖=%.3e  (%.0fs)\n", nnz(A0), norm(c), time()-t0); flush(stdout)

# Brinkman unit diagonal: b1 = rhs_{ρ=1}(1) - rhs_{ρ=0}(1)  (diagonal operator)
onev = ones(N)
fill!(field.rho, 1.0); r1=similar(b); Trixi.rhs!(r1, onev, semi, 0.0)
clear!(field);          r0=similar(b); Trixi.rhs!(r0, onev, semi, 0.0)
b1 = r1 .- r0
@printf("  b1: nonzero dofs=%d (momentum modes)  min=%.2e\n",
        count(!iszero,b1), minimum(b1)); flush(stdout)

# dof -> design-cell map (same floor lookup as penalization)
cellof = zeros(Int, N)
begin
    tf = zeros(N); w = Trixi.wrap_array(tf, semi)
    nc = cache.elements.node_coordinates
    nn = size(w,2); nel = size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x = nc[1,ii,jj,e]; y = nc[2,ii,jj,e]
        ci = clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj = clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        lin = ci + (cj-1)*field.nx
        for v in 1:nvars; w[v,ii,jj,e] = lin; end
    end
    cellof .= round.(Int, tf)
end

# ρ (design grid) -> per-dof vector, same lookup
rho_perdof = zeros(N)
function set_rho_perdof!(rho_grid)
    w = Trixi.wrap_array(rho_perdof, semi)
    nc = cache.elements.node_coordinates
    nn = size(w,2); nel = size(w,4)
    for e in 1:nel, jj in 1:nn, ii in 1:nn
        x = nc[1,ii,jj,e]; y = nc[2,ii,jj,e]
        ci = clamp(floor(Int,(x-field.x0)/field.dx)+1,1,field.nx)
        cj = clamp(floor(Int,(y-field.y0)/field.dy)+1,1,field.ny)
        for v in 1:nvars; w[v,ii,jj,e] = rho_grid[ci,cj]; end
    end
    rho_perdof
end

# forward + adjoint + gradient for a design grid
function fwd_adj(rho_grid)
    set_rho_perdof!(rho_grid)
    A = A0 + spdiagm(0 => rho_perdof .* b1)
    F = lu(A)
    u = F \ (-b)
    f1 = dot(c, u)
    λ = F' \ c
    g = zeros(field.nx*field.ny)
    @inbounds for j in 1:N
        gj = -λ[j]*b1[j]*u[j]
        gj != 0.0 && (g[cellof[j]] += gj)
    end
    return f1, reshape(g, field.nx, field.ny), u
end

# --------------------------------------------------------------------------
# VALIDATION GATES
# --------------------------------------------------------------------------
println("\n=== validation ==="); flush(stdout)
Random.seed!(1)
allpass = true

# V1: A0 matvec vs rhs (ρ=0)
clear!(field)
vr = randn(N)
Trixi.rhs!(tmp, vr, semi, 0.0); ref = tmp .- b
err1 = norm(A0*vr - ref)/norm(ref)
p1 = err1 < 1e-9; allpass &= p1
@printf("V1 assembled A0 vs rhs:       rel err %.2e  %s\n", err1, p1 ? "PASS":"FAIL"); flush(stdout)

# V2: full operator A0+diag(ρ·b1) vs rhs at random ρ
rgrid = rand(field.nx, field.ny)
set_rho_perdof!(rgrid)
# set field.rho so rhs uses the same ρ
field.rho .= rgrid
Trixi.rhs!(tmp, vr, semi, 0.0); ref2 = tmp .- b
opv = (A0 + spdiagm(0=>rho_perdof .* b1)) * vr
err2 = norm(opv - ref2)/norm(ref2)
p2 = err2 < 1e-9; allpass &= p2
@printf("V2 A(ρ) vs rhs (random ρ):    rel err %.2e  %s\n", err2, p2 ? "PASS":"FAIL"); flush(stdout)

# V3: forward LU f1 vs GMRES steady solver, same ρ
f1_lu, _, _ = fwd_adj(rgrid)
field.rho .= rgrid
f1_gmres, _ = run_f1_fixed_steady(ev; itmax=20000, memory=100, atol=1e-11, rtol=1e-10)
err3 = abs(f1_lu - f1_gmres)/max(abs(f1_gmres),1e-12)
p3 = err3 < 1e-4; allpass &= p3
@printf("V3 forward LU vs GMRES:       f1_lu=%+.4e f1_gmres=%+.4e rel %.2e  %s\n",
        f1_lu, f1_gmres, err3, p3 ? "PASS":"FAIL"); flush(stdout)

# V4: adjoint gradient vs finite differences on a few cells
f1_0, g0, _ = fwd_adj(rgrid)
maxrel = 0.0
for (ci,cj) in ((5,7),(12,7),(18,4),(12,10))
    ε = 1e-6
    rp = copy(rgrid); rp[ci,cj]+=ε; fp,_,_ = fwd_adj(rp)
    rm = copy(rgrid); rm[ci,cj]-=ε; fm,_,_ = fwd_adj(rm)
    fd = (fp-fm)/(2ε); ad = g0[ci,cj]
    rel = abs(fd-ad)/max(abs(fd),1e-12); maxrel=max(maxrel,rel)
    @printf("   cell(%2d,%2d) adjoint=%+.4e  FD=%+.4e  rel %.2e\n", ci,cj,ad,fd,rel)
end
p4 = maxrel < 1e-3; allpass &= p4
@printf("V4 gradient vs FD:            max rel %.2e  %s\n", maxrel, p4 ? "PASS":"FAIL"); flush(stdout)

if !allpass
    println("\nVALIDATION FAILED — not running optimization."); flush(stdout); exit(1)
end
println("\nALL VALIDATION PASSED — running free-form optimization\n"); flush(stdout)

# --------------------------------------------------------------------------
# volume-constrained projected gradient ascent (maximize f1)
# --------------------------------------------------------------------------
# 3x3 density filter (regularization) — linear & symmetric, so chain rule is the
# same filter applied to the gradient.
function smooth(M)
    nx,ny=size(M); O=similar(M)
    @inbounds for j in 1:ny, i in 1:nx
        s=0.0;n=0
        for dj in -1:1, di in -1:1
            ii=i+di; jj=j+dj
            (1<=ii<=nx && 1<=jj<=ny) || continue
            s+=M[ii,jj]; n+=1
        end
        O[i,j]=s/n
    end
    O
end
# project ρ to satisfy mean(ρ)=VOLFRAC with box [0,1] via bisection on a shift.
function project_volume(M, V)
    lo,hi = -1.0, 1.0
    for _ in 1:60
        mid=(lo+hi)/2
        mean(clamp.(M .+ mid, 0, 1)) > V ? (hi=mid) : (lo=mid)
    end
    clamp.(M .+ (lo+hi)/2, 0, 1)
end

rho_grid = fill(VOLFRAC, field.nx, field.ny)   # uniform start at target volume
history=NamedTuple[]
step=0.3
best_f1=-Inf; best_rho=copy(rho_grid)
for it in 1:NITER
    rp = smooth(rho_grid)
    f1, g, _ = fwd_adj(rp)
    g = smooth(g)                                  # chain rule through filter
    gd = g ./ (maximum(abs, g) + 1e-30)
    trial = project_volume(rho_grid .+ step .* gd, VOLFRAC)
    ft,_,_ = fwd_adj(smooth(trial))
    if ft >= f1
        rho_grid = trial
        f1 > best_f1 && (best_f1=f1; best_rho=copy(rho_grid))
        ft > best_f1 && (best_f1=ft; best_rho=copy(trial))
    else
        step *= 0.6
        rho_grid = trial   # still accept (projected) to keep moving
    end
    push!(history,(it=it,f1=ft,vol=mean(rho_grid),step=step))
    @printf("it %2d  f1=%+.5e  vol=%.3f  step=%.3f\n", it, ft, mean(rho_grid), step); flush(stdout)
    step < 1e-3 && break
end

@printf("\nBEST free-form f1 = %+.6e  (vol=%.3f)\n", best_f1, mean(best_rho))
# write best design into field & save
field.rho .= best_rho
@save joinpath(workdir,"result_adjoint.jld2") best_rho best_f1 history cfg sim GMC NX NY
println("wrote result_adjoint.jld2")

# ASCII of the optimized free-form obstacle
chars=" .:-=+*#%@"
println("\nOptimized free-form density (top=+W/2):")
for j in field.ny:-1:1
    print("  ")
    for i in 1:field.nx
        k=clamp(floor(Int,best_rho[i,j]*(length(chars)-1))+1,1,length(chars))
        print(chars[k])
    end
    println()
end
println("DONE"); flush(stdout)
