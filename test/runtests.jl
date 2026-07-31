# Regression tests for the FixedMesh exact-operator API.
# Run with:  julia --project=. test/runtests.jl   (or Pkg.test)
using VicinityOpt
using VicinityOpt.Geometry, VicinityOpt.Simulate, VicinityOpt.FixedMesh
using Test, Random

@testset "FixedMesh exact operator + adjoint" begin
    cfg = ChannelConfig(L_x=1.0, W=0.6)
    sim = SimConfig(n_harmonics=4, gamma_mc=100.0, gamma_mr=0.05, gamma_3=100.0, polydeg=1)
    tmp = mktempdir()
    geo = joinpath(tmp, "g.geo"); inp = joinpath(tmp, "g.inp")
    write_point_geo(geo; L=cfg.L_x, W=cfg.W, xPL=cfg.x_probe-cfg.L_probe/2,
                    xPR=cfg.x_probe+cfg.L_probe/2, wc=0.12, h=0.06)
    VicinityOpt.Mesh.geo_to_inp(geo, inp; verbose=false, structured=true)

    NX, NY = 12, 8                       # design grid COARSER than the mesh
    field = DensityField(cfg; nx=NX, ny=NY, alpha_max=2000.0)
    ev = FixedEvaluator(inp, field, sim)
    op = assemble_fixed_operator(ev)
    @test op.N > 0 && op.nx == NX && op.ny == NY

    Random.seed!(0); rg = clamp.(0.2 .+ 0.1 .* randn(NX, NY), 0, 1)

    # exact reg-LU must match the matrix-free GMRES solve
    f_lu = f1_exact(op, rg; alpha=2000.0)
    field.alpha_max = 2000.0; field.rho .= rg
    f_gm, _ = run_f1_fixed_steady(ev; itmax=40000, memory=120, atol=1e-11, rtol=1e-10)
    @test isapprox(f_lu, f_gm; rtol=1e-4)

    # adjoint gradient must match finite differences
    f0, g = f1_adjoint_grad(op, rg; alpha=2000.0)
    @test f0 ≈ f_lu
    for (ci, cj) in ((4, 4), (8, 4))
        e = 1e-6
        rp = copy(rg); rp[ci, cj] += e
        rm = copy(rg); rm[ci, cj] -= e
        fd = (f1_exact(op, rp; alpha=2000.0) - f1_exact(op, rm; alpha=2000.0)) / (2e)
        @test isapprox(g[ci, cj], fd; rtol=2e-3, atol=1e-6)
    end

    # analytic Fourier density path returns a finite f_1
    rpd = fourier_perdof(op; r0=0.20, C=[0.0, -0.02])   # circle + b1 dimple
    @test isfinite(f1_exact_perdof(op, rpd; alpha=2000.0))
end
