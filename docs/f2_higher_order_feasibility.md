# Feasibility of a higher-order (`f_2`) shape-optimization search

*Audit + empirical null-test, 2026-08-02. Synthesis of four parallel audits
(FermiSea code, transport physics, literature, current methodology) plus the
`scripts/fixed_mesh_nulltest.jl` result.*

## Bottom line

**`f_2` is exactly zero in the current code — not by accident, but by construction.**
FermiSea's `IsotropicFermiHarmonics2D` steady problem is *affine* in the state,
`rhs(u) = A·u + b` with the injected current entering only through the constant
`b ∝ I`. Hence `u(I) = I·u(1)` exactly, `ΔV = f_1·I`, and every higher coefficient
`f_2 = f_3 = … ≡ 0` to machine precision, **for any device shape**. A nonzero `f_2`
therefore **requires adding a genuine nonlinearity to the physics engine** — it is a
model change, not a solver tweak or a new objective closure. Searching for "optimal
`f_2`" on the present code would optimize numerical noise around zero.

The empirical null-test confirms this directly (see §5).

---

## 1. Why `f_2 ≡ 0` — the linearity is exact (FermiSea code audit)

Every term of the semidiscrete RHS is linear (homogeneous) in `u` except one
inhomogeneous boundary constant `∝ I`:
- **Streaming flux**: constant matrices `A_x, A_y` (`FermiSea .../isotropic_fermi_harmonics_2d.jl:68-132`).
- **Collision**: `LinearCollisionMatrix` = `-(W·u)`, constant `W` (`:246-284`).
- **Boundary conditions**: all affine in `u_inner`; the *only* nonzero constant is the
  drive `CurrentContactBC.current ∝ I` (`:877, 927-935`). `MaxwellWallBC`,
  `FloatingProbeBC`, `OhmicContactBC(0)` contribute no `I`-nonlinearity.
- **Brinkman obstacle source** (this repo): `-σ·u`, `σ=α·ρ(x)` independent of `u`
  (`src/fixed_mesh.jl:122-132`).

A grep for `nonlinear|newton|picard|convect|poisson|screen|joule|u*u` across FermiSea
`src/` finds nothing physical. **`MagneticFieldSource(B)` is *not* an `I`-nonlinearity**
— it is linear in `u`, set by the external field `B`; it can make `f_1 ≠ 0` (Hall-like)
but keeps `u ∝ I`, so `f_2` stays 0. There is **no** energy dependence, convective term,
self-consistent field, or nonlinear collision anywhere.

**Solvers are linear-only.** `run_f1_fixed_steady` (matrix-free GMRES),
`assemble_fixed_operator` + `f1_exact` (reg-LU), and `f1_adjoint_grad` (discrete
adjoint) all hard-assume `A·u = -b` with a *constant* linear functional `f_1 = cᵀu`
(`src/fixed_mesh.jl:386-703`). There is no Newton/Picard path. The one exception is the
`ROCK4` time-march (`simulate.jl:53-122`), which tolerates a nonlinear `rhs` but is slow
(~1000 s/solve).

## 2. What produces a nonzero `f_2` — recommended: the convective term (physics audit)

`f_2` needs a term that makes `A` depend on `u` (or `b` nonlinear in `I`). Candidates:

| mechanism | fit to this code |
|---|---|
| **Hydrodynamic convective `(u·∇)u`** (Navier–Stokes inertia → Bernoulli/rectification) | **Best.** Native to the hydrodynamic (large γ_mc) regime already used; intrinsically *geometric* so shape optimization is meaningful; a bilinear coupling of the a1/b1/a2/b2 moments. |
| Nonlinear / energy-dependent collision γ(u) | moderate; breaks constant-coefficient assembly; less physically motivated here |
| Self-consistent electrostatics / nonlinear screening | needs a coupled Poisson solve + new field |
| Joule heating + energy relaxation | needs an energy equation (larger NVARS) |
| Nonlinear Hall (Berry-curvature dipole) | **not available** — needs band-geometry the isotropic model lacks |
| Nonlinear contact `I–V` | easy but is *contact* physics, not device-*shape* physics |

**Recommendation (all three physics/lit/method audits agree): the convective
`(u·∇)u` term.** It is the leading nonlinearity of the same electron-fluid regime,
literature-backed (Hui–Oganesyan–Kim, arXiv:2010.00019, Bernoulli/streaming in graphene
hydrodynamics), and its linearization reuses the existing assembled operator as the
Newton/Picard Jacobian.

## 3. Symmetry — `f_2` needs a *different* asymmetry than `f_1` (physics audit)

- `f_1·I` is **odd** in current; the project's selection rule (only even-sine harmonics
  `b_2, b_4` and offset `y_c` carry `f_1`) is about **up–down (y) mirror** breaking
  (a transverse, Hall-like imbalance).
- `f_2·I²` is **even** in current (rectification); it requires breaking the
  **source–drain (x / inversion) mirror**. A device symmetric under source↔drain exchange
  forces `f_2 = 0`.

So an `f_2` optimizer must exploit a **fore–aft (source–drain) asymmetric** shape —
a wedge / ratchet / asymmetric-airfoil "flow-diode" geometry — a genuinely *different*
target than the `f_1` dimples. A shape optimized for `f_1` generically gives `f_2 = 0`.

## 4. Literature grounding + reference correction (literature audit)

- **No literature-standard "`f_2`" vicinity coefficient exists** — the `f_1`/`f_2`
  notation is project-internal. The standard linear object is the Bandurin–Levitov
  *vicinity resistance* R_V; the standard higher-order objects are the **second-order /
  nonlinear-Hall conductivity** `χ_abc` (Sodemann–Fu, PRL 115, 216806 (2015)), the
  **nonreciprocity coefficient** `γ` (Tokura–Nagaosa, Nat. Commun. 9, 3740 (2018)), and
  the experimentally-measured **second-harmonic resistance `R_2ω`**.
- The closest physical analog for *this* electron fluid is the **convective/Bernoulli
  rectification** of Hui–Oganesyan–Kim (arXiv:2010.00019) — reinforcing §2.
- **Validation signatures a credible `f_2` must show** (mirroring how `f_1`'s trust came
  from mesh-convergence): (i) clean `I²` / `2ω` scaling separated from the linear
  background; (ii) sign control / vanishing under the controlling symmetry (→ 0 in the
  symmetric or linear/Ohmic limit); (iii) ε/h/moment convergence to a knob-free number.
- **README citation fix:** arXiv:2605.03030 (Farrell & Lucas, 2026) is a *real* paper and
  the correct reference for the linearized harmonic-moment *model*, but it is
  linear-response only and **does not define `f_1` or a vicinity resistance** — do not
  cite it "for the `f_1` expansion." The vicinity physics traces to
  Bandurin (Science 351, 1055 (2016)) / Levitov–Falkovich / Torre (PRB 92, 165433 (2015)).

## 5. Empirical confirmation — the linear-null-test (`scripts/fixed_mesh_nulltest.jl`)

Solved the obstacle-free vicinity device at currents `I ∈ {0.25, 0.5, 1, 2, 4}`, read the
**raw** probe drop `ΔV = V_A − V_B` (not divided by `I`), and fit
`ΔV = f_1 I + f_2 I² + f_3 I³`:

```
f_1 = +7.3257e-02
f_2 = -2.8e-16    |f_2|/|f_1| = 3.8e-15     <- machine precision
f_3 = +1.0e-16    |f_3|/|f_1| = 1.4e-15
ΔV/I constant to 15 digits across a 16× current range (spread 9.9e-16)
```

**`f_2` and `f_3` are zero to floating-point.** This (a) empirically proves the current
engine cannot produce a nonzero higher-order coefficient, and (b) validates the
multi-amplitude extraction+fit harness against a known-zero, so any curvature seen
*after* a nonlinearity is added is real physics, not a fit artifact. Re-run this as the
regression gate on any nonlinear build.

## 6. Methodology — what survives a nonlinear pivot (methodology audit)

**Reusable as-is** (physics-agnostic): geometry emitters (`write_vicinity_geo` etc.,
`src/fixed_mesh.jl:495-546`), the vicinity device, `FixedEvaluator` build (mesh + BC
wiring, `:287-321`), the optimizer drivers (`Optimize` ParticleSwarm, CMA-ES), the
`Objective`/`EvalState` scaffold, and the ε/h/γ convergence trust-harness pattern
(`scripts/fixed_mesh_vicinity.jl`). The `SolverInterface.evaluate` seam
(`src/simulate.jl:214-217`) means a nonlinear backend is a *new config type*; the whole
outer loop is untouched.

**Breaks under nonlinearity** (all in `src/fixed_mesh.jl:386-703`): the matrix-free
`LinearMap` GMRES, `assemble_fixed_operator` (probes the Jacobian only at `u=0`),
`f1_exact` (single reg-LU solve), `f1_adjoint_grad` (adjoint of a linear functional of a
linear solve), and the `f_1 = cᵀu` extraction.

**New work needed:** (1) the `(u·∇)u` source/flux term; (2) a Newton/Picard steady solve
reusing the reg-LU Jacobian per step (fallback: ROCK4 time-march); (3) higher-order
extraction via the amplitude sweep of §5 or the second-harmonic `V(2ω)`; (4) for
gradients, the nonlinear adjoint — *or* stay derivative-free (ParticleSwarm/CMA need no
gradient, so a search can start the moment the nonlinear forward solve works); (5) the
same ε/h/moment trust protocol, developed on the **obstacle-free / boundary-fitted**
device (never the α/h-divergent Brinkman obstacle — see
[`f1_trustworthiness_obstacle_vs_vicinity.md`](f1_trustworthiness_obstacle_vs_vicinity.md)).

---

## Recommended path

0. **[done]** Linear-null-test — confirms `f_2 ≡ 0` today; validates the harness.
1. **Decide scope of the nonlinear physics change** (the real fork — see below).
2. Add the `(u·∇)u` term; wrap the reg-LU solve in Newton/Picard; re-run the null-test —
   it must now show a *nonzero, convergent* `f_2`.
3. Optimize the **second-harmonic `V(2ω)`** (or DC rectification curvature) over
   **source–drain-asymmetric** shapes; validate by `I²` scaling + symmetry sign control +
   ε/h convergence.

The step-1 scope decision (which nonlinearity, where it lives — FermiSea core vs a source
term in this repo, which solver, and how far to take it) is a project/advisor call because
the convective term touches FermiSea's core flux kernel and its eigendecomposition-based
characteristic boundary machinery — the most invasive part of the engine.

---

## Addendum (2026-08-02) — nonlinear pipeline IMPLEMENTED and validated

Built the end-to-end nonlinear pipeline in a dedicated workspace (editable FermiSea
clone `~/cu/FermiSea-nl` on branch `nonlinear-convective`; this repo on branch
`nonlinear`, `main` unchanged).

**What was added**
- **`InertialStressSource`** (FermiSea `src/equations/isotropic_fermi_harmonics_2d.jl`):
  a Trixi source term for the *local* (algebraic) part of the convective coupling —
  `(a1,b1)~(jx,jy)` feed the m=2 stress harmonics via the traceless `j⊗j` tensor
  `da2 += λ(a1²−b1²)`, `db2 += λ(2 a1 b1)`. Quadratic in the state, `λ=0` = linear.
- **`solve_nonlinear_steady`** (`src/fixed_mesh.jl`): modified-Newton (chord) solve of
  `rhs_nl(u)=0`, reusing ONE reg-LU factorization of the linear operator (`op.A0+εI`) as
  a fixed Jacobian; the current sweep reuses one assembly via a forcing shift.
- **`FixedEvaluator(...; extra_sources=…)`** hook; **`scripts/fixed_mesh_nonlinear.jl`**
  driver (λ-sweep × I-sweep, fits `ΔV=f1·I+f2·I²+f3·I³`).

**Validation (vicinity device, h=0.07, M=4, γ_mc=100)**
- **λ=0 → f2 = 1.1e-17** (machine noise): recovers the linear null-test exactly. ✅
- **λ>0 → nonzero f2**, and **f2/λ = −0.01476 constant to 4 digits** over λ=1e-3…3e-2
  (slight droop to −0.01467 at λ=0.1 as higher-order-in-λ terms enter): f2 is linear in
  λ at leading order — the nonlinearity IS the source of f2. ✅
- Chord solver converges in 2–11 iterations (residual ~1e-11) at every λ and I. ✅
- **Caveat — f2 magnitude not yet mesh-converged**: at coarse h=.09/.07/.055 it reads
  −2.05/−1.48/−2.24 ×1e-4 (~35% scatter, no sign flips). A higher-order quantity needs
  finer meshes than f1 did; a proper finer-h convergence study is the next validation.

**Scope note.** This is the *local* inertial-stress nonlinearity — the correct `j⊗j`
tensor structure but not the full convective operator (the gradient `∇·` part is
flux-level and needs the mentor-coordinated FermiSea-core change). It is a genuine,
controllable nonlinearity sufficient to prove and de-risk the entire machinery
(nonlinear solver + f2 extraction + λ/I validation). Remaining for a *physical* f2:
(1) the flux-level convective term (with the mentor); (2) the finer-h convergence study;
(3) optimize over **source–drain-asymmetric** shapes (f2 needs inversion breaking).
