# Trustworthiness of `f_1`: Brinkman obstacle vs. clean vicinity device

*Audit, 2026-08-01. Companion to the "Fixed-mesh density method" and "Clean
vicinity device" sections of `CLAUDE.md`.*

This note answers three questions that came up while comparing the optimized
Brinkman-density **obstacle** shape (`runs/fixed_mesh/result_circle.jld2`) against
the obstacle-free **vicinity device** (`write_vicinity_geo`):

1. Why is the obstacle `f_1` untrustworthy?
2. Does the obstacle shape "touch every border," and is that the same defect?
3. Is the rectangular vicinity `f_1` really *higher* than the obstacle's?

**TL;DR.** The obstacle `f_1` **magnitude** is not a physical resistance — it is a
soft-Brinkman number set by the penalization strength `α`, by the mesh `h`, and by
how close the obstacle sits to the floating probes. The vicinity `f_1 ≈ 0.020` is
**~500–2000× smaller** than the obstacle's ~14–44, and that is exactly its virtue:
it is mesh-converged and `α`-free. The two also measure **different physics**
(geometric transverse transresistance vs. nonlocal spreading resistance), so their
magnitudes are not comparable. The obstacle **shape** (large circle + even-sine
dimples) is a rigorous symmetry result and is fine; only the **number** is not.

---

## Q1 — Why the obstacle `f_1` is untrustworthy

A Brinkman-density obstacle is a soft, penalized momentum sink, **not** a reflecting
wall. The four separately-observed problems are four faces of that one defect.
Ranked by impact on the reported magnitude:

### 1. Probe contamination — dominant (~90% of the historic inflation)
The floating-probe potential is read **on the probe wall itself**
(`_current_contact_potential(bc_probe_A/B, …)`, `src/fixed_mesh.jl:360-361`,
`574-577`). The obstacle is a smooth sigmoid density with a soft edge of
`width ≈ 0.02` (`paint_blob!` / `fourier_perdof`, `src/fixed_mesh.jl:156, 726`).
When that edge overlaps a probe wall it deposits `α·ρ > 0` **exactly where the
voltage is sampled**, damping current *at the probe* — a measurement artifact, not
a property of the obstacle. `scripts/fixed_mesh_contam.jl`: zeroing ρ in a band off
the `|y| = 0.3` walls drops `f_1` from **269 → 27**.

### 2. α-divergence — no hard-wall limit
`BrinkmanSource` damps the momentum harmonics by `α·ρ` and leaves the density mode
a0 undamped (`src/fixed_mesh.jl:129`). A true reflecting wall would be the
`α → ∞` limit only if the interior were properly sealed; it is not, so
`f_1 ∝ α` without saturating: **37 / 312 / 3046 / 15305** at
`α = 200 / 2e3 / 2e4 / 1e5`. The reported shape fixes `α = 2000`
(`scripts/fixed_mesh_circle_opt.jl:13`), so **its magnitude is a free knob.** The
`damp_a0 = true` "absorbing" variant (`src/fixed_mesh.jl:127`) *does* have a finite
limit, but it drives `f_1 → 0` — so neither variant is a physical reflecting wall.

### 3. h-divergence
Even the staircase-free analytic density gives `f_1` rising monotonically with
refinement: **236 → 269 → 299 → 333 → 344** over `h = .055 → .028`. A Brinkman
region has a momentum boundary layer of thickness `~ 1/√α`; at feasible meshes it
is under-resolved, and every refinement resolves more of it, so the number keeps
moving. A converged wall value never appears.

### 4. Singular operator / Tikhonov gauge
The undamped a0 mode is a null vector of the steady operator, so `A u = -b` is
non-unique. `f1_exact` / `f1_adjoint_grad` add `+εI` (`src/fixed_mesh.jl:664, 679`)
to select the min-norm solution. `f_1` is stable as `ε → 0` **for a given α and h**,
but this only fixes the gauge — it does nothing about (2) or (3).

**How they interrelate.** (4) makes the problem ill-posed; regularization defines
*a* number; but that number is then set by the unresolved boundary layer (3) and the
penalization strength (2), and is read off a probe the obstacle is bleeding onto (1).
Escaping all four requires a boundary-fitted `MaxwellWallBC` obstacle-as-hole —
currently blocked in this environment (singular operator needs a preconditioned
solver; direct assembly OOMs at N≈1e5).

---

## Q2 — Does the obstacle "touch every border"?

**Literally false — but it flags the right bug.** Computed directly from
`runs/fixed_mesh/result_circle.jld2` (`R0 = 0.2`, dominant `b2 = -0.0302`,
`b4 = +0.0101`, stored `f = 14.26`, `EXCL = 0.04`, `YCUT = 0.26`), integrating the
boundary `R(θ) = R0 + Σ(aₙ cos nθ + bₙ sin nθ)` centered at `(x_c = 0.5, 0)`:

| extent | value | nearest boundary | gap |
|---|---|---|---|
| body y-range | `[-0.206, +0.223]`, max\|y\|=0.223 | walls at \|y\|=0.30 | **0.077** |
| body x-range | `[0.304, 0.708]` | source/drain at x=0, 1.0 | **≈0.29** |
| area | 0.1275 | (matches quoted 0.127) | — |

- The obstacle **does not touch** the source/drain (clear by ~0.29) or the
  top/bottom walls (clear by ~0.077).
- The y-asymmetry (`-0.206` vs `+0.223`) is the sine-harmonic symmetry-breaking
  that carries the `f_1` signal.
- Probes A/B sit on the top/bottom walls at `x ∈ [0.425, 0.575]`
  (`src/fixed_mesh.jl:474-475`); the body's x-span `[0.304, 0.708]` **fully brackets
  both probes**, sitting directly beneath them, its top only 0.077 below.

**Causal link to Q1 — yes.** The optimizer caps `sum|C| ≤ CAP = YCUT - R0 = 0.06`
(`scripts/fixed_mesh_circle_opt.jl:16, 60-61`), and the saved shape hits it
**exactly** (`sum|C| = 0.0600`). It is pushing the obstacle as far toward the probe
walls as the mask permits — i.e. still climbing the **probe-contamination gradient**
of Q1. The `EXCL = 0.04` mask (hard-zeroing ρ for `|y| > 0.26`) removes the direct
on-probe density (the worst ~90%) but leaves the soft tail reaching to ≈0.25. So the
masking **bounds** but does not **remove** the pathology. "Touches every border" is
the visual signature of an objective that rewards crowding the probes.

---

## Q3 — Is the rectangular vicinity `f_1` "higher"?

**No — it is ~500–2000× *lower*, and that is the point.** At matched collision
regime (γ_mc ≈ 100, where the obstacle runs live):

| device | `f_1` | notes |
|---|---|---|
| Vicinity, adjacent pair, γ_mc=100, h→0 | **≈ 0.020** | mesh-converged (0.0535→0.0204, h=.065→.027) |
| Vicinity, γ_mc sweep (adjacent) | 6.10 / 1.09 / 0.250 / 0.067 / **0.0157** | γ_mc = 0.1 / 1 / 10 / 50 / 200 |
| Vicinity, γ_mc sweep (far drain) | 7.70 / 2.80 / 0.593 / 0.421 / 0.092 / **0.018** | γ_mc = 0.1 … 200 |
| Obstacle, "clean" | ~14–44 | stored `f = 14.26`; contamination-masked 27 |
| Obstacle, contaminated | up to 269 | — |
| Obstacle, α-tunable | up to 15305 | ∝ α |

The largest vicinity value anywhere (7.7, deep-ballistic γ_mc=0.1) is still below the
*smallest* clean obstacle number; at the obstacle's own γ_mc≈100–200 the vicinity
value is ~0.016–0.067. **The vicinity `f_1` is lower.**

**Why the magnitudes differ — apples to oranges:**

- **(a) Different observable — the γ_mc tell.** The obstacle `f_1` is *flat within
  ~2% across γ_mc = 1–300* → it is a **geometric transverse (Hall-like, B=0)
  transresistance** set by shape, not by the collision regime. The vicinity `f_1`
  varies ~400× across the same range → a genuine **nonlocal / spreading resistance**
  governed by carrier collisions. Comparing their magnitudes is meaningless.
- **(b) Different geometry / normalization.** Obstacle: point source/drain on
  left/right walls, probes on top/bottom, W=0.6, L=1.0. Vicinity: injector *pair* on
  the bottom edge (`CurrentContactBC(+I)`/`(−I)`), probes on the same edge, W=0.8,
  L=2.0, wc=0.1. Entirely different current paths → no reason the scalars should
  match.
- **(c) The obstacle number is inflated.** Its magnitude carries the Q1
  contamination + the free α-knob. It can be made almost any size.

**"Touches every border" for the rectangle is benign.** The vicinity device is a
plain rectangle with **no central object**; the current fills the whole sample
bounded by its walls (`MaxwellWallBC(1.0)` = diffuse / no-slip edges,
`src/fixed_mesh.jl:304`), exactly like a real Bandurin–Levitov Hall bar. The sample
occupying its own boundary is what it is *supposed* to do — the opposite of a
floating internal object crowding the *probe* walls. Same phrase, opposite meaning:
expected for the vicinity sample, diagnostic of a bug for the obstacle.

**Caveat.** The vicinity `f_1` stays positive and monotonic in γ_mc in both layouts
— the truncated-moment + diffuse-wall setup does **not** show the negative
vicinity-resistance (viscous backflow) hallmark. It is a trustworthy *spreading*
resistance; do **not** claim negative vicinity resistance from it.

---

## Bottom line

- **Do not quote the obstacle `f_1` magnitude.** It is α-tunable (∝α, no wall
  limit), h-divergent, and probe-contaminated. The *shape* (big circle + even-sine
  dimples b2/b4) is a rigorous symmetry result; the *number* is not.
- **"Touches every border" is not literal** (clears source/drain by ~0.29, walls by
  ~0.077) **but flags the real bug** — the optimizer saturates its wall-approach cap
  (`sum|C| = 0.06` exactly), still climbing the probe-contamination gradient; the
  mask bounds but does not cure it.
- **The vicinity `f_1 ≈ 0.020` is the trustworthy one** — mesh-converged and α-free
  (a real nonlocal spreading resistance), unlike the obstacle number.
- **For a trustworthy obstacle number**, leave the Brinkman-density representation
  for a boundary-fitted `MaxwellWallBC` obstacle-as-hole (needs a preconditioned
  solver / more memory than this environment allows). For hydrodynamic vicinity
  physics, the rectangular device is already the trustworthy path; to chase the
  negative vicinity resistance, revisit the diffuse-wall BC and the m≥2 moment
  closure.
