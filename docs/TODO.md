# MORKTensorNetworks — TODO / Open Items

Tracked follow-on work from the 2026-06-04 audit (`docs/AUDIT_2026-06-04.md`).
Mechanical findings (C1/C2/C3/H1/H3/H4/M3/M4/M5/M6/L4/L6) are already FIXED and not
listed here. This file tracks what remains.

## 🔴 Owner decisions (block follow-on work)

- [ ] **C4 / H5 — Package identity.** Is MORKTensorNetworks a *reference port* of
      Goertzel's paper (keep dense `Matrix`, cap input sizes, document scope, correct the
      status table) **or** a *PRIMUS substrate kernel* (sparse-output SpGEMM, CSR `(max,+)`
      ECAN SpMV, SoA + Bumper arena — effectively a rewrite of `shard/` + `ecan/` with
      AllocCheck gates)? Most performance findings depend on this answer.
- [ ] **H6 — ECAN identity.** `ECANTensorBridge` diverges from Core's ECAN on 4 points
      (no VLTI; single-tier rent vs Core's WA+AF; product Hebbian vs Core's affine
      `hebbian-conjunction`; no STI conservation vs `trade-sti!`). Either (a) align to
      Core's AV / WA+AF / affine semantics to make it a faithful acceleration, or
      (b) supply the missing §7 spec authorising the simplified tensor form.
- [ ] **M2 — O(1) Λ_s graft reattach.** Spec §2.5 promises an O(1) structural-sharing
      splice; `patch_and_reattach!` does per-path global writes instead. Decide: implement
      the `wz_graft!`-based reattach (record a write-zipper continuation in `capture_shard`),
      or drop the O(1) claim from the §2 docstrings/README.

## 🟠 Feature gaps vs the paper (from the deep paper-vs-code pass)

- [ ] **N4 — PC local learning incomplete (§6.4).** Predictive-coding training updates
      only `W_down` + `alpha`; `W_Q/W_K/W_V` (attention routing) and `W1/W2` (FFN) are
      never updated, though §6.4 calls for "local Hebbian for Q/K/V and projection
      matrices." Needs a defined local error signal for within-level weights (the paper's
      "rough notes" don't specify one). Characterization test pins the current behaviour;
      flip it when implemented. Don't fabricate a rule.
- [ ] **N1 — Cost ≡ MinPlus.** `CostSemiring` is behaviourally identical to
      `MinPlusSemiring`. Consolidate onto MinPlus, or give Q_cost genuinely distinct
      semantics if Occam-complexity ordering needs them. (Public API — decide before removing.)
- [ ] **N3 — real halo join.** `halo_boundary_join` computes the full dense join (correct
      for in-memory `A`/`B`); a genuine halo (shards holding partial `A` + boundary slice)
      is only meaningful once C4 goes "distributed kernel."

## 🟡 Conditional on the C4 decision (only if "PRIMUS kernel")

- [ ] **C4** — sparse-output SpGEMM (`gpu_semiring_spmm` currently allocates a dense output).
- [ ] **H5** — represent `ECANState.W` as CSR; route spreading through
      `gpu_semiring_spmv(MaxPlusSemiring(), …)`; Hebbian update over existing links only.
- [ ] **H3 (perf tail)** — the rebuilt `materialize!` is correct but still builds dense-ish
      Julia containers; SoA + Bumper arena + reusable key buffer per the audit's H3 note.
- [ ] **H1 (gate)** — add an AllocCheck/JET CI gate on `semiring_matmul` once identity is set.

## 🟠 From the per-category deep pass (verified, deferred)

- [ ] **SZ-1/SZ-2 — partition correctness.** `_partition_recursive!` caps at byte-depth 8
      (can cut mid-atom) and OR's the depth clause with the cost clause, so a deep dense
      subtrie is emitted as one shard even when `cost > L_max` — violating the invariant
      partition claims. Use an atom-boundary-aware split + honor L_max.
- [ ] **SZ-3 — Capture Γ_s.** `capture_shard` records only the prefix, never a write-zipper
      continuation → root cause of the missing O(1) reattach (M2). Add a Γ_s field + record it.
- [ ] **SZ-4/SZ-5 — weighted reattach.** `patch_and_reattach!` writes `UNIT_VAL`, dropping
      `rec.value`; `:update` ≡ `:insert` (no-op for weights). Needs a weight-valued trie
      (ties to C4). Round-trip is currently lossy for weighted relations.
- [ ] **G2 — GPU threshold semiring-aware.** `threshold_kernel!` hardcodes `>thresh` (C2 bug
      on GPU). Thread the semiring szero + test `!isequal(x, szero)` before any GPU
      existential projection is wired.
- [ ] **F1/G1 — semiring-aware CSR.** `dense_to_csr` drops `0.0` (wrong for MaxPlus/PLN where
      0=sone). GPU `path_compose` is SumProduct/Boolean-only until this is semiring-parameterized.
- [ ] **CSJ-2/3/4 — cross-shard heuristics.** reshard threshold compares edge-count to area
      (≈never merges sparse); `cross_shard_join` auto-path hardcodes boundary=avg÷10 → always
      Batched (Halo/Reshard unreachable). Make boundary estimate data-driven.

## 🟢 Hygiene (low priority, safe anytime)

- [ ] **L1** — make `ShardZipper`/`ECANTensorBridge` proper submodules (currently bare
      `include`d; their `export`s leak into the parent). Cosmetic; fixes misleading
      include-order comments.
- [ ] **L8** — consolidate the two `dense_to_csr` (PathAlgebra.jl + GPULayout.jl) onto a
      shared CSR utility module.
- [ ] **L9** — make `elementwise_mask_kernel!` semiring-aware (szero on masked-out) IF
      `path_restrict` is ever GPU-dispatched (latent until then).
- [ ] **L7** — adopt Julia 1.12 features (`Memory{T}`, `OncePerProcess`) where they help;
      roadmap, not correctness.

## ✅ Done (this audit) — see `docs/AUDIT_2026-06-04.md` for the full table
C1 exports · C2 Heaviside · C3 universal guard · H1 type-widening · H2 GPU-Boolean contract ·
H3 materialize! rebuilt (symbolic relation) · H4 child-mask partition · M1/M2 documented ·
M3 tucker arity · M4 truncation→assert · M5 pyramid pow2 assert · M6 SVD warn ·
L3 dangling-cite note · L4 unused imports pruned · L5 dead-field documented · L6 alpha comment ·
L8/L9 documented. Verified: Pkg.test 116/116 (incl. Aqua), Blue fixed point.
