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

## 🟡 Conditional on the C4 decision (only if "PRIMUS kernel")

- [ ] **C4** — sparse-output SpGEMM (`gpu_semiring_spmm` currently allocates a dense output).
- [ ] **H5** — represent `ECANState.W` as CSR; route spreading through
      `gpu_semiring_spmv(MaxPlusSemiring(), …)`; Hebbian update over existing links only.
- [ ] **H3 (perf tail)** — the rebuilt `materialize!` is correct but still builds dense-ish
      Julia containers; SoA + Bumper arena + reusable key buffer per the audit's H3 note.
- [ ] **H1 (gate)** — add an AllocCheck/JET CI gate on `semiring_matmul` once identity is set.

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
