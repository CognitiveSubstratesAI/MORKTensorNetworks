# MORKTensorNetworks.jl

[![CI](https://github.com/CognitiveSubstratesAI/MORKTensorNetworks/actions/workflows/CI.yml/badge.svg)](https://github.com/CognitiveSubstratesAI/MORKTensorNetworks/actions/workflows/CI.yml)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://cognitivesubstratesai.github.io/MORKTensorNetworks/stable/)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://cognitivesubstratesai.github.io/MORKTensorNetworks/dev/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Julia 1.12+](https://img.shields.io/badge/Julia-1.12%2B-blue)](https://julialang.org)

> From Path Algebra in MORK to Tensor Logic on GPUs

Julia implementation of ["From Path Algebra in MORK to Tensor Logic on GPUs"](https://github.com/trueagi-io/MORK) (Goertzel, October 2025). Standalone package on top of [MORK.jl](https://github.com/CognitiveSubstratesAI/MORK) and [PathMap.jl](https://github.com/CognitiveSubstratesAI/PathMap).

## What this is

MORK stores knowledge as a large shared trie (PathMap). This package bridges that trie to GPU-accelerated tensor operations by:

1. **Path algebra → tensor logic** — relation R(x,y) becomes sparse matrix R[x,y]; composition becomes SpGEMM; projection becomes reduction
2. **ShardZipper** — extracts bounded trie pieces (shards) into flat GPU-friendly CSR/BCSR arrays, runs kernels, reattaches results in O(1)
3. **Semiring-parameterized kernels** — one kernel set covers Boolean reachability, path counting, Viterbi best-path, and PLN truth values
4. **HRT** — Hierarchical Resolution Transformer mapped onto MORK + ShardZipper
5. **ECAN tensor bridge** — STI spreading as (max,+) matmul; Hebbian weight updates; attention fund rent/wage

## Package structure

```
src/
  core/
    Semirings.jl          # §3.5  AbstractSemiring + Boolean/SumProduct/MaxPlus/MinPlus/PLN/Cost
    PathAlgebra.jl        # §1,§3 compose, union, intersect, restrict, project, viterbi, k-hop
  gpu/
    SemiringKernels.jl    # §5.2  KernelAbstractions.jl GPU kernels (vendor-neutral)
    GPULayout.jl          # §5.1  CSR + BCSR array construction from trie shards
  shard/
    ShardZipper.jl        # §2    6-step: partition→capture→materialize→compute→patch→adapt
    CrossShardJoin.jl     # §5.6  halo / batched-boundary / resharding strategies
  decomp/
    TuckerDecomposition.jl # §5.4 2D + 3D Tucker densification (HOOI)
  hrt/
    HRT.jl                # §6    Multi-resolution pyramid + cross-resolution attention + gated fusion
    PredictiveCodingTrainer.jl # §6.4 Local Hebbian training (no global backprop)
  ecan/
    ECANTensorBridge.jl   # §7.3  ECAN STI spreading + Hebbian + attention fund
```

## Core concepts

### Path algebra → tensor logic (§1–§3)

| Path operation | Tensor formula |
|---|---|
| Composition R∘S | `T[x,z] = H(Σ_y R[x,y]·S[y,z])` |
| k-hop reachability | `Reach_k = H(R^k)` Boolean |
| Label restriction | `R' = R ⊙ M_L` |
| Union / intersection | `H(R+S)` / `R∧S` |
| Viterbi best path | `Score[u,v] = max_y(W[u,y]+W[y,v])` |
| Existential (∃y)Φ | `E[x] = H(Σ_y Φ[x,y])` |
| Universal (∀y)Ψ | `U[x] = 1-H(Σ_y(1-Ψ[x,y]))` |

### ShardZipper 6-step workflow (§2)

```
1. Partition  — split trie by hashed prefixes (each shard ≤ L_max)
2. Capture    — detach shard, record zipper Γ_s for O(1) reattach
3. Materialize — build full-prefix node graph + manual CSR (all path bytes)
4. Compute    — run GPU kernels, emit patch records only (no trie mutation)
5. Patch & Reattach — apply patches, splice via Γ_s in O(1)
6. Adapt      — reshard if too large or chatty
```

### Semirings (§3.5)

```julia
BooleanSemiring()      # (OR, AND)  — reachability
SumProductSemiring()   # (+, ·)     — path counting
MaxPlusSemiring()      # (max, +)   — Viterbi / best path
MinPlusSemiring()      # (min, +)   — shortest paths
PLNSemiring()          # PLN truth value algebra
CostSemiring()         # cost / tropical
```

### GPU kernels (§5.2)

All kernels use [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) — vendor-neutral. Default `backend=CPU()` for testing; pass `backend=CUDABackend()`, `MetalBackend()`, etc. for real GPU dispatch without changing any code.

```julia
gpu_semiring_spmv(MaxPlusSemiring(), rowptr, colval, nzval, x; backend=CUDABackend())
gpu_threshold(output, input, 0.0f0; backend=CUDABackend())
```

### Cross-shard join strategies (§5.6)

| Strategy | When | How |
|---|---|---|
| `HaloStrategy(width)` | Narrow boundary (<5% of shard) | Extend each shard with `width` halo rows from adjacent shards |
| `BatchedBoundaryStrategy()` | Medium overlap (5–20%) | Pass 1: within-shard; Pass 2: cross-shard boundary |
| `ReshardStrategy()` | Wide overlap (>20%) | Merge adjacent shards with >20% cross-column references |

### ECAN tensor bridge (§7.3)

```julia
state = ECANState(n)
state.W = ecan_build_weight_matrix(links, n)

# §7.3.1: STI spreading as (max,+) matmul
# STI_new[x] = max_y(W[x,y] + STI[y])
ecan_sti_spread!(state; decay=0.9f0)

# §7.3.2: Hebbian weight update  ΔW[x,y] = η × STI[x] × STI[y]
ecan_hebbian_update!(state; η=0.01f0)

# §7.3.3: Attention fund (rent/wage)
rent = ecan_collect_rent!(state; af_threshold=0.5f0, rent_rate=0.1f0)
ecan_distribute_wages!(state, rent)
```

## Quick start

```julia
using MORKTensorNetworks, MORK

# Path algebra
sr     = SumProductSemiring()
Sister = Float32[0 1 0; 0 0 0; 0 0 0]
Parent = Float32[0 0 0; 0 0 1; 0 0 0]
Aunt   = path_compose(sr, Sister, Parent)   # Aunt[1,3] = 1.0

# Viterbi best 2-hop path
W     = Float32[0 1 0; 0 0 2; 0 0 0]
Score = path_viterbi(W)                     # Score[1,3] = 3.0

# Tucker densification (2D and 3D)
C2, M2, N2, err2      = tucker_decompose_2d(A, 4)
C3, M3, N3, P3, err3  = tucker_decompose_3d(A3, 4, 4, 4)

# ShardZipper over MORK space
s = new_space()
space_add_all_sexpr!(s, "(edge a b) (edge b c)")
run_all_shards!(s, my_kernel!; l_max=100)

# HRT
cfg    = HRTConfig(n_tokens=64, n_levels=3, d_model=128)
params = init_hrt(cfg)
state  = init_state(cfg)
hrt_forward!(state, params, cfg)

# ECAN
ecan = ECANState(100)
ecan.W = ecan_build_weight_matrix(links, 100)
ecan_sti_spread!(ecan)
ecan_hebbian_update!(ecan)
```

## Dependencies

| Package | Role |
|---|---|
| [MORK.jl](https://github.com/CognitiveSubstratesAI/MORK) | Space, exec atoms, sinks |
| [PathMap.jl](https://github.com/CognitiveSubstratesAI/PathMap) | Trie, zipper, lattice algebra |
| KernelAbstractions.jl | Vendor-neutral GPU kernels (CUDA/ROCm/Metal/oneAPI) |
| LinearAlgebra | SVD, norm (Tucker HOOI, HRT) |
| SparseArrays | CSR utilities |

## Relation to other packages

```
PathMap               ← trie substrate
    ↑
MORK                  ← exec atom engine
    ↑
MORKTensorNetworks    ← tensor logic + ShardZipper + HRT + ECAN bridge
    ↑
MorkSupercompiler     ← query optimization (Rule-of-64 fix)
    ↑
HPC                   ← MPI peer-to-peer distributed spaces
```

## Implementation status

| Component | Paper § | Status | Notes |
|---|---|---|---|
| Semirings (6 instances) | §3.5 | ✅ | Boolean, SumProduct, MaxPlus, MinPlus, PLN, Cost |
| PathAlgebra (8 ops) | §1, §3 | ✅ | compose, union, intersect, restrict, project, reachability, viterbi, count/universal |
| SemiringKernels | §5.2 | ✅ | KernelAbstractions @kernel; real GPU dispatch via backend param |
| GPULayout CSR | §5.1 | ✅ | Manual row-major CSR build (not Julia sparse() which is CSC) |
| GPULayout BCSR | §5.1 | ✅ | Block CSR; zero-block dropping; exact round-trip |
| ShardZipper (6 steps) | §2 | ⚠️ | `materialize!` decodes the symbolic relation (audit H3 fix); reattach is per-path writes, NOT the O(1) Λ_s graft (M2) |
| CrossShardJoin — Halo | §5.6 | ✅ | Halo rows fetched from adjacent shards |
| CrossShardJoin — Batched | §5.6 | ✅ | Two-pass: within then cross-shard |
| CrossShardJoin — Reshard | §5.6 | ✅ | Merges adjacent shards on >20% cross-column overlap |
| TuckerDecomposition 2D | §5.4 | ✅ | ALS: A ≈ M·C·N^T |
| TuckerDecomposition 3D | §5.4 | ✅ | HOOI: A_ijk ≈ Σ C_pqr·M_ip·N_jq·P_kr |
| HRT pyramid | §6 | ✅ | RTB (self-attn+FFN), down-project, cross-attn, gated fusion, recon loss |
| PredictiveCodingTrainer | §6.4 | ✅ | Local Hebbian updates, no global backprop |
| ECAN STI spreading | §7.3.1 | ⚠️ | (max,+) matmul; diverges from Core ECAN — no STI conservation (audit H6) |
| ECAN Hebbian weights | §7.3.2 | ⚠️ | product rule ΔW=η·STI·STI; Core uses affine `hebbian-conjunction` (audit H6) |
| ECAN attention fund | §7.3.3 | ⚠️ | single-tier STI rent; Core is two-tier WA+AF on STI+LTI, with VLTI (audit H6) |

**Tests: `Pkg.test` 116/116 (107 functional + 9 Aqua), warm-REPL 107/107.**
(The earlier "49/49" claim was stale — see `docs/AUDIT_2026-06-04.md`.)

## Documentation & open items

- **[Architecture](docs/src/architecture.md)** — layered design (4-layer stack + 5 categories)
  with both diagrams, mapping each component to its source file + audit status. Published via
  Documenter → GitHub Pages (`docs/make.jl`, `.github/workflows/Documenter.yml`).
- **`docs/AUDIT_2026-06-04.md`** — full finding-by-finding audit closeout (hybrid: JET +
  Aqua + spec-diff + external audit reconciliation).
- **`docs/TODO.md`** — tracked open items. ⚠️ rows above are documented owner decisions
  (package identity C4, ECAN-vs-Core H6, O(1) graft M2), not silent gaps.

## Source paper

Ben Goertzel. *From Path Algebra in MORK to Tensor Logic on GPUs: Rough Notes and a Hierarchical Resolution Transformer Example*. October 18, 2025.
