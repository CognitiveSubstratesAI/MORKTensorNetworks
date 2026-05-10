# MORKTensorNetworks.jl

> From Path Algebra in MORK to Tensor Logic on GPUs

Julia implementation of ["From Path Algebra in MORK to Tensor Logic on GPUs"](https://github.com/trueagi-io/MORK) (Goertzel, October 2025). Standalone package on top of [MORK.jl](https://github.com/sivaji1012/MORK) and [PathMap.jl](https://github.com/sivaji1012/PathMap).

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
| [MORK.jl](https://github.com/sivaji1012/MORK) | Space, exec atoms, sinks |
| [PathMap.jl](https://github.com/sivaji1012/PathMap) | Trie, zipper, lattice algebra |
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
| ShardZipper (6 steps) | §2 | ✅ | Full-prefix node graph; manual CSR; all 6 steps correct |
| CrossShardJoin — Halo | §5.6 | ✅ | Halo rows fetched from adjacent shards |
| CrossShardJoin — Batched | §5.6 | ✅ | Two-pass: within then cross-shard |
| CrossShardJoin — Reshard | §5.6 | ✅ | Merges adjacent shards on >20% cross-column overlap |
| TuckerDecomposition 2D | §5.4 | ✅ | ALS: A ≈ M·C·N^T |
| TuckerDecomposition 3D | §5.4 | ✅ | HOOI: A_ijk ≈ Σ C_pqr·M_ip·N_jq·P_kr |
| HRT pyramid | §6 | ✅ | RTB (self-attn+FFN), down-project, cross-attn, gated fusion, recon loss |
| PredictiveCodingTrainer | §6.4 | ✅ | Local Hebbian updates, no global backprop |
| ECAN STI spreading | §7.3.1 | ✅ | (max,+) matmul: STI_new[x] = max_y(W[x,y]+STI[y]) |
| ECAN Hebbian weights | §7.3.2 | ✅ | ΔW[x,y] = η·STI[x]·STI[y] |
| ECAN attention fund | §7.3.3 | ✅ | Rent = max(0, STI−threshold)×rate; wage ∝ STI |

**49/49 tests pass.**

## Source paper

Ben Goertzel. *From Path Algebra in MORK to Tensor Logic on GPUs: Rough Notes and a Hierarchical Resolution Transformer Example*. October 18, 2025.
