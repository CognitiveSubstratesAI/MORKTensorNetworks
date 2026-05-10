# MORKTensorNetworks.jl

> From Path Algebra in MORK to Tensor Logic on GPUs

Julia implementation of ["From Path Algebra in MORK to Tensor Logic on GPUs"](https://github.com/trueagi-io/MORK) (Goertzel, October 2025). Standalone package on top of [MORK.jl](https://github.com/sivaji1012/MORK) and [PathMap.jl](https://github.com/sivaji1012/PathMap).

## What this is

MORK stores knowledge as a large shared trie (PathMap). This package bridges that trie to GPU-accelerated tensor operations by:

1. **Expressing path algebra as tensor logic** — relation R(x,y) becomes sparse matrix R[x,y]; composition becomes SpGEMM; projection becomes reduction
2. **ShardZipper** — extracts bounded trie pieces (shards) into flat GPU-friendly CSR arrays, runs kernels, reattaches results in O(1)
3. **Semiring-parameterized kernels** — one kernel set covers Boolean reachability, path counting, Viterbi best-path, and PLN truth values
4. **HRT** — Hierarchical Resolution Transformer mapped onto MORK + ShardZipper

## Package structure

```
src/
  core/
    Semirings.jl          # §3.5  AbstractSemiring + Boolean/SumProduct/MaxPlus/MinPlus/PLN/Cost
    PathAlgebra.jl        # §1,§3 compose, union, intersect, restrict, project, viterbi, k-hop
  gpu/
    SemiringKernels.jl    # §5.2  KernelAbstractions.jl GPU kernels (vendor-neutral)
    GPULayout.jl          # §5.1  CSR/BCSR array construction from trie shards
  shard/
    ShardZipper.jl        # §2    6-step workflow: partition→capture→materialize→compute→patch→adapt
    CrossShardJoin.jl     # §5.6  halo / batched-boundary / resharding strategies
  decomp/
    TuckerDecomposition.jl # §5.4 Tucker A≈CMN densification for irregular sparsity
  hrt/
    HRT.jl                # §6    Multi-resolution pyramid + cross-resolution attention + gated fusion
    PredictiveCodingTrainer.jl # §6.4 Local Hebbian training (no global backprop)
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
3. Materialize — convert to CSR arrays (index, value, mask)
4. Compute    — run GPU kernels, emit patch records only (no trie mutation)
5. Patch & Reattach — apply patches, splice via Γ_s in O(1)
6. Adapt      — reshard if too large or chatty
```

### Semirings (§3.5)

```julia
sr = BooleanSemiring()      # (OR, AND) — reachability
sr = SumProductSemiring()   # (+, ·)    — path counting
sr = MaxPlusSemiring()      # (max, +)  — Viterbi / best path
sr = MinPlusSemiring()      # (min, +)  — shortest paths
sr = PLNSemiring()          # PLN truth value algebra
sr = CostSemiring()         # cost / tropical
```

## Quick start

```julia
using MORKTensorNetworks, MORK

# Path algebra
sr = SumProductSemiring()
Sister = Float32[0 1 0; 0 0 0; 0 0 0]
Parent = Float32[0 0 0; 0 0 1; 0 0 0]
Aunt   = path_compose(sr, Sister, Parent)  # Aunt[1,3] = 1.0

# Viterbi best 2-hop path
W     = Float32[0 1 0; 0 0 2; 0 0 0]
Score = path_viterbi(W)                     # Score[1,3] = 3.0

# ShardZipper: GPU kernel over MORK space
s = new_space()
space_add_all_sexpr!(s, "(edge a b) (edge b c) (edge a c)")

run_all_shards!(s, my_kernel!; l_max=100)

# HRT
params = hrt_init(n_tokens=64, n_levels=3, d_model=128)
state  = HRTState(params)
hrt_forward!(state, params)
```

## Dependencies

| Package | Role |
|---|---|
| [MORK.jl](https://github.com/sivaji1012/MORK) | Space, exec atoms, sinks |
| [PathMap.jl](https://github.com/sivaji1012/PathMap) | Trie, zipper, PathMap{V} |
| KernelAbstractions.jl | Vendor-neutral GPU kernels (CUDA/ROCm/Metal/oneAPI) |

## Relation to other packages

```
PathMap          ← trie substrate
    ↑
MORK             ← exec atom engine
    ↑
MORKTensorNetworks  ← tensor logic + ShardZipper + HRT
    ↑
MorkSupercompiler   ← query optimization (Rule-of-64 fix)
    ↑
HPC                 ← MPI peer-to-peer distributed spaces
```

## Implementation status

| Component | Paper § | Status |
|---|---|---|
| Semirings (6 instances) | §3.5 | ✅ |
| PathAlgebra (8 ops) | §1, §3 | ✅ |
| SemiringKernels (GPU) | §5.2 | ✅ |
| GPULayout (CSR/BCSR) | §5.1 | ✅ |
| ShardZipper (6 steps) | §2 | ✅ |
| CrossShardJoin (3 strategies) | §5.6 | ✅ |
| TuckerDecomposition | §5.4 | ✅ |
| HRT pyramid | §6 | ✅ |
| PredictiveCodingTrainer | §6.4 | ✅ |
| ECAN tensor bridge | §5+ECAN | 🔜 |

## Source paper

Ben Goertzel. *From Path Algebra in MORK to Tensor Logic on GPUs: Rough Notes and a Hierarchical Resolution Transformer Example*. October 18, 2025.
