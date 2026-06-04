# MORKTensorNetworks.jl

A Julia **reference port** of Ben Goertzel's *"From Path Algebra in MORK to Tensor Logic
on GPUs: Rough Notes and a Hierarchical Resolution Transformer Example"* (Oct 18 2025).

MORKTensorNetworks bridges MORK's **symbolic path algebra** (relations stored as paths in a
[PathMap](https://github.com/CognitiveSubstratesAI/PathMap) trie) to **tensor logic** —
expressing joins, projections, and quantifiers as semiring matrix operations that map onto
GPU kernels via [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

It provides:

- **Semirings** (`§3.5`) — Boolean, SumProduct, MaxPlus, MinPlus (+ PLN/Cost extensions),
  each with the algebraic laws verified.
- **Path algebra** (`§1`/`§3`) — `path_compose`, `path_union`, `path_intersect`,
  `path_restrict`, `path_project`, `path_viterbi`, `path_count`, `path_universal`, with a
  semiring-aware Heaviside.
- **ShardZipper** (`§2`) — partition the trie into shards, materialize the **symbolic
  relation** `R(src,dst)` as CSR (Rule-of-64 symbol decode), compute, patch & reattach.
- **GPU optimization** (`§5`) — sparse semiring SpGEMM/SpMV, fused reductions, masks, and
  Tucker densification for low-rank relations.
- **HRT** (`§6`) — the Hierarchical Resolution Transformer pyramid with bidirectional
  cross-resolution attention, gated fusion, and predictive-coding (local Hebbian) training.

## Status

This is a faithful **reference port** of the paper's algebra (correctness over scale); the
dense→sparse-output kernel rewrite is a workload-driven future decision. See
[`docs/AUDIT_2026-06-04.md`](https://github.com/CognitiveSubstratesAI/MORKTensorNetworks/blob/main/docs/AUDIT_2026-06-04.md)
for the full finding-by-finding audit and
[`docs/TODO.md`](https://github.com/CognitiveSubstratesAI/MORKTensorNetworks/blob/main/docs/TODO.md)
for tracked open items.

Tests: `Pkg.test` 129/129 (incl. Aqua quality checks).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/CognitiveSubstratesAI/MORKTensorNetworks")
```

See [Architecture](architecture.md) for the layered design and how each layer maps to the
source files.
