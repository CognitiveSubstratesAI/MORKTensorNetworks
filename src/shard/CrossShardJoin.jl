"""
    CrossShardJoin.jl — Strategies for joins spanning multiple shards

MORK-Tensor-Networks paper §2: when a join (semiring matmul) requires
data from multiple shards, three strategies handle the boundary:

1. Halo: small read-only boundary slice included with each shard
2. Batched boundary: second pass streaming boundary rows
3. Resharding: move common cross-cuts inside shard boundaries

These are host-side orchestration strategies; the actual compute
uses the same semiring kernels from SemiringKernels.jl.
"""
module CrossShardJoin

using ..Semirings: AbstractSemiring, oplus, otimes, szero

export HaloStrategy, BatchedBoundaryStrategy, ReshardStrategy,
       cross_shard_join, select_join_strategy

# ─── Strategy Types ──────────────────────────────────────────────────────────

abstract type JoinStrategy end

"""
    HaloStrategy

Include a small read-only boundary halo with each shard.
Fast but uses extra memory proportional to halo width.
Best for: narrow joins (small boundary overlap).
"""
struct HaloStrategy <: JoinStrategy
    halo_width::Int  # Number of boundary rows/cols to include
end
HaloStrategy() = HaloStrategy(2)

"""
    BatchedBoundaryStrategy

First pass: compute within-shard results.
Second pass: stream boundary slices and compute cross-shard contributions.
Best for: medium overlap, memory-constrained.
"""
struct BatchedBoundaryStrategy <: JoinStrategy end

"""
    ReshardStrategy

Re-partition so that common cross-cut nodes fall inside shard boundaries.
Most expensive upfront, but eliminates cross-shard joins entirely.
Best for: frequently-accessed cross-shard patterns.
"""
struct ReshardStrategy <: JoinStrategy end

# ─── Halo Join ───────────────────────────────────────────────────────────────

"""
    halo_join(sr, A_shard, B_full, shard_rows, halo_rows) → C_shard

Compute shard's portion of A * B, including halo rows from neighboring shards.
A_shard includes halo_rows extra rows beyond its owned shard_rows.
"""
function halo_join(sr::AbstractSemiring,
                   A_shard::AbstractMatrix, B_full::AbstractMatrix,
                   shard_rows::UnitRange{Int})
    m = length(shard_rows)
    n = size(B_full, 2)
    k = size(A_shard, 2)

    C = fill(szero(sr), m, n)
    for i in 1:m
        for j in 1:n
            acc = szero(sr)
            for p in 1:k
                acc = oplus(sr, acc, otimes(sr, A_shard[i, p], B_full[p, j]))
            end
            C[i, j] = acc
        end
    end
    return C
end

# ─── Batched Boundary Join ───────────────────────────────────────────────────

"""
    batched_boundary_join(sr, A, B, shard_ranges) → C

Two-pass join:
  Pass 1: compute within-shard blocks (no cross-shard data needed)
  Pass 2: add cross-shard boundary contributions
"""
function batched_boundary_join(sr::AbstractSemiring,
                                A::AbstractMatrix, B::AbstractMatrix,
                                shard_ranges::Vector{UnitRange{Int}})
    m, k = size(A)
    k2, n = size(B)
    C = fill(szero(sr), m, n)

    # Pass 1: within-shard contributions
    for rows in shard_ranges
        for i in rows
            for j in 1:n
                acc = szero(sr)
                for p in rows  # Only within-shard columns
                    if p <= k
                        acc = oplus(sr, acc, otimes(sr, A[i, p], B[p, j]))
                    end
                end
                C[i, j] = oplus(sr, C[i, j], acc)
            end
        end
    end

    # Pass 2: cross-shard boundary contributions
    for rows in shard_ranges
        for i in rows
            for j in 1:n
                acc = szero(sr)
                for p in 1:k
                    if !(p in rows)  # Only cross-shard columns
                        acc = oplus(sr, acc, otimes(sr, A[i, p], B[p, j]))
                    end
                end
                C[i, j] = oplus(sr, C[i, j], acc)
            end
        end
    end

    return C
end

# ─── Strategy Selection ──────────────────────────────────────────────────────

"""
    select_join_strategy(n_shards, avg_boundary_size, total_size) → JoinStrategy

Heuristic for choosing the best cross-shard join strategy:
  - Halo: boundary < 5% of shard size
  - Batched: boundary 5-20% of shard size
  - Reshard: boundary > 20% (restructure is worth it)
"""
function select_join_strategy(n_shards::Int, avg_boundary_size::Int, avg_shard_size::Int)
    if avg_shard_size == 0
        return HaloStrategy()
    end

    boundary_ratio = avg_boundary_size / avg_shard_size

    if boundary_ratio < 0.05
        return HaloStrategy(max(1, avg_boundary_size))
    elseif boundary_ratio < 0.20
        return BatchedBoundaryStrategy()
    else
        return ReshardStrategy()
    end
end

# ─── Unified Interface ───────────────────────────────────────────────────────

"""
    cross_shard_join(sr, A, B, shard_ranges; strategy=auto) → C

Perform a semiring join across shards using the selected strategy.
"""
function cross_shard_join(sr::AbstractSemiring,
                           A::AbstractMatrix, B::AbstractMatrix,
                           shard_ranges::Vector{UnitRange{Int}};
                           strategy::Union{JoinStrategy, Nothing}=nothing)
    if strategy === nothing
        avg_shard = length(shard_ranges) > 0 ?
            sum(length(r) for r in shard_ranges) ÷ length(shard_ranges) : size(A, 1)
        # Estimate boundary as edges between adjacent shards
        avg_boundary = max(1, avg_shard ÷ 10)
        strategy = select_join_strategy(length(shard_ranges), avg_boundary, avg_shard)
    end

    if strategy isa HaloStrategy
        # For halo: each shard includes its boundary rows
        return batched_boundary_join(sr, A, B, shard_ranges)  # fallback to batched for now
    elseif strategy isa BatchedBoundaryStrategy
        return batched_boundary_join(sr, A, B, shard_ranges)
    else
        # Reshard: just do full matmul (resharding is a structural change, not a compute strategy)
        return batched_boundary_join(sr, A, B, shard_ranges)
    end
end

end # module
