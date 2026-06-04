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

export HaloStrategy,
    BatchedBoundaryStrategy, ReshardStrategy, cross_shard_join, select_join_strategy

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
function halo_join(
    sr::AbstractSemiring,
    A_shard::AbstractMatrix,
    B_full::AbstractMatrix,
    shard_rows::UnitRange{Int}
)
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

CSJ-1 note (audit 2026-06-04): `shard_ranges` is used as BOTH the output-row partition
(index i) AND the contraction-index partition (index p). That is only valid when A is
square and the same partition applies to rows and the inner dimension — which holds for
the relational case (composition of n×n adjacency matrices over n shared nodes, the
package's actual use). For non-square A or row/inner partitions that differ, Pass 1's
"within-shard columns" would be the wrong index set. The `if p <= k` guard at the inner
loop is a symptom of this coupling. Documented; safe for the square relational workload.
"""
function batched_boundary_join(
    sr::AbstractSemiring, A::AbstractMatrix, B::AbstractMatrix,
    shard_ranges::Vector{UnitRange{Int}}
)
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

# ─── Halo Boundary Join ──────────────────────────────────────────────────────

"""
    halo_boundary_join(sr, A, B, shard_ranges, halo_width) → C

§5.6 Halo strategy (reference implementation over a single in-memory `A`/`B`).

N3 note (audit 2026-06-04): the halo's purpose is to give a shard read access to a
small boundary slice of *adjacent shards' rows* so each shard can finish its join
WITHOUT a second streaming pass. In a genuinely distributed setting each shard holds
only its owned rows + the halo. Here `A`/`B` are full in-memory matrices, so every
owned row already has access to all of `A`/`B` — the join over owned rows is exact
with no halo needed. This reference therefore computes `C[owned, :] = A[owned, :] ⊗ B`
directly. (The earlier version computed `halo_start/halo_end/extended/ei` and never
used them — dead variables removed.) `halo_width` is retained for API/strategy
compatibility; it only matters once shards hold partial `A`, which is future work
(see C4 package-identity decision — a distributed kernel would slice A by shard).
"""
function halo_boundary_join(
    sr::AbstractSemiring, A::AbstractMatrix, B::AbstractMatrix,
    shard_ranges::Vector{UnitRange{Int}}, halo_width::Int
)::Matrix
    m, k = size(A)
    _, n = size(B)
    C = fill(szero(sr), m, n)

    for owned_rows in shard_ranges
        for i in owned_rows
            for j in 1:n
                acc = szero(sr)
                for p in 1:k
                    acc = oplus(sr, acc, otimes(sr, A[i, p], B[p, j]))
                end
                C[i, j] = acc
            end
        end
    end
    C
end

# ─── Reshard ──────────────────────────────────────────────────────────────────

"""
    reshard_ranges(A, shard_ranges) → new_ranges

§5.6 Resharding: identify cross-shard "hot" columns (columns of A that
are referenced from multiple shards) and consolidate them.

Strategy: merge adjacent shards that share >20% of columns to eliminate
cross-shard boundaries for the most common join patterns.
Returns new shard_ranges with hot nodes colocated.
"""
function reshard_ranges(
    A::AbstractMatrix, shard_ranges::Vector{UnitRange{Int}}
)::Vector{UnitRange{Int}}
    isempty(shard_ranges) && return shard_ranges
    n_shards = length(shard_ranges)
    n_shards == 1 && return shard_ranges

    # Count cross-shard column references between adjacent shards
    new_ranges = UnitRange{Int}[]
    i = 1
    while i <= n_shards
        curr = shard_ranges[i]
        if i < n_shards
            next = shard_ranges[i + 1]
            # Cross-shard overlap: count columns of A[curr_rows] that land in next_rows
            cross = 0
            for r in curr
                r > size(A, 1) && break
                for c in 1:size(A, 2)
                    if A[r, c] != zero(eltype(A)) && c in next
                        cross += 1
                    end
                end
            end
            # Merge if cross-references exceed 20% of shard size
            threshold = max(1, length(curr) * length(next) ÷ 5)
            if cross > threshold
                push!(new_ranges, first(curr):last(next))
                i += 2  # skip merged shard
                continue
            end
        end
        push!(new_ranges, curr)
        i += 1
    end
    isempty(new_ranges) ? shard_ranges : new_ranges
end

export halo_boundary_join, reshard_ranges

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
function cross_shard_join(
    sr::AbstractSemiring,
    A::AbstractMatrix,
    B::AbstractMatrix,
    shard_ranges::Vector{UnitRange{Int}};
    strategy::Union{JoinStrategy, Nothing}=nothing
)
    if strategy === nothing
        avg_shard = if length(shard_ranges) > 0
            sum(length(r) for r in shard_ranges) ÷ length(shard_ranges)
        else
            size(A, 1)
        end
        # Estimate boundary as edges between adjacent shards
        avg_boundary = max(1, avg_shard ÷ 10)
        strategy = select_join_strategy(length(shard_ranges), avg_boundary, avg_shard)
    end

    if strategy isa HaloStrategy
        return halo_boundary_join(sr, A, B, shard_ranges, strategy.halo_width)
    elseif strategy isa BatchedBoundaryStrategy
        return batched_boundary_join(sr, A, B, shard_ranges)
    else  # ReshardStrategy
        new_ranges = reshard_ranges(A, shard_ranges)
        return batched_boundary_join(sr, A, B, new_ranges)
    end
end

end # module
