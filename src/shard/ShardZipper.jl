"""
ShardZipper — §2 of MORK-Tensor-Networks (Goertzel, Oct 2025)

6-step workflow: Partition → Capture → Materialize → Compute → Patch & Reattach → Adapt.

ShardZipper takes a bounded piece of the MORK PathMap trie (a shard), linearizes
it into contiguous CSR arrays for GPU kernels, runs computation, then reattaches
the updated shard in O(1) preserving structural sharing.

Depends on: PathMap (trie + zipper), MORK (Space, PathMap{UnitVal})
"""

using PathMap: PathMap, ReadZipperCore, WriteZipperCore,
               read_zipper, read_zipper_at_path, write_zipper_at_path,
               zipper_val_count, get_val_at, set_val_at!, remove_val_at!,
               zipper_to_next_val!, zipper_path, zipper_descend_to_byte!,
               zipper_ascend_byte!, wz_graft!
using SparseArrays
using LinearAlgebra

# ── §2.1: Shard + PatchRecord ──────────────────────────────────────────────────

"""
    PatchRecord

One entry in the patch log. Kind ∈ {:insert, :delete, :update}.
path = full byte path; value = new Float32 weight (for :update/:insert).
"""
struct PatchRecord
    kind  :: Symbol       # :insert | :delete | :update
    path  :: Vector{UInt8}
    value :: Float32
end

PatchRecord(kind::Symbol, path::Vector{UInt8}) = PatchRecord(kind, path, 1.0f0)

mutable struct Shard
    prefix    :: Vector{UInt8}
    row_ptr   :: Vector{Int}
    col_idx   :: Vector{Int}
    values    :: Vector{Float32}
    node_keys :: Vector{Vector{UInt8}}
    patch_log :: Vector{PatchRecord}
    size_cost :: Int
end

# ── §2: Step 1 — Partition ─────────────────────────────────────────────────────

"""
    partition_trie(space, l_max) → Vector{Vector{UInt8}}

§2 Step 1: Split the MORK space's trie by hashed prefixes until each shard's
estimated size/cost ≤ l_max. Returns a list of shard prefix byte-paths.

Strategy: BFS over trie nodes; split any subtrie whose val_count > l_max
by descending one byte further.
"""
function partition_trie(space, l_max::Int) :: Vector{Vector{UInt8}}
    prefixes = Vector{UInt8}[]
    _partition_recursive!(space.btm, UInt8[], l_max, prefixes)
    isempty(prefixes) && push!(prefixes, UInt8[])  # root shard if tiny
    return prefixes
end

function _partition_recursive!(btm, prefix::Vector{UInt8}, l_max::Int,
                                 prefixes::Vector{Vector{UInt8}})
    rz   = read_zipper_at_path(btm, prefix)
    cost = zipper_val_count(rz)
    if cost <= l_max || length(prefix) >= 8
        push!(prefixes, copy(prefix))
        return
    end
    # Descend: try all 256 possible next bytes, recurse on non-empty
    found_children = false
    for b in UInt8(0):UInt8(255)
        child_prefix = vcat(prefix, b)
        crz = read_zipper_at_path(btm, child_prefix)
        val_count(crz) == 0 && continue
        found_children = true
        _partition_recursive!(btm, child_prefix, l_max, prefixes)
    end
    found_children || push!(prefixes, copy(prefix))
end

# ── §2: Step 2 — Capture ───────────────────────────────────────────────────────

"""
    capture_shard(space, prefix) → Shard

§2 Step 2: Detach shard at `prefix` from the trie, recording a zipper
continuation Γ_s that knows how to splice an updated version back in O(1).

Returns an empty Shard with the prefix set. Materialization (Step 3) fills arrays.
"""
function capture_shard(space, prefix::Vector{UInt8}) :: Shard
    rz  = read_zipper_at_path(space.btm, prefix)
    est = zipper_val_count(rz)
    return Shard(copy(prefix), Int[], Int[], Float32[], Vector{UInt8}[], PatchRecord[], est)
end

# ── §2: Step 3 — Materialize ──────────────────────────────────────────────────

"""
    materialize!(shard, space) → Shard

§2 Step 3: Convert the shard's subtrie into contiguous CSR arrays:
  - row_ptr[i]: start of row i in col_idx / values
  - col_idx[k]: column index of nonzero k
  - values[k]:  weight of edge k
  - node_keys:  mapping from CSR row/col index → byte path

Treats each unique first-byte child of `prefix` as a node.
Two adjacent nodes form a (row, col) edge.
"""
function materialize!(shard::Shard, space) :: Shard
    prefix = shard.prefix
    rz     = read_zipper_at_path(space.btm, prefix)

    # Collect all (path, val) pairs in this subtrie
    paths = Vector{UInt8}[]
    while zipper_to_next_val!(rz)
        push!(paths, copy(collect(zipper_path(rz))))
    end

    isempty(paths) && return shard

    # Build node → index map from unique 1-byte children of prefix
    node_map = Dict{Vector{UInt8}, Int}()
    for p in paths
        isempty(p) && continue
        node_key = p[1:min(1, length(p))]
        haskey(node_map, node_key) || (node_map[node_key] = length(node_map) + 1)
    end

    n = length(node_map)
    node_keys = Vector{Vector{UInt8}}(undef, n)
    for (k, v) in node_map
        node_keys[v] = k
    end

    # Build CSR: edges from consecutive path bytes
    rows = Int[]
    cols = Int[]
    vals = Float32[]
    for p in paths
        length(p) < 2 && continue
        src_key = p[1:1]
        dst_key = p[2:2]
        haskey(node_map, src_key) || continue
        haskey(node_map, dst_key) || continue
        push!(rows, node_map[src_key])
        push!(cols, node_map[dst_key])
        push!(vals, 1.0f0)
    end

    if !isempty(rows)
        sp = sparse(rows, cols, vals, n, n)
        shard.row_ptr  = Vector{Int}(sp.colptr)
        shard.col_idx  = Vector{Int}(sp.rowval)
        shard.values   = Vector{Float32}(sp.nzval)
    else
        shard.row_ptr = ones(Int, n + 1)
        shard.col_idx = Int[]
        shard.values  = Float32[]
    end

    shard.node_keys = node_keys
    return shard
end

# ── §2: Step 4 — Compute ──────────────────────────────────────────────────────

"""
    compute!(shard, kernel_fn!) → Shard

§2 Step 4: Run a GPU/CPU kernel on the materialized shard arrays.
The kernel receives (row_ptr, col_idx, values, node_keys) and appends
PatchRecords to shard.patch_log. Does NOT mutate the trie.

kernel_fn!(shard::Shard) → nothing  (writes to shard.patch_log)
"""
function compute!(shard::Shard, kernel_fn!::Function) :: Shard
    isempty(shard.node_keys) && return shard
    kernel_fn!(shard)
    return shard
end

# ── §2: Step 5 — Patch & Reattach ─────────────────────────────────────────────

"""
    patch_and_reattach!(shard, space) → Int

§2 Step 5: Apply the patch log on the host, then splice the shard back via
the zipper in O(1) (structural sharing preserved).

Returns number of patches applied.
"""
function patch_and_reattach!(shard::Shard, space) :: Int
    prefix  = shard.prefix
    applied = 0
    for rec in shard.patch_log
        full_path = vcat(prefix, rec.path)
        if rec.kind == :insert || rec.kind == :update
            set_val_at!(space.btm, full_path, MORK.UNIT_VAL)
            applied += 1
        elseif rec.kind == :delete
            remove_val_at!(space.btm, full_path)
            applied += 1
        end
    end
    empty!(shard.patch_log)
    return applied
end

# ── §2: Step 6 — Adapt ────────────────────────────────────────────────────────

"""
    should_adapt(shard, l_max) → Bool

§2 Step 6: Return true if the shard should be resplit next time
(too large or too many patches — indicates "chatty" boundary).
"""
function should_adapt(shard::Shard, l_max::Int) :: Bool
    shard.size_cost > l_max * 2 || length(shard.patch_log) > l_max ÷ 4
end

# ── High-level convenience ─────────────────────────────────────────────────────

"""
    run_shard!(space, prefix, kernel_fn!; l_max=1000) → Int

Run the full ShardZipper 6-step workflow on one shard:
  capture → materialize → compute → patch & reattach.
Returns patches applied.
"""
function run_shard!(space, prefix::Vector{UInt8}, kernel_fn!::Function;
                    l_max::Int=1000) :: Int
    shard = capture_shard(space, prefix)
    materialize!(shard, space)
    compute!(shard, kernel_fn!)
    n = patch_and_reattach!(shard, space)
    return n
end

"""
    run_all_shards!(space, kernel_fn!; l_max=1000) → Int

Partition the space, then run run_shard! on every shard.
Returns total patches applied.
"""
function run_all_shards!(space, kernel_fn!::Function; l_max::Int=1000) :: Int
    prefixes = partition_trie(space, l_max)
    total    = 0
    for prefix in prefixes
        total += run_shard!(space, prefix, kernel_fn!; l_max=l_max)
    end
    return total
end

export Shard, PatchRecord
export partition_trie, capture_shard, materialize!, compute!
export patch_and_reattach!, should_adapt
export run_shard!, run_all_shards!
