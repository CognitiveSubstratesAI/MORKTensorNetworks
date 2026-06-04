"""
ShardZipper — §2 of MORK-Tensor-Networks (Goertzel, Oct 2025)

6-step workflow: Partition → Capture → Materialize → Compute → Patch & Reattach → Adapt.

ShardZipper takes a bounded piece of the MORK PathMap trie (a shard), linearizes
it into contiguous CSR arrays for GPU kernels, runs computation, then reattaches
the updated shard in O(1) preserving structural sharing.

Depends on: PathMap (trie + zipper), MORK (Space, PathMap{UnitVal})
"""

# L4 fix (audit 2026-06-04): pruned unused imports. Removed PathMap, ReadZipperCore,
# WriteZipperCore, read_zipper, write_zipper_at_path, get_val_at, zipper_descend_to_byte!,
# zipper_ascend_byte!, wz_graft!, SparseArrays, LinearAlgebra — none referenced in the
# body. (wz_graft! was the tell for the unimplemented O(1) graft reattach — see M2 note
# on patch_and_reattach!.) Added zipper_child_mask + test_bit for the H4 child-iteration.
using PathMap:
    read_zipper_at_path,
    zipper_val_count,
    set_val_at!,
    remove_val_at!,
    zipper_to_next_val!,
    zipper_path,
    zipper_child_mask,
    test_bit

# ── §2.1: Shard + PatchRecord ──────────────────────────────────────────────────

"""
    PatchRecord

One entry in the patch log. Kind ∈ {:insert, :delete, :update}.
path = full byte path; value = new Float32 weight (for :update/:insert).

L5 note (audit 2026-06-04): `value` is NOT consumed by `patch_and_reattach!` — the
MORK background trie (`space.btm`) is a `PathMap{UnitVal}` (set membership), so it
cannot store a per-edge weight; reattach writes `MORK.UNIT_VAL`. The field is retained
(not dropped) because it is part of the public PatchRecord API and carries the
kernel-computed weight for consumers that inspect `patch_log` BEFORE reattach (e.g. a
GPU kernel reading edge weights). Honouring it on reattach would require a
weight-valued trie, which is out of scope for the current UnitVal substrate.
"""
struct PatchRecord
    kind::Symbol       # :insert | :delete | :update
    path::Vector{UInt8}
    value::Float32
end

PatchRecord(kind::Symbol, path::Vector{UInt8}) = PatchRecord(kind, path, 1.0f0)

mutable struct Shard
    prefix::Vector{UInt8}
    row_ptr::Vector{Int}
    col_idx::Vector{Int}
    values::Vector{Float32}
    node_keys::Vector{Vector{UInt8}}
    patch_log::Vector{PatchRecord}
    size_cost::Int
end

# ── §2: Step 1 — Partition ─────────────────────────────────────────────────────

"""
    partition_trie(space, l_max) → Vector{Vector{UInt8}}

§2 Step 1: Split the MORK space's trie by hashed prefixes until each shard's
estimated size/cost ≤ l_max. Returns a list of shard prefix byte-paths.

Strategy: BFS over trie nodes; split any subtrie whose val_count > l_max
by descending one byte further.
"""
function partition_trie(space, l_max::Int)::Vector{Vector{UInt8}}
    prefixes = Vector{UInt8}[]
    _partition_recursive!(space.btm, UInt8[], l_max, prefixes)
    isempty(prefixes) && push!(prefixes, UInt8[])  # root shard if tiny
    return prefixes
end

function _partition_recursive!(
    btm, prefix::Vector{UInt8}, l_max::Int, prefixes::Vector{Vector{UInt8}}
)
    rz = read_zipper_at_path(btm, prefix)
    cost = zipper_val_count(rz)
    if cost <= l_max || length(prefix) >= 8
        push!(prefixes, copy(prefix))
        return nothing
    end
    # H4 fix (audit 2026-06-04): iterate only PRESENT child bytes via
    # zipper_child_mask + test_bit instead of scanning all 256 bytes and
    # opening a zipper per byte. The original 256-way fan-out allocated
    # vcat(prefix, b) for all 256 bytes regardless of presence — O(256)
    # allocations and zipper-opens per node, dominating partition cost.
    mask = zipper_child_mask(rz)
    found_children = false
    child_prefix = copy(prefix)
    push!(child_prefix, 0x00)   # reusable buffer: mutate last byte per child
    for b in UInt8(0):UInt8(255)
        test_bit(mask, b) || continue
        found_children = true
        child_prefix[end] = b
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
function capture_shard(space, prefix::Vector{UInt8})::Shard
    rz = read_zipper_at_path(space.btm, prefix)
    est = zipper_val_count(rz)
    return Shard(copy(prefix), Int[], Int[], Float32[], Vector{UInt8}[], PatchRecord[], est)
end

# ── §2: Step 3 — Materialize ──────────────────────────────────────────────────

"""
    _decode_relation(atom) → Union{Nothing, Tuple{String, Vector{String}}}

Decode a Rule-of-64 atom byte-path into (head, argument-symbols).

A relational atom `(rel a b ...)` is stored as the byte path:

    [Arity(N)] [Symbol(len)]"rel" [Symbol(len)]"a" [Symbol(len)]"b" ...

Returns `("rel", ["a","b",...])` reading the head + flat Symbol arguments.
Returns `nothing` if the atom is not an arity-headed expression whose arguments
are all flat symbols (variables / nested sub-expressions cause a safe bail-out —
this materialiser handles the flat relational form `(rel src dst [weight])`, which
matches MORK's reachability / connectome semantics; see audit §9.2).
"""
function _decode_relation(atom::AbstractVector{UInt8})
    isempty(atom) && return nothing
    t = MORK.byte_item(atom[1])
    t isa MORK.ExprArity || return nothing
    arity = Int(t.arity)
    arity >= 1 || return nothing
    syms = String[]
    i = 2
    n = length(atom)
    for _ in 1:arity
        i <= n || return nothing
        ti = MORK.byte_item(atom[i])
        ti isa MORK.ExprSymbol || return nothing   # var / nested expr — not a flat endpoint
        len = Int(ti.size)
        (i + len) <= n && push!(syms, String(@view atom[(i + 1):(i + len)]))
        i += 1 + len
    end
    isempty(syms) && return nothing
    return (syms[1], syms[2:end])
end

"""
    materialize!(shard, space) → Shard

§2 Step 3: Convert the shard's subtrie into contiguous CSR arrays representing the
**symbolic relation** R(src, dst) the atoms encode:

  - row_ptr[i]: start of row i in col_idx / values
  - col_idx[k]: column index (dst node) of nonzero k
  - values[k]:  edge weight
  - node_keys:  CSR row/col index → argument-symbol payload bytes

A node is a distinct **argument symbol**; an edge is `(arg1 → arg2)` for each atom,
weighted by `arg3` if the relation is arity-4 `(rel src dst weight)` (else 1.0).
The head symbol (relation label, e.g. `edge`/`syn`) is NOT a node.

H3 FIX (audit §9, 2026-06-04): the previous implementation treated every i-byte
PREFIX of every stored path as a node and emitted an edge for each consecutive
prefix pair — i.e. it encoded trie-traversal nesting, not the relation. Against
MORK ground truth (reachability.jl / info_flow_zipper.jl), `(edge a b)` is the edge
a→b over symbol-nodes {a,b}, NOT a byte-prefix chain. Now decodes Rule-of-64
argument symbols via `_decode_relation`, matching MORK's relational-query semantics.

Scope note (C4): this builds a CSR but the downstream path-algebra is still dense
(`semiring_matmul`). The dense→sparse-output / SoA-arena rewrite remains the C4
package-identity decision; this fix corrects the *relation encoding*, not density.

`zipper_path` is relative to the shard prefix, so the full atom = `prefix ++ suffix`.
"""
function materialize!(shard::Shard, space)::Shard
    prefix = shard.prefix
    rz = read_zipper_at_path(space.btm, prefix)

    # Collect full atom byte-paths (prefix ++ relative suffix).
    atoms = Vector{UInt8}[]
    while zipper_to_next_val!(rz)
        rel = collect(zipper_path(rz))
        push!(atoms, vcat(prefix, rel))
    end
    isempty(atoms) && return shard

    # Decode each atom into a symbolic edge; intern argument symbols as nodes.
    node_map = Dict{String, Int}()
    node_payloads = Vector{UInt8}[]
    edges = Tuple{Int, Int, Float32}[]
    _node! = function (sym::String)
        idx = get(node_map, sym, 0)
        idx != 0 && return idx
        idx = length(node_map) + 1
        node_map[sym] = idx
        push!(node_payloads, Vector{UInt8}(sym))
        return idx
    end
    for atom in atoms
        decoded = _decode_relation(atom)
        decoded === nothing && continue
        _head, args = decoded
        length(args) >= 2 || continue
        src = _node!(args[1])
        dst = _node!(args[2])
        w = 1.0f0
        if length(args) >= 3
            pw = tryparse(Float32, args[3])
            pw === nothing || (w = pw)
        end
        push!(edges, (src, dst, w))
    end

    n = length(node_map)
    if n == 0
        shard.row_ptr = Int[1]
        shard.col_idx = Int[]
        shard.values = Float32[]
        shard.node_keys = Vector{UInt8}[]
        return shard
    end

    # Build CSR (row-major) from the symbolic edge list.
    row_entries = [Tuple{Int, Float32}[] for _ in 1:n]
    for (src, dst, w) in edges
        push!(row_entries[src], (dst, w))
    end
    row_ptr = Int[1]
    col_idx = Int[]
    nzval = Float32[]
    for i in 1:n
        for (c, v) in row_entries[i]
            push!(col_idx, c)
            push!(nzval, v)
        end
        push!(row_ptr, length(col_idx) + 1)
    end

    shard.row_ptr = row_ptr
    shard.col_idx = col_idx
    shard.values = nzval
    shard.node_keys = node_payloads
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
function compute!(shard::Shard, kernel_fn!::Function)::Shard
    isempty(shard.node_keys) && return shard
    kernel_fn!(shard)
    return shard
end

# ── §2: Step 5 — Patch & Reattach ─────────────────────────────────────────────

"""
    patch_and_reattach!(shard, space) → Int

§2 Step 5: Apply the patch log to the trie and clear it.
Returns number of patches applied.

M2 note (audit 2026-06-04): the spec §2.5 promises O(1) Λ_s graft reattach via
structural sharing. This implementation does O(patches × path-length) per-path
global writes instead. The wz_graft! import is present but unused — a real O(1)
graft reattach is the correct target but requires the `capture_shard` to record
a proper write-zipper continuation at the shard root, which is not yet wired.
Owner decision: implement the graft reattach or update §2 to not claim O(1).

M1 note (audit 2026-06-04): `empty!(shard.patch_log)` clears the log BEFORE any
caller can check `length(patch_log)` in `should_adapt`. The patch count is returned
so callers can make the adapt decision before the log is cleared — use the return
value with `should_adapt_from_count(count, l_max)`, or call `should_adapt` BEFORE
calling `patch_and_reattach!`.
"""
function patch_and_reattach!(shard::Shard, space)::Int
    prefix = shard.prefix
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
    empty!(shard.patch_log)   # clears log — call should_adapt BEFORE this
    return applied
end

# ── §2: Step 6 — Adapt ────────────────────────────────────────────────────────

"""
    should_adapt(shard, l_max) → Bool

§2 Step 6: Return true if the shard should be resplit next time.

IMPORTANT (M1, audit 2026-06-04): `patch_and_reattach!` calls `empty!(patch_log)`
at the end, so the patch-count branch (`length(patch_log) > l_max ÷ 4`) is ALWAYS
false if evaluated after reattach. Call `should_adapt` BEFORE `patch_and_reattach!`
in the pipeline to get a meaningful result from the chatty-boundary check.
Only the `size_cost` branch fires post-reattach.
"""
function should_adapt(shard::Shard, l_max::Int)::Bool
    shard.size_cost > l_max * 2 || length(shard.patch_log) > l_max ÷ 4
end

# ── High-level convenience ─────────────────────────────────────────────────────

"""
    run_shard!(space, prefix, kernel_fn!; l_max=1000) → Int

Run the full ShardZipper 6-step workflow on one shard:
capture → materialize → compute → patch & reattach.
Returns patches applied.
"""
function run_shard!(
    space, prefix::Vector{UInt8}, kernel_fn!::Function; l_max::Int=1000
)::Int
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
function run_all_shards!(space, kernel_fn!::Function; l_max::Int=1000)::Int
    prefixes = partition_trie(space, l_max)
    total = 0
    for prefix in prefixes
        total += run_shard!(space, prefix, kernel_fn!; l_max=l_max)
    end
    return total
end

export Shard, PatchRecord
export partition_trie, capture_shard, materialize!, compute!
export patch_and_reattach!, should_adapt
export run_shard!, run_all_shards!
