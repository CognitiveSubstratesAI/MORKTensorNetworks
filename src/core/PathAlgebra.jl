"""
    PathAlgebra.jl — Relational path algebra on sparse matrices

MORK-Tensor-Networks paper §1: core operations on PathMap (trie-as-matrix).
All operations parameterized by semiring from Semirings.jl.
Uses CSR format; GPU-accelerated via SemiringKernels.jl.

Operations:

  - path_compose:     R ◦ S (relation composition via semiring matmul)
  - path_union:       R ∪ S (elementwise ⊕)
  - path_intersect:   R ∩ S (elementwise ⊗)
  - path_restrict:    R|_L  (mask by label set)
  - path_project:     ∃y R[x,y] (row-wise reduction + threshold)
  - path_transpose:   R^T
"""
module PathAlgebra

using ..Semirings
using ..SemiringKernels: gpu_semiring_spmv, gpu_semiring_spmm, gpu_elementwise_mask,
    gpu_threshold, gpu_semiring_reduce
using KernelAbstractions: CPU

export path_compose,
    path_union,
    path_intersect,
    path_restrict,
    path_project,
    path_transpose,
    path_reachability,
    path_viterbi,
    path_count,
    path_universal

# ─── CSR Helpers ─────────────────────────────────────────────────────────────

"""
Dense matrix → CSR conversion.
"""
function dense_to_csr(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    rowptr = Int32[1]
    colval = Int32[]
    nzval = T[]

    for i in 1:m
        for j in 1:n
            if !iszero(A[i, j]) && isfinite(A[i, j])
                push!(colval, Int32(j))
                push!(nzval, A[i, j])
            end
        end
        push!(rowptr, Int32(length(colval) + 1))
    end
    return rowptr, colval, nzval, m, n
end

"""
CSR → dense matrix conversion.
"""
function csr_to_dense(rowptr, colval, nzval, m, n, ::Type{T}=Float64) where {T}
    A = zeros(T, m, n)
    for i in 1:m
        for idx in rowptr[i]:(rowptr[i + 1] - 1)
            j = colval[idx]
            A[i, j] = nzval[idx]
        end
    end
    return A
end

# ─── Path Composition: R ◦ S ─────────────────────────────────────────────────
# T[x,z] = ⊕_y (R[x,y] ⊗ S[y,z])
# This is semiring matrix multiply.

"""
    path_compose(sr, R, S; apply_threshold::Bool=true, backend=nothing) → C

Compose relations R and S per spec §3 table:
T[x,z] = H(⊕_y R[x,y] ⊗ S[y,z])

Where H is the semiring-aware Heaviside step: H(x) = sone(sr) if x ≠ szero(sr),
else szero(sr). Default applies H per spec; pass `apply_threshold=false` for
raw semiring matmul (use case: path counting under SumProduct, Viterbi scoring
under MaxPlus — both want the raw weighted sum, not the Heaviside projection).

When `backend !== nothing`, lowers the matmul through `gpu_semiring_spmm`
(the spec §5.2 SpGEMM kernel) which goes via CSR + KernelAbstractions.
Pass `backend=CPU()` for the GPU-shaped path on host, or e.g.
`backend=CUDABackend()` for real GPU dispatch. When `backend===nothing` (default),
uses the dense CPU `semiring_matmul` reference.

Previously the H wrapper was silently dropped, so `path_compose` under
SumProduct returned a path-count matrix instead of a {0,1} reachability matrix.
And the GPU path was unreachable because `gpu_semiring_spmm` did not exist.
"""
function path_compose(
    sr::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix; apply_threshold::Bool=true,
    backend=nothing
)
    raw = if backend === nothing
        semiring_matmul(sr, R, S)
    else
        # F2 fix (audit 2026-06-04): the dense path asserts inner dims via semiring_matmul;
        # the GPU branch did not, so cols(R) > rows(S) caused an OOB read of rowptr_S in
        # the SpGEMM kernel. Assert here to match the dense path.
        @assert size(R, 2) == size(S, 1) "path_compose: inner dims must match — R is $(size(R)), S is $(size(S))"
        # NOTE (F1/G1): dense_to_csr drops entries via `!iszero && isfinite`, i.e. it
        # treats numeric 0.0 as structural absence. That is correct for SumProduct/Boolean
        # (szero=0) but WRONG for MaxPlus/PLN where 0.0 is a valid present edge (sone) and
        # szero is ±Inf. The GPU compose path is therefore sound for SumProduct/Boolean
        # only; MaxPlus/PLN GPU compose is not yet faithful (use the dense path). See TODO.
        # Convert dense → CSR, run SpGEMM, return dense output matrix.
        rowptr_R, colval_R, nzval_R, _, _ = dense_to_csr(R)
        rowptr_S, colval_S, nzval_S, _, n = dense_to_csr(S)
        gpu_semiring_spmm(
            sr,
            rowptr_R,
            colval_R,
            nzval_R,
            rowptr_S,
            colval_S,
            nzval_S,
            n;
            backend=backend
        )
    end
    apply_threshold ? _heaviside(sr, raw) : raw
end

"""
    _heaviside(sr, x) → element or matrix

Semiring-aware Heaviside step. For Boolean sr this is identity (already in
{false, true}). For SumProduct (szero=0, sone=1) this is the classical
H(x) = 1 if x > 0. For MaxPlus (szero=-Inf, sone=0) this is "present" iff
x ≠ -Inf, which correctly captures "a path exists" (cost 0 is a valid path).
"""
@inline _heaviside(sr::AbstractSemiring, x) = isequal(x, szero(sr)) ? szero(sr) : sone(sr)
_heaviside(sr::AbstractSemiring, M::AbstractMatrix) = map(x -> _heaviside(sr, x), M)

# ─── Path Union: R ∪ S ──────────────────────────────────────────────────────
# U[i,j] = R[i,j] ⊕ S[i,j]

"""
    path_union(sr, R, S; apply_threshold::Bool=true) → U

Elementwise union per spec §3 table: U = H(R ⊕ S).

Default applies H; pass `apply_threshold=false` for raw semiring ⊕
(use case: keep weighted union under SumProduct or MaxPlus).
Previously H was silently dropped — under SumProduct, `union(R, R)` returned
`2*R` instead of `R`.
"""
function path_union(
    sr::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix; apply_threshold::Bool=true
)
    @assert size(R) == size(S)
    U = similar(R)
    for i in eachindex(R)
        U[i] = oplus(sr, R[i], S[i])
    end
    apply_threshold ? _heaviside(sr, U) : U
end

# ─── Path Intersection: R ∩ S ───────────────────────────────────────────────
# I[i,j] = R[i,j] ⊗ S[i,j]

"""
    path_intersect(sr, R, S) → I

Elementwise intersection per spec §3 row 5: `I = R ∧ S` — **elementwise min**
(logical AND). The semiring parameter is retained for API uniformity but is
unused; min is the spec-defined operator regardless of semiring.

Previously used `otimes(sr, R[i], S[i])` which is wrong: under SumProduct,
`R ∩ R = R.^2` instead of `R`; under MaxPlus, `R ∩ S = R + S` (semiring sum)
instead of `min(R, S)`. Audit caught this because `S ∩ S` on a 0/1 matrix
happens to satisfy `1*1=1`, so the existing single test couldn't detect it.
"""
function path_intersect(::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix)
    @assert size(R) == size(S)
    return min.(R, S)
end

# ─── Path Restriction: R|_mask ──────────────────────────────────────────────
# R'[i,j] = R[i,j] * mask[i,j]

"""
    path_restrict(sr, R, mask) → R'
    path_restrict(R, mask) → R'    # 2-arg form: assumes SumProduct semantics

Label restriction per spec §3 row 3: `R' = R ⊙ M_L` — mask R by a 0/1 label
indicator. Where mask is "absent" (iszero), the entry collapses to the
semiring's additive zero; where present, R passes through unchanged.

Previously the 2-arg form did `R .* mask` which silently broke under MaxPlus
(sone=0, szero=-Inf): `R .* 0` collapses to 0 but the semiring-correct
"absent" value is -Inf. The 3-arg form is semiring-aware.
"""
function path_restrict(sr::AbstractSemiring, R::AbstractMatrix, mask::AbstractMatrix)
    @assert size(R) == size(mask)
    z = szero(sr)
    out = similar(R)
    for i in eachindex(R)
        out[i] = iszero(mask[i]) ? z : R[i]
    end
    return out
end

# 2-arg shim — defaults to SumProduct semantics for backward compat with the
# previous `R .* mask` behavior on numeric matrices.
path_restrict(R::AbstractMatrix, mask::AbstractMatrix) =
    path_restrict(SumProductSemiring(), R, mask)

# ─── Path Projection: ∃y R[x,y] ─────────────────────────────────────────────
# E[x] = H(⊕_y R[x,y]) where H is the Heaviside threshold

"""
    path_project(sr, R; thresh=0) → E

Existential projection: for each row x, reduce over columns with ⊕, then apply
the semiring-aware Heaviside H: E[x] = sone(sr) if acc ≠ szero(sr), else szero(sr).
Spec §3: H(x) = sone iff x ≠ additive-zero — NOT a numeric `> 0` test.

The `thresh` keyword is kept for callers that want a numeric cutoff under
SumProduct/PLN (where szero=0 and the semiring-aware test suffices anyway);
it is IGNORED by the semiring-aware branch. Pass `thresh` only when you have
a specific numeric reason to override the identity-based threshold.

BUGS FIXED (audit 2026-06-04):
  (1) `acc > thresh` with hardcoded `thresh=0` was wrong for MinPlus (szero=+Inf →
      Inf>0=true, falsely reported a path) and MaxPlus (sone=0, so a 0-weight path
      gave 0>0=false, falsely reported no path).
  (2) `one(eltype(R))` / `zero(eltype(R))` were returned instead of `sone(sr)` /
      `szero(sr)` — wrong under MaxPlus (sone=0, not 1) and MinPlus (szero=Inf).
"""
function path_project(sr::AbstractSemiring, R::AbstractMatrix; thresh=0)
    m, n = size(R)
    T = typeof(sone(sr))
    E = Vector{T}(undef, m)
    for i in 1:m
        acc = szero(sr)
        for j in 1:n
            acc = oplus(sr, acc, R[i, j])
        end
        # Semiring-aware Heaviside: path exists iff accumulator ≠ additive zero.
        # _heaviside is used by path_compose/path_union — same logic, consistent.
        E[i] = _heaviside(sr, acc)
    end
    return E
end

# ─── Path Transpose ─────────────────────────────────────────────────────────

"""
    path_transpose(R) → R^T
"""
path_transpose(R::AbstractMatrix) = permutedims(R)

# ─── Derived Operations (Paper §3 Table) ─────────────────────────────────────

"""
    path_reachability(R, k) → Reach_k

k-hop Boolean reachability: H(R^k) using Boolean semiring.
"""
function path_reachability(R::AbstractMatrix, k::Int)
    sr = BooleanSemiring()
    B = map(x -> !iszero(x), R)  # Convert to Bool
    result = B
    for _ in 2:k
        result = semiring_matmul(sr, result, B)
    end
    return result
end

"""
    path_viterbi(W) → Score

Best-path scoring via (max,+) semiring: Score[u,v] = max_y(W[u,y] + W[y,v])
"""
function path_viterbi(W::AbstractMatrix)
    sr = MaxPlusSemiring()
    return semiring_matmul(sr, W, W)
end

"""
    path_count(A, k) → Count

Count k-hop paths using (+,*) semiring: Count = A^k
"""
function path_count(A::AbstractMatrix, k::Int)
    sr = SumProductSemiring()
    result = copy(A)
    for _ in 2:k
        result = semiring_matmul(sr, result, A)
    end
    return result
end

"""
    path_universal(sr, R; thresh=0) → U

Universal quantification: U[x] = sone - H(⊕_y (sone - R[x,y]))
"Do ALL y satisfy the condition?"

Spec §3 defines this in the Boolean/fuzzy reading: complement = 1 - R[x,y].
This operation is only semantically defined for semirings where weights live in
[0,1] (Boolean, SumProduct/PLN). Under tropical semirings (MaxPlus, MinPlus, Cost)
the complement `sone(sr) - R[x,y]` is meaningless; callers should use those
semirings only for existential (`path_project`) operations.

FIX (audit 2026-06-04): now uses `sone(sr)` instead of hardcoded `one(eltype(R))`
so that the output identity is semiring-correct (Boolean: true=1, SumProduct: 1.0).
path_project is now semiring-aware, which fixes the secondary bug in the projection.
"""
function path_universal(sr::AbstractSemiring, R::AbstractMatrix; thresh=0)
    # C3 fix (audit 2026-06-04): guard against tropical semirings where complement
    # `sone - R[x,y]` is meaningless (MaxPlus sone=0, MinPlus/Cost sone=0 — subtracting
    # weights in (-∞,+∞] produces nonsense). Previously ran silently and returned garbage.
    @assert sr isa Union{BooleanSemiring, SumProductSemiring, PLNSemiring} """
path_universal is only defined for fuzzy/Boolean semirings where weights ∈ [0,1]
and complement = sone - R[x,y] is meaningful.
Got: $(typeof(sr)). For tropical semirings (MaxPlus, MinPlus, Cost) use path_project
for existential quantification instead."""
    id = sone(sr)
    complement = id .- R
    proj = path_project(sr, complement; thresh=thresh)
    return id .- proj
end

end # module
