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
using ..SemiringKernels: gpu_semiring_spmv, gpu_semiring_spmm,
                         gpu_elementwise_mask, gpu_threshold, gpu_semiring_reduce
using KernelAbstractions: CPU

export path_compose, path_union, path_intersect,
       path_restrict, path_project, path_transpose,
       path_reachability, path_viterbi, path_count,
       path_universal

# ─── CSR Helpers ─────────────────────────────────────────────────────────────

"""Dense matrix → CSR conversion."""
function dense_to_csr(A::AbstractMatrix{T}) where T
    m, n = size(A)
    rowptr = Int32[1]
    colval = Int32[]
    nzval = T[]

    for i in 1:m
        for j in 1:n
            if !iszero(A[i,j]) && isfinite(A[i,j])
                push!(colval, Int32(j))
                push!(nzval, A[i,j])
            end
        end
        push!(rowptr, Int32(length(colval) + 1))
    end
    return rowptr, colval, nzval, m, n
end

"""CSR → dense matrix conversion."""
function csr_to_dense(rowptr, colval, nzval, m, n, ::Type{T}=Float64) where T
    A = zeros(T, m, n)
    for i in 1:m
        for idx in rowptr[i]:rowptr[i+1]-1
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
function path_compose(sr::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix;
                       apply_threshold::Bool=true,
                       backend=nothing)
    raw = if backend === nothing
        semiring_matmul(sr, R, S)
    else
        # Convert dense → CSR, run SpGEMM, return dense output matrix.
        rowptr_R, colval_R, nzval_R, _, _ = dense_to_csr(R)
        rowptr_S, colval_S, nzval_S, _, n = dense_to_csr(S)
        gpu_semiring_spmm(sr, rowptr_R, colval_R, nzval_R,
                              rowptr_S, colval_S, nzval_S, n;
                          backend=backend)
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
function path_union(sr::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix;
                     apply_threshold::Bool=true)
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

Existential projection: for each row x, reduce over columns with ⊕,
then apply threshold. Returns a vector.
"""
function path_project(sr::AbstractSemiring, R::AbstractMatrix; thresh=0)
    m, n = size(R)
    E = Vector{eltype(R)}(undef, m)
    for i in 1:m
        acc = szero(sr)
        for j in 1:n
            acc = oplus(sr, acc, R[i,j])
        end
        E[i] = acc > thresh ? one(eltype(R)) : zero(eltype(R))
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

Universal quantification: U[x] = 1 - H(⊕_y (1 - R[x,y]))
"Do ALL y satisfy the condition?"
"""
function path_universal(sr::AbstractSemiring, R::AbstractMatrix; thresh=0)
    m, n = size(R)
    complement = one(eltype(R)) .- R
    proj = path_project(sr, complement; thresh=thresh)
    return one(eltype(R)) .- proj
end

end # module
