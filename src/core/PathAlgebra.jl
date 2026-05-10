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
using ..SemiringKernels: gpu_semiring_spmv, gpu_elementwise_mask, gpu_threshold, gpu_semiring_reduce

export path_compose, path_union, path_intersect,
       path_restrict, path_project, path_transpose,
       path_reachability, path_viterbi, path_count

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
    path_compose(sr, R, S) → C

Compose relations R and S using semiring matmul.
R and S are dense matrices; returns dense result.
For sparse: use semiring_matmul from Semirings.jl directly.
"""
function path_compose(sr::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix)
    return semiring_matmul(sr, R, S)
end

# ─── Path Union: R ∪ S ──────────────────────────────────────────────────────
# U[i,j] = R[i,j] ⊕ S[i,j]

"""
    path_union(sr, R, S) → U

Elementwise ⊕ of two relation matrices.
"""
function path_union(sr::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix)
    @assert size(R) == size(S)
    U = similar(R)
    for i in eachindex(R)
        U[i] = oplus(sr, R[i], S[i])
    end
    return U
end

# ─── Path Intersection: R ∩ S ───────────────────────────────────────────────
# I[i,j] = R[i,j] ⊗ S[i,j]

"""
    path_intersect(sr, R, S) → I

Elementwise ⊗ of two relation matrices.
"""
function path_intersect(sr::AbstractSemiring, R::AbstractMatrix, S::AbstractMatrix)
    @assert size(R) == size(S)
    I = similar(R)
    for i in eachindex(R)
        I[i] = otimes(sr, R[i], S[i])
    end
    return I
end

# ─── Path Restriction: R|_mask ──────────────────────────────────────────────
# R'[i,j] = R[i,j] * mask[i,j]

"""
    path_restrict(R, mask) → R'

Elementwise mask (Hadamard product). Mask is 0/1 matrix.
"""
function path_restrict(R::AbstractMatrix, mask::AbstractMatrix)
    @assert size(R) == size(mask)
    return R .* mask
end

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
