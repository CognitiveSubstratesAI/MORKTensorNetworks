"""
    GPULayout.jl — GPU data layout optimization for MORK tensor operations

MORK-Tensor-Networks paper §3, §4: efficient data layout decisions for
GPU dispatch. Routes operations to sparse or dense kernels based on
matrix properties.

Components:
  - CSR/BCSR conversion from dense matrices
  - Densification routing via Tucker decomposition
  - Quantifier-as-reduction mapping
  - Layout analysis and recommendation
"""
module GPULayout

using ..Semirings: AbstractSemiring, szero
using ..TuckerDecomposition: should_densify, tucker_decompose_2d, tucker_reconstruct_2d

export CSRMatrix, dense_to_csr, csr_to_dense,
       analyze_layout, recommend_strategy,
       LayoutStrategy, SparseCSR, DenseTucker, DenseDirect

# ─── CSR Matrix ──────────────────────────────────────────────────────────────

"""
    CSRMatrix{T}

Compressed Sparse Row format for GPU-friendly sparse operations.
"""
struct CSRMatrix{T}
    rowptr::Vector{Int32}
    colval::Vector{Int32}
    nzval::Vector{T}
    m::Int32  # rows
    n::Int32  # cols
end

"""Convert dense matrix to CSR."""
function dense_to_csr(A::AbstractMatrix{T}) where T
    m, n = size(A)
    rowptr = Int32[1]
    colval = Int32[]
    nzval = T[]

    z = zero(T)
    for i in 1:m
        for j in 1:n
            v = A[i, j]
            if v != z && isfinite(v)
                push!(colval, Int32(j))
                push!(nzval, v)
            end
        end
        push!(rowptr, Int32(length(colval) + 1))
    end
    return CSRMatrix{T}(rowptr, colval, nzval, Int32(m), Int32(n))
end

"""Convert CSR back to dense."""
function csr_to_dense(csr::CSRMatrix{T}) where T
    A = zeros(T, csr.m, csr.n)
    for i in 1:csr.m
        for idx in csr.rowptr[i]:csr.rowptr[i+1]-1
            A[i, csr.colval[idx]] = csr.nzval[idx]
        end
    end
    return A
end

"""Number of nonzeros."""
nnz(csr::CSRMatrix) = length(csr.nzval)

"""Fill ratio."""
fill_ratio(csr::CSRMatrix) = nnz(csr) / (Float64(csr.m) * csr.n)

# ─── Layout Strategy ─────────────────────────────────────────────────────────

abstract type LayoutStrategy end

"""Use sparse CSR format (SpMV/SpGEMM kernels)."""
struct SparseCSR <: LayoutStrategy end

"""Use Tucker-decomposed dense format (dense einsum kernels)."""
struct DenseTucker <: LayoutStrategy
    rank::Int
end

"""Use direct dense format (standard matmul)."""
struct DenseDirect <: LayoutStrategy end

# ─── Layout Analysis ─────────────────────────────────────────────────────────

"""
    analyze_layout(A) → (fill_ratio, effective_rank, recommended_strategy)

Analyze a matrix and recommend the best GPU layout strategy.
"""
function analyze_layout(A::AbstractMatrix)
    m, n = size(A)
    nnz_count = count(x -> x != 0 && isfinite(x), A)
    fill = nnz_count / (m * n)

    # Effective rank via SVD (if small enough)
    effective_rank = min(m, n)
    if m <= 512 && n <= 512
        try
            sv = svdvals(Float64.(A))
            total = sum(sv .^ 2)
            if total > 0
                cum = cumsum(sv .^ 2) ./ total
                r = findfirst(>=(0.9), cum)
                effective_rank = r === nothing ? length(sv) : r
            end
        catch
            # SVD failed — assume full rank
        end
    end

    strategy = recommend_strategy(fill, effective_rank, min(m, n))
    return (fill_ratio=fill, effective_rank=effective_rank, strategy=strategy)
end

"""
    recommend_strategy(fill, eff_rank, min_dim) → LayoutStrategy

Heuristic routing:
  - fill < 0.1: SparseCSR (sparse kernels efficient)
  - fill 0.1-0.5 and low rank: DenseTucker (compress then dense)
  - fill > 0.5: DenseDirect (already dense)
"""
function recommend_strategy(fill::Float64, eff_rank::Int, min_dim::Int)
    if fill < 0.1
        return SparseCSR()
    elseif fill <= 0.5 && eff_rank <= min_dim ÷ 2
        return DenseTucker(eff_rank)
    else
        return DenseDirect()
    end
end

# ─── Quantifier-as-Reduction Mapping ─────────────────────────────────────────

"""
    existential_reduce(v) → Bool

Existential quantification as GPU reduction: ∃x P(x) ≡ (sum(v) > 0).
Native GPU primitive: sum + threshold.
"""
existential_reduce(v::AbstractVector) = sum(v) > 0

"""
    universal_reduce(v) → Bool

Universal quantification as GPU reduction: ∀x P(x) ≡ (min(v) > 0).
Native GPU primitive: min + threshold.
"""
universal_reduce(v::AbstractVector) = minimum(v) > 0

end # module
