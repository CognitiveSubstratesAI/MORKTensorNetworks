"""
    TuckerDecomposition.jl — Low-rank tensor densification for MORK

MORK-Tensor-Networks paper §4: Tucker decomposition converts sparse
relations into low-rank dense form for GPU-efficient dense einsums.

    A[i,j,k] ≈ Σ_{p,q,r} C[p,q,r] * M[i,p] * N[j,q] * P[k,r]

Where:
  - C: core tensor (small, dense, mixes embedding coordinates)
  - M, N, P: factor matrices (learned embeddings)

For 2D relations (most common in MORK):
    A[i,j] ≈ Σ_p Σ_q C[p,q] * M[i,p] * N[j,q]
            = M * C * N^T

This module provides:
  - tucker_decompose_2d: ALS-based decomposition of a 2D matrix
  - tucker_reconstruct_2d: reconstruct from factors
  - should_densify: heuristic for when Tucker beats sparse
"""
module TuckerDecomposition

using LinearAlgebra

export tucker_decompose_2d, tucker_reconstruct_2d, should_densify

"""
    tucker_decompose_2d(A, rank; max_iter=50, tol=1e-6) → (C, M, N, error)

Decompose m×n matrix A into core C (rank×rank) and factor matrices
M (m×rank), N (n×rank) using Alternating Least Squares (ALS).

Returns (C, M, N, reconstruction_error).
"""
function tucker_decompose_2d(A::AbstractMatrix{T}, rank::Int;
                              max_iter::Int=50, tol::Float64=1e-6) where T
    m, n = size(A)
    rank = min(rank, m, n)  # can't exceed dimensions

    # Initialize M, N via truncated SVD
    F = svd(A)
    M = F.U[:, 1:rank]
    N = F.V[:, 1:rank]

    # ALS iterations
    prev_error = Inf
    C = zeros(T, rank, rank)

    for iter in 1:max_iter
        # Fix M, solve for C and N
        # A ≈ M * C * N^T
        # A^T * M ≈ N * C^T  →  N = (A^T * M) * inv(C^T) ... but ALS is simpler:

        # Step 1: Fix N, update M and C
        # A * N ≈ M * C  →  M * C = A * N
        AN = A * N  # m × rank
        # Use QR to get orthogonal M
        qr_AN = qr(AN)
        M = Matrix(qr_AN.Q)[:, 1:rank]
        C = Matrix(qr_AN.R)[1:rank, 1:rank]

        # Step 2: Fix M, update N
        # A^T * M ≈ N * C^T
        ATM = A' * M  # n × rank
        qr_ATM = qr(ATM)
        N = Matrix(qr_ATM.Q)[:, 1:rank]
        # Update core
        C = M' * A * N  # rank × rank

        # Check convergence
        A_approx = M * C * N'
        err = norm(A - A_approx) / max(norm(A), eps(T))
        if abs(prev_error - err) < tol
            return (C, M, N, err)
        end
        prev_error = err
    end

    return (C, M, N, prev_error)
end

"""
    tucker_reconstruct_2d(C, M, N) → A_approx

Reconstruct matrix from Tucker factors: A ≈ M * C * N^T
"""
function tucker_reconstruct_2d(C::AbstractMatrix, M::AbstractMatrix, N::AbstractMatrix)
    return M * C * N'
end

"""
    should_densify(A; fill_threshold=0.3, rank_ratio=0.5) → Bool

Heuristic for when Tucker densification beats sparse operations.
Returns true if:
  - Fill ratio is between 5% and fill_threshold (too sparse = keep sparse,
    too dense = already dense)
  - Effective rank is low (rank_ratio of min dimension)
"""
function should_densify(A::AbstractMatrix; fill_threshold::Float64=0.3, rank_ratio::Float64=0.5)
    m, n = size(A)
    nnz = count(!iszero, A)
    fill = nnz / (m * n)

    # Too sparse or already dense — don't bother
    if fill < 0.05 || fill > fill_threshold
        return false
    end

    # Check effective rank via SVD
    sv = svdvals(A)
    total_energy = sum(sv .^ 2)
    if total_energy ≈ 0
        return false
    end

    # Find rank capturing 90% of energy
    cumulative = cumsum(sv .^ 2) ./ total_energy
    effective_rank = findfirst(>=(0.9), cumulative)
    if effective_rank === nothing
        effective_rank = length(sv)
    end

    # Densify if effective rank is small relative to dimensions
    return effective_rank <= rank_ratio * min(m, n)
end

end # module
