"""
    SemiringKernels.jl — GPU-accelerated semiring operations via KernelAbstractions.jl

MORK-Tensor-Networks paper §3: maps tensor logic to GPU kernels parameterized
by configurable semirings. Vendor-neutral: CUDA, ROCm, oneAPI, Metal.

Kernel inventory (paper Table 1):
  1. semiring_spmv!     — sparse matrix-vector (SpMV) with semiring
  2. semiring_reduce!   — ⊕-reduction over a vector
  3. elementwise_mask!  — Hadamard product for label restriction
  4. threshold!         — Heaviside H(x) for existential quantification
  5. semiring_spmm!     — sparse matrix-matrix (SpGEMM) with semiring

Uses CSR (Compressed Sparse Row) format for sparse matrices:
  rowptr::Vector{Int32}   — row pointer array (length m+1)
  colval::Vector{Int32}   — column indices (length nnz)
  nzval::Vector{T}        — nonzero values (length nnz)
"""
module SemiringKernels

using KernelAbstractions
using KernelAbstractions: @kernel, @index
using ..Semirings

export semiring_spmv_kernel!, semiring_spmm_kernel!,
       semiring_reduce_kernel!,
       elementwise_mask_kernel!, threshold_kernel!,
       gpu_semiring_spmv, gpu_semiring_spmm, gpu_semiring_reduce,
       gpu_elementwise_mask, gpu_threshold,
       semiring_tag

# ─── Kernel 1: Sparse Matrix-Vector (SpMV) ──────────────────────────────────
# y[i] = ⊕_j (A[i,j] ⊗ x[j])  for nonzero entries only
#
# CSR format: rowptr, colval, nzval
# One thread per row.

@kernel function semiring_spmv_kernel!(y, rowptr, colval, nzval, x,
                                        @Const(sr_zero), @Const(sr_type))
    i = @index(Global, Linear)
    row_start = rowptr[i]
    row_end = rowptr[i + 1] - 1

    acc = sr_zero
    for idx in row_start:row_end
        j = colval[idx]
        v = nzval[idx]
        # Inline semiring ops based on type tag
        # sr_type: 1=SumProduct, 2=MaxPlus, 3=MinPlus, 4=Boolean, 5=PLN, 6=Cost
        prod = _sr_otimes(sr_type, v, x[j])
        acc = _sr_oplus(sr_type, acc, prod)
    end
    @inbounds y[i] = acc
end

# Semiring dispatch via integer tag (GPU-friendly — no dynamic dispatch)
@inline function _sr_oplus(tag::Int32, a, b)
    if tag == Int32(1)      # SumProduct
        return a + b
    elseif tag == Int32(2)  # MaxPlus
        return max(a, b)
    elseif tag == Int32(3)  # MinPlus
        return min(a, b)
    elseif tag == Int32(4)  # Boolean
        return max(a, b)  # OR as max for numeric
    elseif tag == Int32(5)  # PLN
        return max(a, b)
    elseif tag == Int32(6)  # Cost
        return min(a, b)
    else
        return a + b  # default: sum
    end
end

@inline function _sr_otimes(tag::Int32, a, b)
    if tag == Int32(1)      # SumProduct
        return a * b
    elseif tag == Int32(2)  # MaxPlus
        return a + b
    elseif tag == Int32(3)  # MinPlus
        return a + b
    elseif tag == Int32(4)  # Boolean
        return min(a, b)  # AND as min for numeric
    elseif tag == Int32(5)  # PLN
        return a * b
    elseif tag == Int32(6)  # Cost
        return a + b
    else
        return a * b  # default: product
    end
end

# Map AbstractSemiring to integer tag
semiring_tag(::SumProductSemiring) = Int32(1)
semiring_tag(::MaxPlusSemiring)    = Int32(2)
semiring_tag(::MinPlusSemiring)    = Int32(3)
semiring_tag(::BooleanSemiring)    = Int32(4)
semiring_tag(::PLNSemiring)        = Int32(5)
semiring_tag(::CostSemiring)       = Int32(6)

# ─── Kernel 1b: Sparse Matrix-Matrix (SpGEMM) ───────────────────────────────
# C[i, j] = ⊕_k (A[i, k] ⊗ B[k, j])   for nonzero entries only
#
# Inputs:
#   A in CSR: rowptr_A (length m+1), colval_A, nzval_A
#   B in CSR: rowptr_B (length k+1), colval_B, nzval_B
#   Output: dense matrix C (m × n) — caller pre-allocates and fills with szero(sr)
#
# Algorithm: one thread per row of A. Each thread walks A[i, *] nonzeros,
# then for each such (k, a_ik), walks B[k, *] nonzeros, accumulating into
# the dense output row C[i, j]. Output is dense to avoid the classical
# SpGEMM symbolic phase (which is hard to GPU well). For C sparsity above
# ~10%, this is competitive; below that a sparse-output variant would win.
#
# This is the headline §5.2 kernel — the spec's "sparse einsums (SpGEMM-like)
# with semirings" promise — previously missing from this file (audit 2026-05-30).
# Used by PathAlgebra.path_compose to lower `T = H(R ⊗ S)` to GPU.

@kernel function semiring_spmm_kernel!(C,                 # dense output (m × n)
                                        rowptr_A, colval_A, nzval_A,
                                        rowptr_B, colval_B, nzval_B,
                                        @Const(n_cols),   # n: columns of B (= cols of C)
                                        @Const(sr_zero),
                                        @Const(sr_type))
    i = @index(Global, Linear)
    row_a_start = rowptr_A[i]
    row_a_end   = rowptr_A[i + 1] - 1
    for ka in row_a_start:row_a_end
        k_idx = colval_A[ka]
        a_ik  = nzval_A[ka]
        row_b_start = rowptr_B[k_idx]
        row_b_end   = rowptr_B[k_idx + 1] - 1
        for kb in row_b_start:row_b_end
            j     = colval_B[kb]
            b_kj  = nzval_B[kb]
            prod  = _sr_otimes(sr_type, a_ik, b_kj)
            old   = @inbounds C[i, j]
            @inbounds C[i, j] = _sr_oplus(sr_type, old, prod)
        end
    end
end

# ─── Kernel 2: Reduction ────────────────────────────────────────────────────
# result = v[1] ⊕ v[2] ⊕ ... ⊕ v[n]
# Simple per-block reduction; for production use tree reduction.

@kernel function semiring_reduce_kernel!(output, input, @Const(sr_zero), @Const(sr_type))
    # Single-thread reduction — correct for any vector size.
    # For vectors > 10k elements, a tree-based parallel reduction
    # (e.g., 2-pass block reduce) would be faster on GPU.
    i = @index(Global, Linear)
    if i == 1
        acc = sr_zero
        for j in 1:length(input)
            acc = _sr_oplus(sr_type, acc, input[j])
        end
        output[1] = acc
    end
end

# ─── Kernel 3: Elementwise Mask (Label Restriction) ─────────────────────────
# out[i] = val[i] * mask[i]  (Hadamard product)
# For Boolean: out[i] = val[i] if mask[i] != 0, else 0

@kernel function elementwise_mask_kernel!(out, val, mask)
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = val[i] * mask[i]
    end
end

# ─── Kernel 4: Threshold (Existential Quantification) ───────────────────────
# out[i] = 1 if input[i] > threshold, else 0
# H(Σ_y Φ[x,y]) — applied after reduction

@kernel function threshold_kernel!(out, input, @Const(thresh))
    i = @index(Global, Linear)
    if i <= length(out)
        @inbounds out[i] = input[i] > thresh ? one(eltype(out)) : zero(eltype(out))
    end
end

# ─── Host-Side Wrappers ─────────────────────────────────────────────────────
# Kernels above use KernelAbstractions @kernel — vendor-neutral.
# Default backend=CPU() for testing. Pass backend=CUDABackend() or
# MetalBackend() etc. for real GPU dispatch. No code change needed.

"""
    gpu_semiring_spmv(sr, rowptr, colval, nzval, x; backend=CPU())

Sparse matrix-vector multiply with semiring on GPU.
Returns result vector y.
"""
function gpu_semiring_spmv(sr::AbstractSemiring, rowptr, colval, nzval, x;
                           backend=KernelAbstractions.CPU())
    m = length(rowptr) - 1
    T = eltype(nzval)
    y = KernelAbstractions.zeros(backend, T, m)
    fill!(y, T(szero(sr)))

    tag = semiring_tag(sr)
    z = T(szero(sr))

    kernel = semiring_spmv_kernel!(backend, 256)
    kernel(y, rowptr, colval, nzval, x, z, tag; ndrange=m)
    KernelAbstractions.synchronize(backend)
    return y
end

"""
    gpu_semiring_spmm(sr, rowptr_A, colval_A, nzval_A,
                          rowptr_B, colval_B, nzval_B, n_cols;
                       backend=CPU()) → Matrix

Sparse-by-sparse matrix multiply (SpGEMM) under semiring `sr`. Both A and B
are in CSR form. Output is a dense (m × n_cols) matrix, with entries
initialized to `szero(sr)` and accumulated via `_sr_oplus`/`_sr_otimes`.

This is the spec §5.2 "sparse einsums (SpGEMM-like) parameterized by
semiring" kernel — required to lower `path_compose` to GPU per spec §3
row 1 formula `T[x,z] = H(Σ_y R[x,y] ⊗ S[y,z])`.

Backend-neutral via KernelAbstractions; pass `backend=CUDABackend()` etc.
for real GPU dispatch.
"""
function gpu_semiring_spmm(sr::AbstractSemiring,
                            rowptr_A, colval_A, nzval_A,
                            rowptr_B, colval_B, nzval_B,
                            n_cols::Integer;
                            backend=KernelAbstractions.CPU())
    m = length(rowptr_A) - 1
    T = eltype(nzval_A)
    C = KernelAbstractions.zeros(backend, T, m, n_cols)
    fill!(C, T(szero(sr)))

    tag = semiring_tag(sr)
    z   = T(szero(sr))

    kernel = semiring_spmm_kernel!(backend, 256)
    kernel(C, rowptr_A, colval_A, nzval_A,
              rowptr_B, colval_B, nzval_B,
              Int(n_cols), z, tag; ndrange=m)
    KernelAbstractions.synchronize(backend)
    return C
end

"""
    gpu_semiring_reduce(sr, v; backend=CPU())

⊕-reduction of vector v using semiring sr.
"""
function gpu_semiring_reduce(sr::AbstractSemiring, v;
                             backend=KernelAbstractions.CPU())
    T = eltype(v)
    output = KernelAbstractions.zeros(backend, T, 1)
    fill!(output, T(szero(sr)))

    tag = semiring_tag(sr)
    z = T(szero(sr))

    kernel = semiring_reduce_kernel!(backend, 1)
    kernel(output, v, z, tag; ndrange=1)
    KernelAbstractions.synchronize(backend)
    return output[1]
end

"""
    gpu_elementwise_mask(val, mask; backend=CPU())

Hadamard product: out[i] = val[i] * mask[i]
"""
function gpu_elementwise_mask(val, mask; backend=KernelAbstractions.CPU())
    n = length(val)
    T = eltype(val)
    out = KernelAbstractions.zeros(backend, T, n)

    kernel = elementwise_mask_kernel!(backend, 256)
    kernel(out, val, mask; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return out
end

"""
    gpu_threshold(v, thresh=0; backend=CPU())

Heaviside threshold: out[i] = v[i] > thresh ? 1 : 0
"""
function gpu_threshold(v, thresh=zero(eltype(v)); backend=KernelAbstractions.CPU())
    n = length(v)
    T = eltype(v)
    out = KernelAbstractions.zeros(backend, T, n)

    kernel = threshold_kernel!(backend, 256)
    kernel(out, v, thresh; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return out
end

end # module
