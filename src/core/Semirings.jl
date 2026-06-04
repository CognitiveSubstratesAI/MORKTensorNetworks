"""
    Semirings.jl — Configurable algebraic semirings for MORK tensor operations

Implements the semiring abstraction from MORK-Tensor-Networks paper §3.
All tensor logic operations (join, projection, restriction, quantification,
path scoring) are parameterized by a semiring (⊕, ⊗, 0̄, 1̄).

Semirings supported:

  - BooleanSemiring:    (∨, ∧, false, true)   — reachability
  - SumProductSemiring: (+, *, 0, 1)           — path counting
  - MaxPlusSemiring:    (max, +, -∞, 0)        — best-path (Viterbi)
  - MinPlusSemiring:    (min, +, +∞, 0)        — shortest path (tropical)
  - PLNSemiring:        (max, *, 0, 1)          — PLN truth values (Q_PLN)
  - CostSemiring:       (min, +, +∞, 0)         — Occam complexity (Q_cost)

Usage:
sr = MaxPlusSemiring()
oplus(sr, 3.0, 5.0)   # → 5.0 (max)
otimes(sr, 3.0, 5.0)  # → 8.0 (+)
zero(sr)               # → -Inf
one(sr)                # → 0.0

Designed for KernelAbstractions.jl GPU dispatch — all operations are
@inline and type-stable for scalar elements.
"""
module Semirings

export AbstractSemiring,
    BooleanSemiring,
    SumProductSemiring,
    MaxPlusSemiring,
    MinPlusSemiring,
    PLNSemiring,
    CostSemiring,
    oplus,
    otimes,
    szero,
    sone,
    semiring_matmul,
    semiring_matvec,
    semiring_reduce

# ─── Abstract Type ───────────────────────────────────────────────────────────

abstract type AbstractSemiring end

"""
Return the additive identity (⊕-identity).
"""
function szero end

"""
Return the multiplicative identity (⊗-identity).
"""
function sone end

"""
Additive operation ⊕.
"""
function oplus end

"""
Multiplicative operation ⊗.
"""
function otimes end

# ─── Boolean Semiring (∨, ∧, false, true) ───────────────────────────────────

struct BooleanSemiring <: AbstractSemiring end

@inline szero(::BooleanSemiring) = false
@inline sone(::BooleanSemiring) = true
@inline oplus(::BooleanSemiring, a::Bool, b::Bool) = a | b
@inline otimes(::BooleanSemiring, a::Bool, b::Bool) = a & b

# Coerce non-Bool inputs
@inline oplus(s::BooleanSemiring, a, b) = oplus(s, !iszero(a), !iszero(b))
@inline otimes(s::BooleanSemiring, a, b) = otimes(s, !iszero(a), !iszero(b))

# ─── Sum-Product Semiring (+, *, 0, 1) ──────────────────────────────────────

struct SumProductSemiring <: AbstractSemiring end

@inline szero(::SumProductSemiring) = 0.0
@inline sone(::SumProductSemiring) = 1.0
@inline oplus(::SumProductSemiring, a, b) = a + b
@inline otimes(::SumProductSemiring, a, b) = a * b

# ─── Max-Plus Semiring (max, +, -∞, 0) — Viterbi ────────────────────────────

struct MaxPlusSemiring <: AbstractSemiring end

@inline szero(::MaxPlusSemiring) = -Inf
@inline sone(::MaxPlusSemiring) = 0.0
@inline oplus(::MaxPlusSemiring, a, b) = max(a, b)
@inline otimes(::MaxPlusSemiring, a, b) = a + b

# ─── Min-Plus Semiring (min, +, +∞, 0) — Tropical / Shortest Path ───────────

struct MinPlusSemiring <: AbstractSemiring end

@inline szero(::MinPlusSemiring) = Inf
@inline sone(::MinPlusSemiring) = 0.0
@inline oplus(::MinPlusSemiring, a, b) = min(a, b)
@inline otimes(::MinPlusSemiring, a, b) = a + b

# ─── PLN Semiring (max, *, 0, 1) — Q_PLN ────────────────────────────────────

struct PLNSemiring <: AbstractSemiring end

@inline szero(::PLNSemiring) = 0.0
@inline sone(::PLNSemiring) = 1.0
@inline oplus(::PLNSemiring, a, b) = max(a, b)
@inline otimes(::PLNSemiring, a, b) = a * b

# ─── Cost Semiring (min, +, +∞, 0) — Q_cost / Occam ─────────────────────────

struct CostSemiring <: AbstractSemiring end

@inline szero(::CostSemiring) = Inf
@inline sone(::CostSemiring) = 0.0
@inline oplus(::CostSemiring, a, b) = min(a, b)
@inline otimes(::CostSemiring, a, b) = a + b

# ─── Generic Semiring Operations ─────────────────────────────────────────────

"""
    semiring_matmul(sr, A, B) → C

Generalized matrix multiply: C[i,k] = ⊕_j (A[i,j] ⊗ B[j,k])
Works with any semiring. This is the CPU reference implementation;
GPU version will use KernelAbstractions.jl.
"""
function semiring_matmul(sr::AbstractSemiring, A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(A)
    n2, p = size(B)
    @assert n == n2 "Inner dimensions must match: A is $(m)×$(n), B is $(n2)×$(p)"

    C = fill(szero(sr), m, p)
    for i in 1:m
        for k in 1:p
            acc = szero(sr)
            for j in 1:n
                acc = oplus(sr, acc, otimes(sr, A[i, j], B[j, k]))
            end
            C[i, k] = acc
        end
    end
    return C
end

"""
    semiring_matvec(sr, A, x) → y

Generalized matrix-vector multiply: y[i] = ⊕_j (A[i,j] ⊗ x[j])
"""
function semiring_matvec(sr::AbstractSemiring, A::AbstractMatrix, x::AbstractVector)
    m, n = size(A)
    @assert n == length(x)

    y = fill(szero(sr), m)
    for i in 1:m
        acc = szero(sr)
        for j in 1:n
            acc = oplus(sr, acc, otimes(sr, A[i, j], x[j]))
        end
        y[i] = acc
    end
    return y
end

"""
    semiring_reduce(sr, v) → scalar

Reduce a vector using ⊕: result = v[1] ⊕ v[2] ⊕ ... ⊕ v[n]
"""
function semiring_reduce(sr::AbstractSemiring, v::AbstractVector)
    acc = szero(sr)
    for x in v
        acc = oplus(sr, acc, x)
    end
    return acc
end

"""
    threshold(x, t=0) → Bool

Heaviside threshold: H(x) = x > t. Used for existential quantification.
"""
@inline threshold(x, t=0) = x > t

end # module
