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
    semiring_reduce,
    heaviside_default

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
#
# N1 (audit 2026-06-04): PLNSemiring and CostSemiring are PRIMUS EXTENSIONS beyond
# the paper's §3.5, which defines exactly 4 semirings (Boolean, SumProduct, MaxPlus,
# MinPlus). PLN (max, *) is a genuine distinct semiring (probabilistic-logic truth
# ordering Q_PLN). CostSemiring below, however, is (min, +, Inf, 0) — BEHAVIOURALLY
# IDENTICAL to MinPlusSemiring (its GPU tag 6 also dispatches min/+ like tag 3). It is
# kept as a named alias for caller intent (Occam complexity Q_cost) but adds no new
# algebra; consider consolidating onto MinPlus if the Q_cost naming isn't load-bearing.

# ⚠️ PATH-INDEPENDENCE, CHECKED 2026-08-12 after Zarathustra Goertzel relayed a claim that this
# semiring double-counts shared evidence and so returns confidences that are TOO HIGH. Checked against
# the code: the claim is directionally INVERTED for this implementation, and a different gap is real.
#
#   ⊕ = max  is the ACROSS-PATHS merge. It cannot double-count: two derivation paths sharing a premise
#            yield the STRONGER, never the sum. Inflation needs ⊕ = `+` or probabilistic-OR (a+b-ab),
#            neither of which is here. The OPPOSITE defect is present — max DISCARDS corroboration,
#            where PLN's revision rule combines independent support into something stronger than
#            either input. This under-counts; it does not over-count.
#   ⊗ = *    is the ALONG-A-PATH chain, and multiplying link strengths DOES assume conditional
#            independence of successive links. A genuine assumption, and the standard PLN deduction
#            issue — PLN's own deduction formula needs more than the product.
#
# 🔴 A THIRD DEFECT, AND IT IS IN A PATH THAT DOES DISPATCH PLN. `path_compose` (PathAlgebra.jl)
# APPLIES THE HEAVISIDE STEP BY DEFAULT — `H(x) = sone if x != szero` — so composing two PLN relations
# returns 1.0 for every non-zero entry and the truth STRENGTH is destroyed. Its docstring lists only
# SumProduct (path counting) and MaxPlus (Viterbi) as the cases wanting `apply_threshold=false`; PLN is
# omitted, though a truth value is exactly the kind of weight that must not be projected to {0,1}.
# `semiring_tag(::PLNSemiring) = 5` and the GPU `oplus` dispatch (SemiringKernels.jl:77) mean PLN is
# genuinely routed through these kernels, so this is not hypothetical.
#
# ⚠️ AN EARLIER VERSION OF THIS NOTE SAID "no consumer, so neither bites today". That was WRONG on the
# facts (the GPU kernels dispatch tag 5; `path_universal` and `path_compose` both accept PLN) and wrong
# as a way to reason — see `[[feedback_never_deprioritize_by_consumer_count]]`. The defect is in the
# algebra; who calls it today is not what makes it a defect.
#
# 🔴 AND WE ALREADY HAVE THE CORRECT IMPLEMENTATION — IN MeTTa. `Core/lib/pln/pln_core_logic.metta`
# carries the PLN book's deduction formula (5.2.2.2, p.74; cross-referenced to trueagi-io/hyperon-pln
# and PeTTa's lib_pln.metta), which is NOT a product:
#
#     sAC = sAB*sBC + (1 - sAB) * (sC - sB*sBC) / (1 - sB)
#
# guarded by `conditional-probability-consistency` — the Fréchet bounds
# max(0,(A+B-1)/A) <= P(B|A) <= min(1,B/A) — which REJECTS a probabilistically impossible triple
# instead of computing with it. That guard is exactly the dependency structure a scalar semiring has
# nowhere to put, and it has been in the tree all along.
#
# MEASURED 2026-08-12, `simpleDeductionStrength` vs `otimes(PLNSemiring, sAB, sBC)`:
#
#     sA   sB   sC   sAB  sBC  |  PLN lib   a*b     diff
#     0.5  0.5  0.5  0.8  0.7  |  0.62      0.56    0.06
#     0.3  0.6  0.4  0.7  0.9  |  (empty)   0.63    — preconditions FAILED, semiring answered anyway
#     0.9  0.8  0.7  0.85 0.75 |  0.7125    0.6375  0.075
#     0.2  0.5  0.9  0.6  0.8  |  0.88      0.48    0.40
#     0.5  0.2  0.5  0.3  0.9  |  0.55      0.27    0.28
#
# The product UNDER-STATES in every case (confirming the ⊕/⊗ analysis above and contradicting the
# relayed "confidences too high"), by up to 0.40 absolute — a different answer, not a rounding gap. The
# second row is the worse failure: the MeTTa lib refuses an inconsistent premise set, the semiring
# launders it into a confident number.
#
# ⚠️ SO "NO ORACLE" WAS WRONG (an earlier version of this note said so). The oracle is in-tree, in
# another package, and the two PLN implementations were simply never differentially tested against each
# other. That cross-package differential is the work: `Core/lib/pln` is the reference, this is the
# approximation, and nothing currently asserts they agree.
#
# THIS IS WORK OWED. `(max, *)` is the wrong algebra for PLN truth values in BOTH directions above, and
# it is an ADDITION ABOVE THE PAPER'S FOUR SEMIRINGS — see N1 above: the source
# paper's §3.5 defines exactly four semirings and PLN is ours, which by
# `[[feedback_additions_above_upstream_need_own_oracle]]` is exactly the kind of addition that needs
# its own ground truth and has never had one.
#
# ⇒ THE MISSING STRUCTURE HAS A WORKED FORMALISATION TO BORROW (all three checked present):
#     MeTTapedia `lean/mettapedia/Mettapedia/PLN/WorldModel/PLNWorldModelOverlap.lean`
#                `PLN/RuleFamilies/FirstOrder/PLNMultiPathDependency.lean`
#                `PLN/Bridges/HOL/LedgerMultiPathAdapter.lean`  — union measure = sum − dependency
# A scalar semiring genuinely has nowhere to PUT an overlap term; that part of the relayed claim is
# right, and it is why the fix is a different algebra rather than a different constant.
struct PLNSemiring <: AbstractSemiring end

@inline szero(::PLNSemiring) = 0.0
@inline sone(::PLNSemiring) = 1.0
@inline oplus(::PLNSemiring, a, b) = max(a, b)
@inline otimes(::PLNSemiring, a, b) = a * b

"""
    heaviside_default(sr) -> Bool

Should `path_compose` apply the spec's Heaviside step H by DEFAULT under this semiring?

Spec §3 defines composition as `T[x,z] = H(⊕_y R[x,y] ⊗ S[y,z])`, and for the FOUR semirings the paper
defines, that stays the default — the spec's reading is reachability, and deviating from the paper is
how we get things wrong.

`PLNSemiring` is OURS, not the paper's (see N1 above), and H is wrong for it: `H(x) = sone if x != szero`
maps every non-zero composed truth value to `1.0`, so a composition of PLN relations returns
REACHABILITY with the strength destroyed. A truth value is the same kind of quantity as a SumProduct
count or a MaxPlus score — both of which the docstring already tells callers to obtain with
`apply_threshold=false`. PLN was simply missing from that list, and defaulting it correctly is better
than expecting every caller to remember.

⚠️ `CostSemiring` HAS THE SAME PROBLEM AND IS DELIBERATELY LEFT ALONE: H maps any finite cost to
`sone = 0.0`, destroying it. Not changed here because Cost is behaviourally identical to MinPlus (N1),
so giving it a different default would make the alias load-bearing in a new way — a decision about
whether `Q_cost` is a real semiring or a naming convenience, which is not this change.
"""
@inline heaviside_default(::AbstractSemiring) = true
@inline heaviside_default(::PLNSemiring) = false

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

    # H1 fix (audit 2026-06-04): seed with element type of inputs, not szero(sr) which
    # returns Float64. `fill(szero(sr), m, p)` allocates Matrix{Float64} even when A,B
    # are Float32, causing silent widening + type instability downstream (HRT/ECAN/Shard
    # code is Float32 throughout). Use T(szero(sr)) so output eltype matches inputs.
    T = promote_type(eltype(A), eltype(B))
    z = T(szero(sr))
    C = fill(z, m, p)
    for i in 1:m
        for k in 1:p
            acc = z
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

    T = promote_type(eltype(A), eltype(x))   # H1 fix: preserve input type
    y = fill(T(szero(sr)), m)
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

# N2 (audit 2026-06-04): removed dead `threshold(x, t=0) = x > t`. It was unexported,
# called nowhere, and embodied the exact C2 bug (hardcoded `> 0` Heaviside that is wrong
# for tropical semirings). The semiring-aware Heaviside lives in PathAlgebra._heaviside.

end # module
