"""
ECANTensorBridge.jl — ECAN Attention as Tensor Operations

Normative source: **Ikle', Pitt, Goertzel, Sellman, "Economic Attention Networks: Associative
Memory and Resource Allocation for General Intelligence", AGI-2009**, §5.4 — extracted at
`docs/specs/Algorithms/Ecan/economic_attention_networks_2009_spec.md`.
Reference implementation: **iCog Labs `metta-attention`**, `dev-zone/metta-attention/attention/`
(the repo `docs/research/papers/Algorithms/ECAN Attention/ecan links.txt` points at,
`github.com/singnet/attention`).
Alignment target: Core's MeTTa ECAN lane, `Core/lib/ecan/` (8 modules, 132 rules).

FOUR INDEPENDENT SOURCES AGREE that ECAN spreading is a conservative, normalised, left-stochastic
operator — checked before writing this, because the previous implementation had none of them:
  - AGI-2009 §5.4 — "a version of the connection matrix C normalized so that D is a left
    stochastic matrix"; §3.2 calls conservation "the key dynamical difference from an ordinary
    attractor neural network, in which there is no law of conservation of activation."
  - `metta-attention/…/ImportanceDiffusionBase.metta` — `tradeSti` is literally zero-sum
    (`newSourceSTI = sourceSTI - value`, `newTargetSti = targetSti + value`), and the targets
    come from `probabilityVectorIncident` (1/size) and `probabilityVectorHebbianAjacent`.
  - "Going With the Flow" §4.1 — "a left-stochastic operator D, so Σ STI is conserved";
    conservation PROVED as its Lemma 4.1, with a renormalisation remedy in Remark 4.6.
  - TECAN §2.1 — classic ECAN spreads "over Hebbian and inverse-Hebbian links, often through a
    left-stochastic or approximately budget-preserving operator"; TECAN's own Eq. 50 keeps
    conservation (inbound J_ji at i is debited at j), and §4.5 warns that unbooked credit means
    "a loop can learn to counterfeit its own metabolic support" — which is precisely what
    copying a neighbour's STI without debiting the neighbour does.

§5.4 specifies importance spreading AS A TENSOR OPERATION already:

    v' = D v

where **D** is the connection matrix **C** normalised so that D is LEFT-STOCHASTIC (every column
sums to 1). Because each column sums to 1, Σᵢ(Dv)ᵢ = Σⱼ vⱼ — the total STI is conserved EXACTLY.
That conservation is not a detail: §3.2 of the same paper calls it *"the key dynamical difference
from an ordinary attractor neural network, in which there is no law of conservation of
activation."* Non-conservative spreading is the thing ECAN is defined in opposition to.

Construction of D from C (§5.4), with j the SOURCE (column) and i the DESTINATION (row):

    if c_ij ≥ 0:  d_ij = c_ij          else:  d_ji = -c_ij     (inverse-Hebbian links REVERSE)
    if Σᵢ d_ij > ⟨MaxSpread⟩:  scale off-diagonals to ⟨MaxSpread⟩;  d_jj = 1 - ⟨MaxSpread⟩
    else:                                                          d_jj = 1 - Σ_{i≠j} d_ij

The diagonal is what an atom KEEPS. An atom with no outgoing links gets d_jj = 1 and retains its
STI untouched. ⟨MaxSpread⟩ matches Core's `(max-spread-percentage)` = 0.3 (ECAN_Policies.metta:67).

────────────────────────────────────────────────────────────────────────────────
PROVENANCE — the AUDIT 2026-06-04 L3/H6 items are RESOLVED here (2026-08-05).

L3 said the "§7.3.x" citations were DANGLING. They are not dangling — they RESOLVE, just not to
a paper: `docs/tracking/MORK_TENSOR_NETWORKS_TODO_2026-04-09.md` §7.3 "ECAN ↔ Tensor Bridge",
rows 7.3.1/7.3.2/7.3.3 (lines 237-244), matching the old docstring verbatim. The word "TODO" in
the old header `"MORK-Tensor-Networks §5+ECAN, TODO §7.3"` was that DOCUMENT'S NAME, not a status
marker. That TODO's own "Paper Ref" column reads "§3 + ECAN" — it declared itself an
extrapolation. MORK-Tensor-Networks.pdf has 6 sections, no §7, and the string "ECAN" occurs
0 times in its 10 pages (verified against the PDF, not only the extraction; both on-disk copies
byte-identical, no /Outlines, no alternate edition in the 249-PDF corpus).

H6 posed an OWNER DECISION: (a) align to Core's semantics, or (b) find a source authorising the
simplified tensor form. Branch (b) is CLOSED — no paper authorises it, and the one paper that
proposes changing ECAN's transport at all ("Going With the Flow", spec at
`docs/specs/Algorithms/Ecan/going_with_the_flow_ecan_fluid_spec.md`) does the opposite: its §4.1
recaps spreading as "a left-stochastic operator D, so Σ STI is conserved", it PROVES budget
conservation as Lemma 4.1, and supplies a renormalisation remedy in Remark 4.6 for numerical
drift. Its §4.8 is explicit: "Unchanged: ECAN's accounting … Changed: the transport mechanism."

So this file now takes branch (a). Note that aligning is NOT a retreat from tensors — the
alignment target IS a tensor operation. Only the SEMIRING was wrong.

WHAT THE PREVIOUS IMPLEMENTATION DID, and why each part was wrong:

  STI_new[x] = max_y (W[x,y] + STI[y])          -- a (max,+) Viterbi best-path score

  1. WRONG SEMIRING. (max,+) computes a best path. ECAN diffuses a currency. The (max,+)
     machinery came from MORK-Tensor-Networks §3.5/§5.3, which specifies it for best-path
     scoring over ARC LOG-PROBABILITIES — a real formula, applied to the wrong quantity.
  2. NOT CONSERVATIVE. `W[x,y] + STI[y]` COPIES y's importance into x; y keeps its own. Total
     STI grows without bound. Core's `trade-sti!` (SpreadingActivation.metta:60-67) deducts from
     the source and clamps the transfer to what the source actually holds — zero-sum per transfer.
  3. TRANSPOSED. `ecan_build_weight_matrix` writes W[src,dst]; the spread read W[x,y] and added
     sti[y] into x, i.e. pulled along the OUTGOING edge x→y. D[dst,src] is the required
     orientation, which is why `ecan_build_diffusion_matrix` transposes explicitly below.
  4. NO DIAGONAL. With no d_jj term, `best` won outright and an atom's own STI was DISCARDED the
     moment it had any incoming link.
  5. FIXED RAW-STI CAP. `clamp(·, 0, max_spread=1.0)` is precisely the scale bug Core removed on
     2026-06-30 (core_logic.metta:120-126): raw STI rides the economic scale
     (funds/target/wage at 100000/10000/10), so a fixed cap of 1.0 makes it impossible to absorb
     attention from the pool and makes trades non-zero-sum. Raw STI is UNBOUNDED here; the [0,1]
     concentration is a normalised VIEW (`get-normalised-sti`), never a clamp.

DELIBERATE DEVIATION FROM §5.4, recorded per house convention. The 2009 paper additionally
min-max scales STI into v before the multiply (v_i = (s_i - minSTI)/(maxSTI - minSTI)) and
rescales afterwards. That step CONTRADICTS the same section's own conservation statement
("the vector v is simply the total STI times a probability vector") — a min-max scaled vector
does not sum to the total, and an affine rescale does not commute with Dv. We apply D to RAW STI,
which conserves Σ STI exactly and matches Core's treatment of raw STI as unbounded economic scale.
Where the paper is internally inconsistent, Core's live semantics break the tie.

STILL NOT IMPLEMENTED (unchanged by this commit, listed so nobody reads silence as parity):

  - HEBBIAN LINKS ONLY. D is built from the Hebbian connection matrix C. Upstream diffuses along
    BOTH structural incidence and Hebbian adjacency, combining the two probability vectors under
    HEBBIAN_MAX_ALLOCATION_PERCENTAGE (`combineIncidentAdjacentVectors`); Core mirrors this
    (`incident-prob-vector` + `hebbian-prob-vector`, SpreadingActivation.metta:79-140). A caller
    wanting the structural half must fold it into C themselves.
  - ONE TIER. Upstream has separate WA (whole-atomspace) and AF (attentional-focus) diffusion
    agents, and Core's rent is likewise two-tier charging both STI and LTI. This is a single
    undifferentiated sweep.
  - NO FUND. Collected rent is returned to the caller rather than booked, so the rent/wage cycle
    only conserves if the caller pipes one into the other.
  - LTI carried but never updated; no VLTI; no link creation/removal, so topology is frozen;
    Hebbian update is a plain symmetric product where Core's `hebbian-conjunction` is an
    asymmetric affine map on normalised STI.

This file is a CONSERVATIVE SPREADING KERNEL that agrees with Core and upstream on transport —
not a full ECAN.
────────────────────────────────────────────────────────────────────────────────
"""

using ..Semirings: SumProductSemiring, semiring_matvec
using SparseArrays
using LinearAlgebra

export ECANState, ecan_sti_spread!, ecan_hebbian_update!, ecan_apply_decay!
export ecan_collect_rent!, ecan_distribute_wages!
export ecan_build_weight_matrix, ecan_build_diffusion_matrix, ecan_sti_vector

# ─── Hebbian connection matrix C ─────────────────────────────────────────────

"""
    ECANState

ECAN attention state as tensors:

  sti      — Short-Term Importance vector (n_atoms). RAW, economic-scale, UNBOUNDED and
             signed: an atom may hold negative STI (debt). Do not clamp it — see the
             2026-06-30 note in the file header.
  lti      — Long-Term Importance vector (n_atoms). Carried, never updated here.
  C        — Hebbian CONNECTION matrix, `C[src, dst]` (the paper's **C**). 0 means "no link".
             DENSE `Matrix{Float32}`; the old docstring claimed sparse and was wrong. Dense is
             deliberate for small spaces — see the struct field comment.
  atom_ids — atom_ids[i] = identifier for index i.
"""
mutable struct ECANState
    sti::Vector{Float32}      # RAW STI — unbounded, may be negative
    lti::Vector{Float32}      # carried; no update rule here (see header)
    C::Matrix{Float32}        # Hebbian connection matrix C[src,dst]; dense for small spaces
    atom_ids::Vector{Any}     # atom_ids[i] = identifier for index i
end

"""
    ECANState(n) → ECANState

Empty ECAN state for `n` atoms. STI and LTI zeroed; C zeroed (no links).

`0` is the no-link value, not `-Inf`. `-Inf` was the (max,+) annihilator; under (+,×) the
annihilator is `0`, and a Hebbian link of strength 0 contributes nothing to spreading, so the
two readings coincide.
"""
function ECANState(n::Int)
    ECANState(zeros(Float32, n), zeros(Float32, n), zeros(Float32, n, n), Any[i for i in 1:n])
end

"""
    ecan_build_weight_matrix(links, n) → Matrix{Float32}

Build the Hebbian connection matrix **C** from a link list.
`links` = `(src_idx, dst_idx, weight)` triples; `C[src, dst] = weight`, 0 where no link exists.

Weights may be NEGATIVE — those are inverse-Hebbian links, and
[`ecan_build_diffusion_matrix`](@ref) reverses their direction per §5.4.

Last write wins on a duplicate `(src, dst)`.
"""
function ecan_build_weight_matrix(links::Vector{<:Tuple}, n::Int)::Matrix{Float32}
    C = zeros(Float32, n, n)
    for (src, dst, w) in links
        C[src, dst] = Float32(w)
    end
    C
end

"""
    ecan_sti_vector(state) → Vector{Float32}

Current STI as a column vector, defensively copied.
"""
ecan_sti_vector(s::ECANState) = copy(s.sti)

# ─── Diffusion matrix D (AGI-2009 §5.4) ──────────────────────────────────────

"""
    ecan_build_diffusion_matrix(C; max_spread=0.3f0) → Matrix{Float32}

Build the LEFT-STOCHASTIC diffusion matrix **D** from connection matrix **C**, per AGI-2009 §5.4.

`D[dst, src]` — note the TRANSPOSE relative to `C[src, dst]`. Column `j` is source `j`'s outgoing
distribution, which is what §5.4 normalises and caps ("caps each node's outgoing spread at a
⟨MaxSpread⟩ proportion of its STI"). Getting this orientation wrong is silent: the spread still
runs, and importance simply flows the wrong way down every asymmetric link.

Per §5.4:
  - `c_ij ≥ 0` contributes in the link's own direction; `c_ij < 0` (inverse-Hebbian) contributes
    `-c_ij` in the REVERSE direction.
  - each column is normalised to a probability vector, scaled to `max_spread`, and completed
    with a self-retention diagonal `d_jj = 1 - max_spread`, so the column sums to exactly 1.
    A column with no outgoing links gets `d_jj = 1`. See the inline note on why normalisation
    comes first — it is what reconciles §5.4 with upstream `metta-attention` and Core.

POSTCONDITION: every column sums to 1 (to Float32 rounding), hence `Σ(Dv) = Σv` for any `v`.
`max_spread` defaults to 0.3 = Core's `(max-spread-percentage)` (ECAN_Policies.metta:67).
"""
function ecan_build_diffusion_matrix(C::AbstractMatrix{Float32}; max_spread::Float32=0.3f0)
    n = size(C, 1)
    @assert size(C, 2) == n "C must be square; got $(size(C))"
    @assert 0.0f0 <= max_spread <= 1.0f0 "max_spread must lie in [0,1]; got $max_spread"

    D = zeros(Float32, n, n)
    for src in 1:n, dst in 1:n
        src == dst && continue                    # self-links are the diagonal's job, set below
        c = C[src, dst]
        c == 0.0f0 && continue
        if c > 0.0f0
            D[dst, src] += c                      # forward: src pays dst
        else
            D[src, dst] += -c                     # inverse-Hebbian: dst pays src (§5.4)
        end
    end

    # Column-normalise to a PROBABILITY VECTOR, then scale by max_spread and complete with the
    # self-retention diagonal.
    #
    # WHY NORMALISE FIRST — this reconciles §5.4 with the reference implementation. §5.4 reads
    # "if Σᵢ dᵢⱼ > ⟨MaxSpread⟩: scale to ⟨MaxSpread⟩, dⱼⱼ = 1 - ⟨MaxSpread⟩; else dⱼⱼ = 1 - Σ",
    # i.e. it branches on the RAW weight sum. Upstream `metta-attention` instead builds a
    # normalised probability vector and spends exactly `STI × MAX_SPREAD_PERCENTAGE`
    # (ImportanceDiffusionBase.metta: `probabilityVectorIncident` = 1/size,
    # `probabilityVectorHebbianAjacent` = maxAllocation × hebbian%, `calculateDiffusionAmount`
    # = getSti × MAX_SPREAD_PERCENTAGE), and Core does the same (`normalise-prob-vector`,
    # SpreadingActivation.metta:111; `(max-spread-percentage) 0.3`, ECAN_Policies.metta:67).
    # The two AGREE: once a column is normalised to 1, §5.4's `Σ > ⟨MaxSpread⟩` branch fires
    # unconditionally (any ⟨MaxSpread⟩ < 1), yielding exactly this. Following the raw-weight
    # reading instead would make the spread amount depend on the arbitrary scale of the Hebbian
    # weights, which upstream deliberately normalises away.
    for j in 1:n
        outgoing = 0.0f0
        for i in 1:n
            i != j && (outgoing += D[i, j])
        end
        if outgoing > 0.0f0
            scale = max_spread / outgoing         # normalise to 1, then take max_spread of it
            for i in 1:n
                i != j && (D[i, j] *= scale)
            end
            D[j, j] = 1.0f0 - max_spread
        else
            D[j, j] = 1.0f0                       # no outgoing links ⇒ keeps all its STI
        end
    end
    D
end

# ─── STI spreading: v' = D v  (AGI-2009 §5.4) ────────────────────────────────

"""
    ecan_sti_spread!(state; max_spread=0.3f0) → state

One importance-spreading step, `v' = D v`, over the (+,×) semiring — AGI-2009 §5.4.

CONSERVES Σ STI exactly (up to Float32 rounding), because D is left-stochastic. This is the
defining invariant of ECAN as an economy; see the file header.

Decay is deliberately NOT applied here. Core keeps forgetting as a separate policy step
(`apply-decay!`, ECAN_Policies.metta), and folding it in would silently break conservation —
which is how the previous implementation's `decay` parameter hid its non-conservation behind a
plausible-looking shrink. Use [`ecan_apply_decay!`](@ref) explicitly.
"""
function ecan_sti_spread!(state::ECANState; max_spread::Float32=0.3f0)::ECANState
    n = length(state.sti)
    n == 0 && return state
    D = ecan_build_diffusion_matrix(state.C; max_spread=max_spread)
    # Use the package's own semiring primitive rather than `*`, so the algebra is explicit and a
    # future GPU semiring kernel swaps in here. Previously MaxPlusSemiring was IMPORTED and then
    # never used — the (max,+) loop was hand-inlined. `semiring_matvec` also accumulates in
    # Float64 before narrowing, which keeps the conservation residual near eps(Float32).
    state.sti = semiring_matvec(SumProductSemiring(), D, state.sti)
    state
end

"""
    ecan_apply_decay!(state, rate) → state

Multiply every STI by `rate` — ECAN's forgetting/decay policy pass, kept separate from transport.

NOT CONSERVATIVE, by design: this is where importance is meant to leave the system. Keeping it
out of [`ecan_sti_spread!`](@ref) is what lets the spreading step assert conservation.
"""
function ecan_apply_decay!(state::ECANState, rate::Float32)::ECANState
    @. state.sti *= rate
    state
end

# ─── Hebbian weight update ───────────────────────────────────────────────────

"""
    ecan_hebbian_update!(state; η=0.01f0, decay=0.99f0) → state

Update Hebbian connection weights for co-active pairs:

    C[x,y] ← clamp(decay × C[x,y] + η × STI[x] × STI[y], -10, 10)

Only pre-existing links are touched (`C[x,y] != 0`), so topology is FROZEN — no link is ever
created or removed. Core does create and remove links (SpreadingActivation.metta:234-261); that
divergence is listed in the file header and is not addressed here.

Also unlike Core: this is a plain symmetric product on RAW STI, where Core's
`hebbian-conjunction` is an asymmetric affine map on NORMALISED STI. Retained as-is deliberately
— this commit's scope is the transport operator, not the learning rule.
"""
function ecan_hebbian_update!(
    state::ECANState; η::Float32=0.01f0, decay::Float32=0.99f0
)::ECANState
    n = length(state.sti)
    sti = state.sti
    C = state.C

    for x in 1:n, y in 1:n
        C[x, y] == 0.0f0 && continue   # no link — topology is frozen
        Δ = η * sti[x] * sti[y]
        C[x, y] = clamp(decay * C[x, y] + Δ, -10.0f0, 10.0f0)
    end
    state
end

# ─── Attention fund — rent/wage as tensor reductions ─────────────────────────

"""
    ecan_collect_rent!(state; af_threshold=0.5f0, rent_rate=0.1f0) → Float32

Collect rent from atoms above the Attentional Focus threshold, as a reduction:

    rent[x]    = max(0, STI[x] - af_threshold) × rent_rate
    total_rent = Σ_x rent[x]

Deducts in place and RETURNS the total. ⚠️ The collected STI is not booked anywhere — `ECANState`
has no fund field. Unless the caller feeds this return value to
[`ecan_distribute_wages!`](@ref), the STI is destroyed and the economy leaks. Core keeps a global
fund (core_logic.metta §3). See the file header's not-implemented list.
"""
function ecan_collect_rent!(
    state::ECANState; af_threshold::Float32=0.5f0, rent_rate::Float32=0.1f0
)::Float32
    total = 0.0f0
    for x in eachindex(state.sti)
        excess = max(0.0f0, state.sti[x] - af_threshold)
        rent = excess * rent_rate
        state.sti[x] -= rent
        total += rent
    end
    total
end

"""
    ecan_distribute_wages!(state, budget) → state

Distribute `budget` in proportion to positive STI:

    weight[x] = max(0, STI[x])
    wage[x]   = budget × weight[x] / Σ_y weight[y]

A plain proportional split — a normalisation, NOT a softmax. (The previous docstring called it
"softmax-style"; there is no exponential in the body.)

Pairs with [`ecan_collect_rent!`](@ref): passing that function's return value here makes the
rent/wage cycle conserve STI.
"""
function ecan_distribute_wages!(state::ECANState, budget::Float32)::ECANState
    weights = max.(0.0f0, state.sti)
    total = sum(weights)
    total == 0.0f0 && return state  # no positive STI — nothing to distribute
    @. state.sti += budget * weights / total
    state
end
