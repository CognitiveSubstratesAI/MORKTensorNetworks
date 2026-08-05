"""
ECANTensorBridge.jl — ECAN Attention as Tensor Operations

╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ 🛑 STOP. READ THIS BEFORE ADDING ANYTHING TO THIS FILE.                                      ║
║                                                                                              ║
║ THE OWNER OF ECAN LOGIC IS `Core/lib/ecan/*.metta`. NOT THIS FILE.                           ║
║ This file may contain a NUMERIC KERNEL and nothing else: given a matrix and a vector,        ║
║ multiply them. No policy constants. No thresholds. No selection rules. No tiering.           ║
║                                                                                              ║
║ WHY, in one line: MeTTa rules are ATOMS IN A SPACE — the system can rewrite them at runtime. ║
║ `Core/lib/ecan/ECAN_Policies.metta:62` says so verbatim: "Spreading Parameters (overridden   ║
║ by self-evolution)". A Julia keyword default is invisible to PLN, MOSES, the supercompiler   ║
║ and `attention-evolution-step!`. Policy written here is policy REMOVED from the cognitive    ║
║ loop. That is not a style preference; it is the difference between a system that can modify  ║
║ its own attention allocation and one that cannot.                                            ║
║                                                                                              ║
║ THIS MISTAKE HAS BEEN MADE TWICE IN THIS FILE. Do not make it a third time.                  ║
║   2026-05-10  14d49fe  ECAN implemented here in Julia, from an internal TODO row, without    ║
║                        anyone opening Core/lib/ecan/. Diverged from Core on four points.     ║
║   2026-06-04  c30e8b3  An audit caught it and wrote INTO THIS FILE: "Core's MeTTa ECAN       ║
║                        (Core/lib/ecan/) is the only ground truth available." Left open.      ║
║   2026-08-05  74694fa  While FIXING the first instance, MORE policy was added here —         ║
║                        including a reimplementation of Core's `combine-prob-vectors`         ║
║                        (SpreadingActivation.metta:126) that ALREADY EXISTED, and a constant  ║
║                        hardcoded at 0.05 when Core's own atom says 0.5. A 10x silent         ║
║                        divergence, introduced on the day it was written, by someone who had  ║
║                        read the audit note above and quoted the project's own                ║
║                        "check what already exists" rule earlier in the same session.         ║
║                                                                                              ║
║ THE CHECK THAT WOULD HAVE CAUGHT ALL THREE — sixty seconds, before writing any ECAN code:    ║
║     ls  ~/code/CognitiveSubstratesAI/Core/lib/ecan/                                          ║
║     grep -rn "^(= (" Core/lib/ecan/ECAN_Policies.metta Core/lib/ecan/AttentionPolicies.metta ║
║ If what you are about to write appears there, it is ALREADY IMPLEMENTED and you are about to ║
║ build a duplicate that will drift. Reading upstream `metta-attention` is NOT a substitute —  ║
║ Core is already a faithful port of it (SpreadingActivation.metta:3 cites it as its           ║
║ reference), so "upstream has X and we don't" is a claim about Core, and must be checked      ║
║ against Core.                                                                                ║
║                                                                                              ║
║ ⚠️ The parameters below are REQUIRED, deliberately — no defaults. A default here is a second ║
║ source of truth for a policy that Core already owns, and the 0.05-vs-0.5 divergence is what  ║
║ that costs. Pass them from the MeTTa side.                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

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

⚠️ WHAT CORE ALREADY HAS — DO NOT REIMPLEMENT ANY OF THIS HERE. Every line below was found in
`Core/lib/ecan/` AFTER a Julia duplicate of it had been written. The list exists so the next
session checks here first instead of repeating the search that produced the duplicate.

  incident + Hebbian combination   `combine-prob-vectors`   SpreadingActivation.metta:126
                                   documented at :9 as step 4 of the canonical algorithm, and at
                                   :122-124 citing upstream's `combineIncidentAdjacentVectors`.
                                   A Julia copy of this was written on 2026-08-05 as if it were a
                                   gap. It was not a gap.
  the Hebbian/incident split       `(hebbian-max-allocation-percentage) 0.5`
                                   ECAN_Policies.metta:68 — note 0.5, NOT upstream's 0.05. The
                                   Julia duplicate took upstream's value and disagreed with Core
                                   by 10x from the moment it was written.
  two-tier rent (WA + AF)          `collect-wa-rent!` core_logic.metta:223
                                   `collect-af-rent!` core_logic.metta:233
                                   `collect-all-wa-rent!` / `collect-all-af-rent!`
                                   AttentionPolicies.metta:75,83, with rates
                                   `(af-sti-rent-rate) 0.05` / `(af-lti-rent-rate) 0.02`
  spreading policy constants       `(max-spread-percentage) 0.3`, `(max-spreading-depth) 3`,
                                   `(spreading-decay-factor) 0.7`, `(spreading-threshold) 0.1`
                                   ECAN_Policies.metta:64-68 — under the header
                                   "Spreading Parameters (overridden by self-evolution)"
  conservative transfer            `trade-sti!` SpreadingActivation.metta:60 — clamps to the
                                   source's available STI, zero-sum
  probability vectors              `incident-prob-vector` :79, `hebbian-prob-vector` :99,
                                   `normalise-prob-vector` :111
  funds, forgetting, AF, decay     core_logic.metta §3, Forgetting.metta (9 rules),
                                   state_logic.metta (16 rules), AttentionPolicies.metta
  fluid ECAN                       FluidECAN.metta — already present

MISSING FROM THIS FILE (a bridge-side gap ONLY — Core has all of these; nothing here is a
system-level gap, and a plan row saying otherwise is wrong):

  - NO ELAPSED-TIME DECAY IN THE WA SPREAD AMOUNT. Upstream's `calculateDiffusionAmountWA` is
    `getSti - diffusedValue(atom, maxSpread)` where `diffusedValue = sti × (1-decayRate)^elapsed`
    (attention-bank/…/stochastic-importance-diffusion.metta:109-119). At elapsed = 1 that equals
    the AF amount `sti × MAX_SPREAD_PERCENTAGE`, which is what we implement; the compounding over
    per-atom elapsed time needs timestamps `ECANState` does not carry.
  - WA SOURCE SAMPLING IS NOT STOCHASTIC. `WAImportanceDiffusionAgent` draws sources via
    `getRandomAtomNotInAF`; `ecan_below_focus` returns the whole non-AF candidate set. Sample it
    yourself for upstream's behaviour.
  - AF MEMBERSHIP IS THRESHOLD-ONLY — no rank rules, no caps, no `recentMaxSTI` normalisation.
  - NO FUND. Collected rent is returned to the caller rather than booked, so the rent/wage cycle
    only conserves if the caller pipes one into the other.
  - LTI carried but never updated; no VLTI; no link creation/removal, so topology is frozen;
    Hebbian update is a plain symmetric product where Core's `hebbian-conjunction` is an
    asymmetric affine map on normalised STI.
  - RENT IS SINGLE-TIER. Core charges WA rent on all atoms plus AF rent on focus atoms, against
    both STI and LTI. `ecan_collect_rent!` is one threshold-gated STI-only deduction. (The
    two-tier split now exists for SPREADING, via `sources`, but not for rent.)

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
export ecan_attentional_focus, ecan_below_focus

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
mutable struct ECANState{A}
    sti::Vector{Float32}      # RAW STI — unbounded, may be negative
    lti::Vector{Float32}      # carried; no update rule here (see header)
    C::Matrix{Float32}        # Hebbian connection matrix C[src,dst]; dense for small spaces
    S::Matrix{Float32}        # STRUCTURAL incidence S[src,dst]; all-zero ⇒ Hebbian-only spreading
    atom_ids::Vector{A}       # atom_ids[i] = identifier for index i
end

"""
    ECANState(n)            → ECANState{Int}
    ECANState(atom_ids)     → ECANState{eltype(atom_ids)}

Empty ECAN state. STI and LTI zeroed; C and S zeroed (no links).

PARAMETRIC IN THE ID TYPE. `atom_ids` was `Vector{Any}`, which this project prohibits — an
`Any`-typed container is a type-instability barrier and defeats dispatch on the identifier.
Pass your own id vector to fix the type (`ECANState(Symbol[:a, :b])`,
`ECANState(["atom1", "atom2"])`); `ECANState(n)` gives `ECANState{Int}` with ids `1:n`.

`0` is the no-link value, not `-Inf`. `-Inf` was the (max,+) annihilator; under (+,×) the
annihilator is `0`, and a Hebbian link of strength 0 contributes nothing to spreading, so the
two readings coincide.
"""
function ECANState(atom_ids::Vector{A}) where {A}
    n = length(atom_ids)
    ECANState{A}(
        zeros(Float32, n), zeros(Float32, n),
        zeros(Float32, n, n), zeros(Float32, n, n),
        atom_ids,
    )
end

ECANState(n::Int) = ECANState(collect(1:n))

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
function ecan_build_diffusion_matrix(
    C::AbstractMatrix{Float32},
    S::Union{Nothing,AbstractMatrix{Float32}}=nothing;
    max_spread::Float32,                 # REQUIRED — Core: (max-spread-percentage)
    hebbian_max_allocation::Float32,     # REQUIRED — Core: (hebbian-max-allocation-percentage)
)
    n = size(C, 1)
    @assert size(C, 2) == n "C must be square; got $(size(C))"
    @assert 0.0f0 <= max_spread <= 1.0f0 "max_spread must lie in [0,1]; got $max_spread"
    @assert 0.0f0 <= hebbian_max_allocation <= 1.0f0 "hebbian_max_allocation must lie in [0,1]"
    S === nothing || @assert size(S) == (n, n) "S must match C; got $(size(S)) vs $(size(C))"

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

    # STRUCTURAL INCIDENCE (upstream `combineIncidentAdjacentVectors`). When S is supplied, the
    # Hebbian vector is capped at `hebbian_max_allocation` of each column's budget and the
    # structural-incidence vector takes the remainder, so the column still sums to 1 before the
    # max_spread scaling below. Upstream:
    #   hebbianDiffusionAvailable = HEBBIAN_MAX_ALLOCATION_PERCENTAGE × 1.0     (0.05)
    #   hebbianProportion         = each Hebbian weight × hebbianMaximumLinkAllocation
    #   incidentProportion        = each incident weight × (1 - hebbianDiffusionUsed)
    #   final                     = concat(hebbianProportion, incidentProportion)
    # and `probabilityVectorIncident` weights every structural neighbour equally at 1/nI.
    #
    # ⚠️ DELIBERATE DIVERGENCE, recorded rather than copied or silently corrected.
    # Upstream normalises the Hebbian vector by nH TWICE: `probabilityVectorHebbianAjacent`
    # (ImportanceDiffusionBase.metta:120-136) already sets maxAllocation = 1.0/atomCount, and
    # `combineIncidentAdjacentVectors` (:145-166) then multiplies by
    # hebbianMaximumLinkAllocation = HEBBIAN_MAX_ALLOCATION_PERCENTAGE/adajecentSize. The Hebbian
    # share therefore scales as 0.05·(strength×confidence)/nH², i.e. it VANISHES as an atom gains
    # Hebbian links — which cannot be the meaning of a constant named "MAX_ALLOCATION_PERCENTAGE".
    # We apply the single normalisation the name implies, so the Hebbian share is at most
    # `hebbian_max_allocation` regardless of nH. Not replicating an unexplained behaviour as if it
    # were a contract; not quietly patching it either.
    if S !== nothing
        Sd = zeros(Float32, n, n)
        for src in 1:n, dst in 1:n
            src == dst && continue
            S[src, dst] == 0.0f0 && continue
            Sd[dst, src] += abs(S[src, dst])      # incidence is undirected-in-effect, never negative
        end
        for j in 1:n
            heb = 0.0f0
            inc = 0.0f0
            for i in 1:n
                i == j && continue
                heb += D[i, j]
                inc += Sd[i, j]
            end
            inc == 0.0f0 && continue              # no structural neighbours ⇒ Hebbian keeps the column
            if heb > 0.0f0
                # Hebbian normalised to at most `hebbian_max_allocation` of the budget...
                hscale = hebbian_max_allocation / heb
                for i in 1:n
                    i != j && (D[i, j] *= hscale)
                end
                iscale = (1.0f0 - hebbian_max_allocation) / inc
                for i in 1:n
                    i != j && (D[i, j] += Sd[i, j] * iscale)
                end
            else
                # ...and with no Hebbian links the structural vector takes the whole budget.
                iscale = 1.0f0 / inc
                for i in 1:n
                    i != j && (D[i, j] += Sd[i, j] * iscale)
                end
            end
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
function ecan_sti_spread!(
    state::ECANState;
    max_spread::Float32,                 # REQUIRED — Core: (max-spread-percentage)
    hebbian_max_allocation::Float32,     # REQUIRED — Core: (hebbian-max-allocation-percentage)
    sources::Union{Nothing,AbstractVector{Int}}=nothing,
)::ECANState
    n = length(state.sti)
    n == 0 && return state
    D = ecan_build_diffusion_matrix(
        state.C, state.S; max_spread=max_spread,
        hebbian_max_allocation=hebbian_max_allocation,
    )

    # SOURCE RESTRICTION — the two-tier WA/AF split, expressed matrix-natively. Upstream runs two
    # agents differing in which atoms are picked as diffusion SOURCES: AFImportanceDiffusionAgent
    # draws from the attentional focus, WAImportanceDiffusionAgent from `getRandomAtomNotInAF`.
    # In the matrix form that is exactly a mask on COLUMNS: a non-source atom gets an identity
    # column, so it keeps all its STI and pays nobody. Left-stochasticity — hence conservation —
    # is preserved by construction, since an identity column also sums to 1.
    if sources !== nothing
        is_source = falses(n)
        for j in sources
            1 <= j <= n || throw(ArgumentError("source index $j out of range 1:$n"))
            is_source[j] = true
        end
        for j in 1:n
            is_source[j] && continue
            for i in 1:n
                D[i, j] = 0.0f0
            end
            D[j, j] = 1.0f0
        end
    end
    # Use the package's own semiring primitive rather than `*`, so the algebra is explicit and a
    # future GPU semiring kernel swaps in here. Previously MaxPlusSemiring was IMPORTED and then
    # never used — the (max,+) loop was hand-inlined. `semiring_matvec` also accumulates in
    # Float64 before narrowing, which keeps the conservation residual near eps(Float32).
    state.sti = semiring_matvec(SumProductSemiring(), D, state.sti)
    state
end

"""
    ecan_attentional_focus(state; af_threshold=0.5f0) → Vector{Int}
    ecan_below_focus(state; af_threshold=0.5f0) → Vector{Int}

Indices of atoms in / below the Attentional Focus, by STI threshold.

Feed either into [`ecan_sti_spread!`](@ref)'s `sources` to get upstream's two tiers:

    ecan_sti_spread!(st; sources = ecan_attentional_focus(st))   # AFImportanceDiffusionAgent
    ecan_sti_spread!(st; sources = ecan_below_focus(st))         # WAImportanceDiffusionAgent

⚠️ Threshold-only. Upstream and Core also apply rank rules and caps to AF membership, and
`WAImportanceDiffusionAgent` samples its sources RANDOMLY from the non-AF set
(`getRandomAtomNotInAF`) rather than taking all of them — so `ecan_below_focus` is the full
candidate set, not one stochastic draw from it. Sample it yourself if you want upstream's
stochastic behaviour.
"""
function ecan_attentional_focus(state::ECANState, af_threshold::Float32)::Vector{Int}
    findall(>=(af_threshold), state.sti)
end

function ecan_below_focus(state::ECANState, af_threshold::Float32)::Vector{Int}
    findall(<(af_threshold), state.sti)
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
function ecan_hebbian_update!(state::ECANState, η::Float32, decay::Float32)::ECANState
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
    state::ECANState, af_threshold::Float32, rent_rate::Float32
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
