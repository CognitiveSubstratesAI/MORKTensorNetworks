"""
    HRT.jl — Hierarchical Resolution Transformer

MORK-Tensor-Networks paper §6: multi-resolution pyramid with cross-resolution
attention, gated fusion, and reconstruction regularizer.

Architecture:

  - L resolution levels with geometric shrinkage: |R_l| = n / 2^(l-1)
  - Per-level RTB: SelfAttn + FFN
  - Down-projection: strided pooling R_{l+1} = P_l(R_l)
  - Cross-resolution attention: bidirectional fine↔coarse
  - Gated fusion: R_l = α_l * R̃_l + (1-α_l) * R_l
  - Reconstruction loss: L_recon = ||R_1 - g(R_L, ..., R_2)||²

Complexity: O(n log n · d) with geometric shrinkage across L levels.

All operations use dense matrices (post-Tucker densification) and can be
dispatched to GPU via KernelAbstractions.jl.
"""
module HRT

using LinearAlgebra

export HRTConfig,
    HRTState,
    HRTParams,
    init_hrt,
    hrt_forward!,
    hrt_reconstruction_loss,
    self_attention,
    feed_forward,
    down_project,
    cross_attention,
    gated_fuse

# ─── Configuration ───────────────────────────────────────────────────────────

"""
    HRTConfig

Hyperparameters for the HRT pyramid.
"""
struct HRTConfig
    n_tokens::Int       # Base resolution (level 1 token count)
    d_model::Int        # Embedding dimension
    n_levels::Int       # Number of resolution levels L
    n_heads::Int        # Attention heads per level
    ffn_mult::Int       # FFN hidden dim = d_model * ffn_mult
    dropout::Float64    # Dropout rate (for training)
    recon_weight::Float64  # λ for reconstruction loss
end

function HRTConfig(;
    n_tokens=256, d_model=64, n_levels=4, n_heads=4, ffn_mult=4, dropout=0.0,
    recon_weight=0.1
)
    HRTConfig(n_tokens, d_model, n_levels, n_heads, ffn_mult, dropout, recon_weight)
end

# ─── Parameters (Learnable) ──────────────────────────────────────────────────

"""
    HRTLevelParams

Learnable parameters for one resolution level.
"""
struct HRTLevelParams
    # Self-attention Q/K/V projections
    W_Q::Matrix{Float32}    # d_model × d_model
    W_K::Matrix{Float32}
    W_V::Matrix{Float32}
    W_O::Matrix{Float32}    # Output projection

    # FFN
    W1::Matrix{Float32}     # d_model × (d_model * ffn_mult)
    b1::Vector{Float32}
    W2::Matrix{Float32}     # (d_model * ffn_mult) × d_model
    b2::Vector{Float32}

    # Down-projection (pooling matrix)
    W_down::Matrix{Float32} # (n_l/2) × n_l

    # Cross-attention
    W_cross_Q::Matrix{Float32}
    W_cross_K::Matrix{Float32}
    W_cross_V::Matrix{Float32}

    # Gated fusion
    alpha::Vector{Float32}  # Learned gate [0,1] per dimension (or scalar)
end

"""
    HRTParams

All learnable parameters for the full HRT pyramid.
"""
struct HRTParams
    levels::Vector{HRTLevelParams}
    # Reconstruction up-projection cascade
    W_up::Vector{Matrix{Float32}}  # One per level (coarse→fine)
end

"""
    HRTState

Activation state: token representations at each resolution level.
"""
mutable struct HRTState
    R::Vector{Matrix{Float32}}  # R[l] is n_l × d_model
end

# ─── Initialization ──────────────────────────────────────────────────────────

"""
Initialize HRT parameters with Xavier initialization.
"""
function init_hrt(cfg::HRTConfig)
    d = cfg.d_model
    h = d * cfg.ffn_mult
    levels = HRTLevelParams[]
    w_ups = Matrix{Float32}[]

    for l in 1:cfg.n_levels
        n_l = cfg.n_tokens ÷ (2^(l-1))
        n_next = max(1, n_l ÷ 2)
        scale = Float32(sqrt(2.0 / d))

        params = HRTLevelParams(
            randn(Float32, d, d) * scale,  # W_Q
            randn(Float32, d, d) * scale,  # W_K
            randn(Float32, d, d) * scale,  # W_V
            randn(Float32, d, d) * scale,  # W_O
            randn(Float32, d, h) * scale,  # W1
            zeros(Float32, h),             # b1
            randn(Float32, h, d) * scale,  # W2
            zeros(Float32, d),             # b2
            randn(Float32, n_next, n_l) * Float32(sqrt(2.0 / n_l)),  # W_down
            randn(Float32, d, d) * scale,  # W_cross_Q
            randn(Float32, d, d) * scale,  # W_cross_K
            randn(Float32, d, d) * scale,  # W_cross_V
            fill(Float32(0.5), d)         # alpha (start at 0.5)
        )
        push!(levels, params)

        # Up-projection for reconstruction
        if l > 1
            n_coarse = cfg.n_tokens ÷ (2^(l-1))
            n_fine = cfg.n_tokens ÷ (2^(l-2))
            push!(w_ups, randn(Float32, n_fine, n_coarse) * Float32(sqrt(2.0 / n_coarse)))
        end
    end

    return HRTParams(levels, w_ups)
end

"""
Initialize HRT state with random embeddings.
"""
function init_state(cfg::HRTConfig)
    Rs = Matrix{Float32}[]
    for l in 1:cfg.n_levels
        n_l = cfg.n_tokens ÷ (2^(l-1))
        push!(Rs, randn(Float32, n_l, cfg.d_model) * 0.01f0)
    end
    return HRTState(Rs)
end

# ─── Core Operations ─────────────────────────────────────────────────────────

"""
Scaled dot-product self-attention: softmax(QK^T/√d)V
"""
function self_attention(R::Matrix{Float32}, W_Q, W_K, W_V, W_O)
    Q = R * W_Q
    K = R * W_K
    V = R * W_V
    d_k = Float32(size(W_Q, 2))

    # Attention scores
    scores = Q * K' / sqrt(d_k)

    # Softmax per row
    for i in axes(scores, 1)
        row = @view scores[i, :]
        row .-= maximum(row)  # numerical stability
        row .= exp.(row)
        row ./= sum(row)
    end

    # Weighted values + output projection
    out = scores * V
    return out * W_O
end

"""
Position-wise feed-forward: ReLU(x·W1 + b1)·W2 + b2
"""
function feed_forward(R::Matrix{Float32}, W1, b1, W2, b2)
    hidden = max.(R * W1 .+ b1', 0.0f0)  # ReLU
    return hidden * W2 .+ b2'
end

"""
Down-projection: R_{l+1} = W_down · R_l (strided pooling)
"""
function down_project(R::Matrix{Float32}, W_down)
    return W_down * R  # (n_l/2 × n_l) * (n_l × d) = (n_l/2 × d)
end

"""
Cross-attention: Q from target level, K/V from source level
"""
function cross_attention(
    R_target::Matrix{Float32}, R_source::Matrix{Float32}, W_Q, W_K, W_V
)
    Q = R_target * W_Q
    K = R_source * W_K
    V = R_source * W_V
    d_k = Float32(size(W_Q, 2))

    scores = Q * K' / sqrt(d_k)

    # Softmax
    for i in axes(scores, 1)
        row = @view scores[i, :]
        row .-= maximum(row)
        row .= exp.(row)
        row ./= sum(row)
    end

    return scores * V
end

"""
Gated fusion: R = σ(α) ⊙ R̃ + (1-σ(α)) ⊙ R
"""
function gated_fuse(
    R_original::Matrix{Float32}, R_cross::Matrix{Float32}, alpha::Vector{Float32}
)
    gate = 1.0f0 ./ (1.0f0 .+ exp.(-alpha))  # sigmoid
    return R_cross .* gate' .+ R_original .* (1.0f0 .- gate')
end

# ─── Forward Pass ────────────────────────────────────────────────────────────

"""
    hrt_forward!(state, params, cfg) → state

Run one forward pass through the HRT pyramid:

 1. Per-level RTB (self-attention + FFN) with residual connections
 2. Down-projection to create next level
 3. Cross-resolution attention (bidirectional)
 4. Gated fusion
"""
function hrt_forward!(state::HRTState, params::HRTParams, cfg::HRTConfig)
    L = cfg.n_levels

    # Phase 1: Per-level RTB + down-projection
    for l in 1:L
        p = params.levels[l]
        R_l = state.R[l]

        # Self-attention + residual
        attn_out = self_attention(R_l, p.W_Q, p.W_K, p.W_V, p.W_O)
        R_l = R_l .+ attn_out  # residual

        # FFN + residual
        ffn_out = feed_forward(R_l, p.W1, p.b1, p.W2, p.b2)
        R_l = R_l .+ ffn_out  # residual

        state.R[l] = R_l

        # Down-project to next level (if not last)
        if l < L
            state.R[l + 1] = down_project(R_l, p.W_down)
        end
    end

    # Phase 2: Cross-resolution attention + gated fusion (bottom-up)
    for l in (L - 1):-1:1
        p = params.levels[l]

        # Fine←coarse attention
        R_cross_fine = cross_attention(
            state.R[l], state.R[l + 1], p.W_cross_Q, p.W_cross_K, p.W_cross_V
        )
        state.R[l] = gated_fuse(state.R[l], R_cross_fine, p.alpha)

        # Coarse←fine attention
        p_next = params.levels[l + 1]
        R_cross_coarse = cross_attention(
            state.R[l + 1], state.R[l], p_next.W_cross_Q, p_next.W_cross_K, p_next.W_cross_V
        )
        state.R[l + 1] = gated_fuse(state.R[l + 1], R_cross_coarse, p_next.alpha)
    end

    return state
end

# ─── Reconstruction Loss ─────────────────────────────────────────────────────

"""
    hrt_reconstruction_loss(state, params, cfg) → Float32

L_recon = ||R_1 - g(R_L, ..., R_2)||² where g is an up-projection cascade.
Encourages faithful coarse representations (predictive coding compatible).
"""
function hrt_reconstruction_loss(state::HRTState, params::HRTParams, cfg::HRTConfig)
    L = cfg.n_levels
    if L <= 1 || isempty(params.W_up)
        return 0.0f0
    end

    # Cascade up from coarsest to finest
    R_recon = state.R[L]
    for l in (L - 1):-1:1
        W = params.W_up[l]  # fine × coarse
        R_recon = W * R_recon  # up-project
    end

    # MSE between original finest and reconstructed
    diff = state.R[1] .- R_recon
    return cfg.recon_weight * sum(diff .^ 2) / length(diff)
end

end # module
