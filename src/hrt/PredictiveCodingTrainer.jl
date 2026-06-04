"""
    PredictiveCodingTrainer.jl — Forward-only training for HRT via Predictive Coding

MORK-Tensor-Networks paper §5.5: local, asynchronous weight updates using
predictive coding instead of backpropagation. Per Goertzel's recommendation:
forward propagation only, no global gradients.

Algorithm (per level l):

 1. Predict: R̂_l from current weights
 2. Error:  e_l = Y_l - R̂_l  (prediction error)
 3. Update: R_l += lr * e_l  (activity update, few inner steps)
 4. Learn:  W += η * e_l * R_l^T  (Hebbian weight update)

Properties:

  - No backpropagation — all updates are local
  - No cross-shard synchronization needed
  - All primitives are matrix multiply (einsum) and axpy — GPU-friendly
  - Compatible with ShardZipper: patch logging unchanged
  - Aligns with free energy minimization / active inference

Reference: Goertzel "World Modeling in Hyperon for PRIMUS" v2 §7.7
"""
module PredictiveCodingTrainer

using LinearAlgebra
using ..HRT: HRTConfig, HRTState, HRTParams, HRTLevelParams, self_attention, feed_forward,
    down_project

export PCTrainerConfig, pc_train_step!, pc_inner_loop!, hebbian_update!

# ─── Configuration ───────────────────────────────────────────────────────────

struct PCTrainerConfig
    lr_activity::Float32      # Learning rate for activity updates (inner loop)
    lr_weights::Float32       # Learning rate for Hebbian weight updates
    inner_steps::Int          # Number of inner-loop activity refinement steps
    surprise_threshold::Float32  # Only update if prediction error > threshold
    precision_weight::Float32 # Inverse variance weighting (higher = trust predictions more)
end

function PCTrainerConfig(;
    lr_activity=0.01f0, lr_weights=0.001f0, inner_steps=3, surprise_threshold=0.1f0,
    precision_weight=1.0f0
)
    PCTrainerConfig(
        lr_activity, lr_weights, inner_steps, surprise_threshold, precision_weight
    )
end

# ─── Prediction Error ────────────────────────────────────────────────────────

"""
    prediction_error(predicted, observed) → error, surprise

Compute prediction error and scalar surprise (MSE).
"""
function prediction_error(predicted::Matrix{Float32}, observed::Matrix{Float32})
    # M4 fix (audit 2026-06-04): silent truncation (min on both dims) was masking
    # shape bugs — a pyramid level size mismatch (M5) silently produced plausible-
    # looking numerics on a truncated submatrix. Replace with assertion so shape
    # errors surface at the call site rather than propagating silently.
    @assert size(predicted) == size(observed) "prediction_error: shape mismatch — predicted $(size(predicted)) vs observed $(size(observed)). Check HRT pyramid level sizing (n_tokens must be a power of 2 ≥ 2^(n_levels-1))."
    err = observed .- predicted
    surprise = sum(err .^ 2) / length(err)
    return err, surprise
end

# ─── Inner Loop: Activity Update ─────────────────────────────────────────────

"""
    pc_inner_loop!(state, params, cfg, pc_cfg, observations)

Inner loop: refine activities R_l to minimize prediction error.
Forward-only: each level predicts the next, error drives activity updates.

`observations` is a vector of target matrices (one per level), or nothing
for unsupervised (self-prediction from coarser levels).
"""
function pc_inner_loop!(
    state::HRTState, params::HRTParams, cfg::HRTConfig, pc_cfg::PCTrainerConfig,
    observations=nothing
)
    L = cfg.n_levels
    total_surprise = 0.0f0

    for step in 1:pc_cfg.inner_steps
        for l in 1:(L - 1)
            p = params.levels[l]

            # Predict level l+1 from level l (top-down prediction)
            predicted = down_project(state.R[l], p.W_down)

            # Observation: either external target or current state
            observed =
                if observations !== nothing && l < length(observations) &&
                    observations[l + 1] !== nothing
                    observations[l + 1]
                else
                    state.R[l + 1]
                end

            # Compute error
            error, surprise = prediction_error(predicted, observed)
            total_surprise += surprise

            # Only update if surprising enough
            if surprise > pc_cfg.surprise_threshold
                # Activity update: R_{l+1} += lr * precision * error.
                # M4 (audit 2026-06-04): prediction_error already asserted
                # error matches the predicted/observed shape, so a direct `.+=`
                # is correct and surfaces any mismatch (no silent min-truncation).
                state.R[l + 1] .+= pc_cfg.lr_activity * pc_cfg.precision_weight .* error
            end
        end
    end

    return total_surprise
end

# ─── Hebbian Weight Update ───────────────────────────────────────────────────

"""
    hebbian_update!(params, state, cfg, pc_cfg) → total_delta

Local Hebbian weight updates based on cross-level prediction errors.
Forward-only: W_down += η · error · R_l^T (outer-product learning rule).

ACTUALLY updates:
  - W_down (down-projection): correlates the cross-level prediction error with the
    finer-level activity → learns better compression.
  - alpha (fusion gate): nudged up when surprise is high.

N4 GAP (audit 2026-06-04): §6.4 calls for "local Hebbian for Q/K/V and projection
matrices", and this docstring previously claimed W_Q/W_K/W_V/W1/W2 were updated — they
are NOT. The within-level attention/FFN weights have no cross-level prediction-error
signal in this forward-prediction PC setup, and the paper (§6.4, "rough notes") does
not specify the local rule for them. Implementing attention/FFN local learning needs a
defined local error signal — tracked in docs/TODO.md, NOT fabricated here. As written,
predictive-coding training learns only the down-projection + fusion gates.
"""
function hebbian_update!(
    params::HRTParams, state::HRTState, cfg::HRTConfig, pc_cfg::PCTrainerConfig
)
    L = cfg.n_levels
    total_delta = 0.0f0
    η = pc_cfg.lr_weights

    for l in 1:(L - 1)
        p = params.levels[l]

        # Prediction error for this level
        predicted = down_project(state.R[l], p.W_down)
        observed = state.R[l + 1]
        error, surprise = prediction_error(predicted, observed)

        if surprise <= pc_cfg.surprise_threshold
            continue
        end

        # --- Update W_down (down-projection): ΔW_down = η · error · R_l^T ---
        # error is (n_{l+1} × d), R_l is (n_l × d) → outer product (n_{l+1} × n_l),
        # which equals W_down's shape under correct pyramid sizing (M5).
        # M4 (audit 2026-06-04): replaced silent min-truncation with a shape assert
        # so a sizing bug throws instead of being masked by a truncated update.
        delta_W = η .* (error * state.R[l]')
        @assert size(delta_W) == size(p.W_down) "hebbian_update!: ΔW_down $(size(delta_W)) ≠ W_down $(size(p.W_down)) — check HRT pyramid sizing."
        p.W_down .+= delta_W
        total_delta += sum(abs.(delta_W))

        # --- Update alpha (gated fusion) ---
        # Increase alpha (use more cross-attention) when surprise is high
        # Decrease alpha when predictions are accurate
        alpha_delta = Float32(η * (surprise - pc_cfg.surprise_threshold))
        p.alpha .+= alpha_delta
        clamp!(p.alpha, -3.0f0, 3.0f0)  # Keep in sigmoid range
        total_delta += abs(alpha_delta) * length(p.alpha)
    end

    return total_delta
end

# ─── Full Training Step ──────────────────────────────────────────────────────

"""
    pc_train_step!(state, params, cfg, pc_cfg; observations=nothing) → (surprise, delta)

One complete predictive coding training step:

 1. Inner loop: refine activities (forward prediction error minimization)
 2. Hebbian update: adjust weights based on remaining prediction errors

Returns total surprise and total weight delta.
"""
function pc_train_step!(
    state::HRTState, params::HRTParams, cfg::HRTConfig, pc_cfg::PCTrainerConfig;
    observations=nothing
)
    # 1. Inner loop: activity refinement
    surprise = pc_inner_loop!(state, params, cfg, pc_cfg, observations)

    # 2. Hebbian weight update
    delta = hebbian_update!(params, state, cfg, pc_cfg)

    return (surprise, delta)
end

end # module
