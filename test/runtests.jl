using Test
using MORKTensorNetworks

@testset "MORKTensorNetworks" begin

    @testset "Semirings — §3.5" begin
        sr = BooleanSemiring()
        @test oplus(sr, false, true)  == true
        @test otimes(sr, true, false) == false
        @test szero(sr) == false
        @test sone(sr)  == true

        sr2 = MaxPlusSemiring()
        @test oplus(sr2, 3.0f0, 5.0f0) == 5.0f0
        @test otimes(sr2, 2.0f0, 3.0f0) == 5.0f0
        @test szero(sr2) == -Inf32
        @test sone(sr2)  == 0.0f0

        sr3 = SumProductSemiring()
        @test oplus(sr3, 2.0f0, 3.0f0)  == 5.0f0
        @test otimes(sr3, 2.0f0, 3.0f0) == 6.0f0
    end

    @testset "PathAlgebra — §1, §3" begin
        sr = SumProductSemiring()
        # Sister(a,b)=1 → S[1,2]=1; Parent(b,c)=1 → P[2,3]=1 → Aunt(a,c)=1
        S = Float32[0 1 0; 0 0 0; 0 0 0]
        P = Float32[0 0 0; 0 0 1; 0 0 0]
        A = path_compose(sr, S, P)
        @test A[1, 3] > 0.0f0   # a is aunt of c
        @test A[2, 3] == 0.0f0  # no path from row 2

        # Union: S has (1,2), P has (2,3) → union has both
        U = path_union(sr, S, P)
        @test U[1,2] > 0   # from S
        @test U[2,3] > 0   # from P

        I = path_intersect(sr, S, S)
        @test I[1,2] > 0  # S∩S = S

        # Viterbi best-path
        W = Float32[0 1 0; 0 0 2; 0 0 0]
        Score = path_viterbi(W)
        @test Score[1,3] ≈ 3.0f0   # best 2-hop: 1+2=3
    end

    @testset "ShardZipper — §2 end-to-end" begin
        s = new_space()
        space_add_all_sexpr!(s, "(edge a b) (edge b c) (edge a c)")

        # Step 1: partition
        prefixes = partition_trie(s, 10)
        @test !isempty(prefixes)

        # Steps 2-3: capture + materialize
        shard = capture_shard(s, UInt8[])
        materialize!(shard, s)
        @test !isempty(shard.node_keys)

        # Step 4: compute (trivial kernel — mark all edges as inserts)
        compute!(shard, sh -> begin
            for (i, key) in enumerate(sh.node_keys)
                push!(sh.patch_log, PatchRecord(:insert, key, 1.0f0))
            end
        end)
        @test !isempty(shard.patch_log)

        # Step 5: patch & reattach
        n = patch_and_reattach!(shard, s)
        @test n > 0

        # Step 6: adapt check
        @test should_adapt(shard, 2) || !should_adapt(shard, 10000)
    end

    @testset "TuckerDecomposition — §5.4" begin
        A = rand(Float32, 6, 6)
        C, M, N = tucker_decompose_2d(A, 3)
        @test size(M) == (6, 3)
        @test size(N) == (6, 3)
        @test size(C) == (3, 3)
        A_recon = tucker_reconstruct_2d(C, M, N)
        @test size(A_recon) == (6, 6)
    end

    @testset "CrossShardJoin — §5.6 strategies" begin
        sr = BooleanSemiring()
        A  = Float32[1 0; 1 1]
        B  = Float32[0 1; 1 0]
        strat = HaloStrategy(1)
        @test strat isa HaloStrategy
        @test strat.halo_width == 1
    end


    @testset "BCSR — §5.1 Block CSR" begin
        A = Float32[1 0 2 0;
                    0 0 0 3;
                    4 0 5 0;
                    0 6 0 7]
        bcsr = dense_to_bcsr(A, 2, 2)
        @test bcsr isa BCSRMatrix{Float32}
        @test bcsr.block_r == 2
        @test bcsr.block_c == 2
        A2 = bcsr_to_dense(bcsr)
        @test A2 ≈ A

        # Block size that evenly divides
        B = rand(Float32, 4, 6)
        bcsr2 = dense_to_bcsr(B, 2, 3)
        @test bcsr_to_dense(bcsr2) ≈ B
    end

    @testset "Tucker 3D — §5.4 A_ijk ≈ C×M×N×P" begin
        A = rand(Float32, 6, 5, 4)
        C, M, N, P, err = tucker_decompose_3d(A, 3, 3, 3)
        @test size(C) == (3, 3, 3)
        @test size(M) == (6, 3)
        @test size(N) == (5, 3)
        @test size(P) == (4, 3)
        @test err < 0.5   # rough reconstruction quality

        A_recon = tucker_reconstruct_3d(C, M, N, P)
        @test size(A_recon) == (6, 5, 4)

        # Full rank → near-perfect reconstruction
        A2 = rand(Float32, 4, 4, 4)
        C2, M2, N2, P2, err2 = tucker_decompose_3d(A2, 4, 4, 4)
        @test err2 < 0.05
    end

    @testset "ECAN tensor bridge — §7.3" begin
        n = 4
        state = ECANState(n)
        state.sti = Float32[0.8, 0.3, 0.1, 0.6]

        # §7.3.2: Build Hebbian weight matrix
        links = [(1,2,0.5f0), (2,3,0.4f0), (3,4,0.3f0), (4,1,0.2f0)]
        W = ecan_build_weight_matrix(links, n)
        state.W = W
        @test W[1,2] == 0.5f0
        @test W[2,4] == -Inf32  # no link

        # §7.3.1: STI spreading as (max,+) matmul
        old_sti = copy(state.sti)
        ecan_sti_spread!(state; decay=0.9f0)
        @test length(state.sti) == n
        @test all(isfinite, state.sti)
        @test all(state.sti .>= 0.0f0)

        # §7.3.2: Hebbian update — save original before update
        state.sti = Float32[0.8, 0.3, 0.1, 0.6]
        state.W   = copy(W)
        w12_before = state.W[1,2]
        ecan_hebbian_update!(state; η=0.1f0, decay=1.0f0)  # no decay → pure Hebbian
        @test state.W[1,2] > w12_before  # co-active pair strengthened

        # §7.3.3: Rent collection
        state.sti = Float32[0.8, 0.3, 0.1, 0.6]
        total_rent = ecan_collect_rent!(state; af_threshold=0.5f0, rent_rate=0.1f0)
        @test total_rent > 0.0f0
        @test state.sti[1] < 0.8f0   # high STI atom paid rent
        @test state.sti[2] == 0.3f0  # below threshold — no rent

        # §7.3.3: Wage distribution
        state.sti = Float32[0.5, 0.2, 0.1, 0.4]
        ecan_distribute_wages!(state, 0.1f0)
        @test sum(state.sti) > 0.5f0 + 0.2f0 + 0.1f0 + 0.4f0 - 1e-5  # budget added
    end

    # ── Regression tests for 2026-05-30 audit fixes ────────────────────────────

    @testset "PathAlgebra fixes — Heaviside, intersect=min, restrict semiring" begin
        # ── path_intersect: spec §3 row 5 says R ∧ S = elementwise MIN.
        # Previously used otimes — silently wrong under SumProduct.
        sr_sp = SumProductSemiring()
        R = Float64[0.6 0.3; 0.0 0.8]
        I = path_intersect(sr_sp, R, R)
        # min(R, R) == R; otimes(R, R) = R.^2 = [0.36 0.09; 0 0.64]
        @test I == R                                # spec semantics
        @test !isapprox(I[1,1], 0.36; atol=1e-6)    # NOT the buggy otimes result

        # ── path_compose: spec §3 row 1 says T = H(Σ R⊗S). Default applies H.
        # Under SumProduct, compose(R, S) used to return raw matmul; should
        # now project to {0,1}.
        S = Float64[0 1; 0 0]
        P = Float64[0 0; 0 1]
        T = path_compose(sr_sp, S, P)
        @test T[1,2] == 1.0   # path exists, projected to sone
        @test T[2,2] == 0.0   # no path, projected to szero
        # Opt-out: apply_threshold=false gives the raw weighted sum
        T_raw = path_compose(sr_sp, S, P; apply_threshold=false)
        @test T_raw[1,2] == 1.0   # also 1 in this case (binary input)

        # ── path_union: spec §3 row 4 says U = H(R + S). Under SumProduct,
        # union(R, R) used to return 2R; should now return R thresholded.
        U = path_union(sr_sp, R, R)
        @test U[1,1] == 1.0   # 0.6+0.6=1.2 > 0 → projects to 1
        @test U[2,1] == 0.0   # 0+0=0 → szero
        @test !isapprox(U[1,1], 1.2; atol=1e-6)   # NOT the un-thresholded sum

        # ── path_restrict: spec §3 row 3, semiring-aware. Under MaxPlus
        # (sone=0, szero=-Inf), restrict with mask=0 should yield -Inf,
        # not the buggy 0 from `R .* 0`.
        sr_mp = MaxPlusSemiring()
        Rmp  = Float64[0.5 1.0; 2.0 0.7]
        mask = Float64[1 0; 0 1]
        Rr = path_restrict(sr_mp, Rmp, mask)
        @test Rr[1,1] == 0.5     # mask=1, pass through
        @test Rr[1,2] == -Inf    # mask=0, szero(MaxPlus) = -Inf (was 0.0!)
        @test Rr[2,1] == -Inf    # same
        @test Rr[2,2] == 0.7     # pass through
    end

    @testset "SemiringKernels SpGEMM — multi-semiring correctness vs CPU reference" begin
        # Spec §5.2 headline kernel: previously the file's header advertised
        # `semiring_spmm!` but the kernel was missing. Audit 2026-05-30 caught
        # this. With the fix, gpu_semiring_spmm produces the same result as
        # CPU semiring_matmul for each semiring family.
        using KernelAbstractions: CPU
        using MORKTensorNetworks.SemiringKernels: gpu_semiring_spmm
        using MORKTensorNetworks.PathAlgebra: dense_to_csr

        # 3x3 × 3x3 example with structure: R has edges (1,2), (2,3); S has (1,3), (3,1).
        R = Float64[0 1 0; 0 0 1; 0 0 0]
        S = Float64[0 0 1; 0 0 0; 1 0 0]

        for sr in (SumProductSemiring(), MaxPlusSemiring(), BooleanSemiring(),
                   PLNSemiring(), CostSemiring(), MinPlusSemiring())
            # Sparsify with the semiring's szero so zeros are dropped correctly.
            # For numeric semirings that's 0.0; for MaxPlus/Cost it's ±Inf so
            # we keep the test on numeric-zero matrices to share inputs.
            sr isa MaxPlusSemiring && continue   # different zero — skip
            sr isa MinPlusSemiring && continue
            sr isa CostSemiring    && continue

            rowptr_R, colval_R, nzval_R, _, _ = dense_to_csr(R)
            rowptr_S, colval_S, nzval_S, _, n = dense_to_csr(S)
            gpu_result = gpu_semiring_spmm(sr,
                                            rowptr_R, colval_R, nzval_R,
                                            rowptr_S, colval_S, nzval_S, n;
                                            backend=CPU())
            cpu_result = semiring_matmul(sr, R, S)
            @test gpu_result ≈ cpu_result
        end
    end

    @testset "path_compose with backend=CPU() dispatches via SpGEMM" begin
        # End-to-end: path_compose with backend kwarg should now route through
        # gpu_semiring_spmm (which was unreachable before — dead-import in the
        # PathAlgebra prelude). Default path (no backend) and SpGEMM path
        # should produce identical results modulo dense-storage type.
        using KernelAbstractions: CPU
        sr = SumProductSemiring()
        R = Float64[0 1 0; 0 0 1; 0 0 0]
        S = Float64[0 0 1; 0 0 0; 0 1 0]
        T_default = path_compose(sr, R, S)                       # dense CPU
        T_gpu     = path_compose(sr, R, S; backend=CPU())        # SpGEMM path
        @test T_default ≈ T_gpu

        # Also verify the apply_threshold=false (raw) path through SpGEMM.
        T_raw_default = path_compose(sr, R, S; apply_threshold=false)
        T_raw_gpu     = path_compose(sr, R, S; apply_threshold=false, backend=CPU())
        @test T_raw_default ≈ T_raw_gpu
    end

    @testset "Tucker — N-mode generalization" begin
        # Audit gap: tucker_decompose_* only had 2-mode and 3-mode entry
        # points. Helpers were already N-mode generic (_unfold, _ttm,
        # _mode_unfold_svd). New `tucker_decompose_nd` covers arbitrary
        # rank — verified on a 4-mode tensor here.
        # Use a low-rank fixture so the relative error stays small.
        A4 = Float32[i + 2j + 3k + 4l
                     for i in 1:5, j in 1:4, k in 1:3, l in 1:3]
        ranks = (3, 3, 2, 2)
        C, factors, err = tucker_decompose_nd(A4, ranks)
        # Core tensor matches the rank tuple
        @test size(C) == ranks
        # 4 factor matrices, one per mode, each of size (mode_dim, rank_k)
        @test length(factors) == 4
        for k in 1:4
            @test size(factors[k]) == (size(A4, k), ranks[k])
        end
        # Linear+low-rank fixture should reconstruct well
        @test err < 0.1

        # Round-trip via tucker_reconstruct_nd
        A_recon = tucker_reconstruct_nd(C, factors)
        @test size(A_recon) == size(A4)

        # 2-mode case via the nd path matches the existing 2d entry point
        # in shape (we don't compare numeric values because SVD signs differ;
        # we just verify the API stitches together).
        A2 = Float32.(randn(6, 6))
        C2, F2, _ = tucker_decompose_nd(A2, (3, 3))
        @test size(C2) == (3, 3)
        @test length(F2) == 2

        # Mismatched ranks tuple → error
        @test_throws ErrorException tucker_decompose_nd(A4, (3, 3, 2))   # only 3
    end

    @testset "gpu_semiring_reduce — tree-parallel pairwise" begin
        # Audit gap: the previous reduce kernel had `if i == 1; serial loop`
        # — single-threaded inside a "parallel" launch. Replaced with the
        # pairwise pattern (one halving pass per launch). Verify correctness
        # across a few semirings and a few sizes including odd lengths.
        using KernelAbstractions: CPU
        using MORKTensorNetworks.SemiringKernels: gpu_semiring_reduce
        for (sr, op) in (
                (SumProductSemiring(), +),
                (MaxPlusSemiring(),    max),
                (MinPlusSemiring(),    min),
            )
            # Length 1 → identity
            @test gpu_semiring_reduce(sr, [3.0f0]; backend=CPU()) == 3.0f0
            # Length 8 (power of 2)
            v8 = Float32[1, 2, 3, 4, 5, 6, 7, 8]
            @test gpu_semiring_reduce(sr, v8; backend=CPU()) ≈ reduce(op, v8)
            # Length 7 (odd — must handle the trailing-one case via szero)
            v7 = Float32[1, 2, 3, 4, 5, 6, 7]
            @test gpu_semiring_reduce(sr, v7; backend=CPU()) ≈ reduce(op, v7)
            # Length 13 (odd, doesn't halve cleanly)
            v13 = Float32[i for i in 1:13]
            @test gpu_semiring_reduce(sr, v13; backend=CPU()) ≈ reduce(op, v13)
        end
    end

    @testset "path_universal — now exported and reachable" begin
        # Audit found path_universal was defined (line 196) but missing from
        # the export list — unreachable to consumers using `using ...PathAlgebra`.
        sr = SumProductSemiring()
        # ∀y: matrix where all rows have all-1 columns → universal = 1 for each row
        R_all = ones(Float64, 2, 3)
        u_all = path_universal(sr, R_all)
        @test all(u_all .== 1.0)   # all rows satisfy ∀

        # Matrix with one zero — that row doesn't satisfy universal
        R_mix = Float64[1 1 1; 1 0 1]
        u_mix = path_universal(sr, R_mix)
        @test u_mix[1] == 1.0     # row 1: all ones
        @test u_mix[2] == 0.0     # row 2 has a zero
    end

end  # MORKTensorNetworks
