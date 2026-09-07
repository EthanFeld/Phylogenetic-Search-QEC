import json

import numpy as np

from distance_sketch import (
    fast_refute_supports,
    gf2_logical_basis_packed,
)
from inverse_design_core import (
    TargetSpec,
    _combine_css_estimates,
    _systematic_probe_from_generator,
    _mutate_pairs,
    cyclic_group,
    infer_product_shapes,
    logicals_from_systematic_generators,
    seed_check,
    semidirect_cyclic_group,
    systematic_seed_generator,
)
from search_efficiency import fast_refute_submission
from phylogenetic_search import compact_history, select_phylogenetic_elites, upgma
from phylogenetic_report import _literature_tree
from pareto_campaign import ml_group_indices
from concatenated_codes import (css_block_concatenate, css_concatenate,
                                css_heterogeneous_block_concatenate,
                                even_parity_css_code, four_two_two_code,
                                repetition3_code, shor9_code, steane_code,
                                validate_css_code)
from concatenation_campaign import run as run_concatenation
from generalized_bicycle import (REFERENCE_A, REFERENCE_B, REFERENCE_M,
                                  build_matrices, common_gcd_degree, mine_low_words)
from gb_collapse_guard import _update_blacklist, classify_rows, select_guard_beam
from gb_divisor_escape import (factor_catalog, sample_divisors,
                               structured_bivariate_divisors)
from gf2_factor import degree as gf2_degree, factor_xm_plus_one, mul as gf2_mul
from hyper_validator import (block_swap_reverse_support, refine_candidate,
                             structural_validate)


def test_native_batch_systematic_matches_python_when_built():
    import cpp_fast
    if not cpp_fast.available():
        return
    group = semidirect_cyclic_group(11, 4, 10)
    pairs = np.asarray([
        ((0, 1, 2), (0, 3, 4)),
        ((4, 8, 10), (1, 2, 9)),
    ], dtype=np.int32)
    generators, probes, valid = cpp_fast.batch_systematic(
        group.mult, group.inv, pairs, left=True)
    for i, pair in enumerate(pairs):
        expected = systematic_seed_generator(tuple(map(tuple, pair)), group, True)
        assert (expected is not None) == bool(valid[i])
        if expected is not None:
            assert np.array_equal(expected, generators[i])
            assert probes[i] == _systematic_probe_from_generator(expected)["best_weight"]


def test_native_accelerator_status_is_auditable():
    import cpp_fast
    info = cpp_fast.accelerator_info()
    assert info["backend"] == "qldpc_fast"
    assert "css_ris" in info["kernels"]
    if info["available"]:
        assert info["library"]
        assert info["load_error"] is None


def test_native_gf2_rank_matches_python_when_built():
    import cpp_fast
    from inverse_design_core import gf2_rank
    if not cpp_fast.available():
        return
    matrix = np.random.default_rng(20260910).integers(
        0, 2, size=(73, 119), dtype=np.uint8)
    assert cpp_fast.gf2_rank(matrix) == gf2_rank(matrix)


def test_even_xm_plus_one_factorization_preserves_multiplicity():
    factors = factor_xm_plus_one(350)
    product = 1
    for factor in factors:
        product = gf2_mul(product, factor)
    assert product == (1 << 350) | 1
    assert len(factors) == 18
    assert sum(gf2_degree(factor) for factor in factors) == 350


def test_m348_repeated_root_has_twenty_distinct_q94_ideals():
    import gb_m348_k188_repeated_campaign  # configures shared campaign module
    import gb_m350_k188_repeated_campaign as campaign
    vectors = campaign.degree94_exponents()
    assert len(vectors) == 20
    assert len(set(vectors)) == 20
    assert all(sum(exponent * gf2_degree(factor)
                   for exponent, factor in zip(vector, campaign.BASE_FACTORS)) == 94
               for vector in vectors)


def test_gb_pair_key_quotients_independent_block_shifts():
    import gb_m345_k188_phylo_campaign as campaign
    a = (0, 4, 19, 41)
    b = (0, 7, 13, 52)
    baseline = campaign._pair_key(a, b)
    assert campaign._pair_key(campaign._shift(a, 23), b) == baseline
    assert campaign._pair_key(a, campaign._shift(b, 117)) == baseline
    assert campaign._pair_key(campaign._shift(b, 9),
                              campaign._shift(a, 31)) == baseline
    affine = campaign._affine_pair_key(campaign.M, a, b)
    multiplied_a = tuple((17 * q) % campaign.M for q in a)
    multiplied_b = tuple((17 * q) % campaign.M for q in b)
    assert campaign._affine_pair_key(campaign.M, multiplied_a, multiplied_b) == affine


def test_gb_phase_equivalence_transfers_logical_witness():
    from gb_phase_equivalence import equivalence_certificate
    from generalized_bicycle import build_supports
    m = 7
    a, b = (0, 1), (0, 2)
    shifted_b = tuple(sorted((q + 4) % m for q in b))
    hx0, hz0 = build_supports(m, a, b)
    hx1, hz1 = build_supports(m, a, shifted_b)
    # This small code has the single-qubit Z logical [0].
    def doc(hx, hz):
        return {"n": 2 * m, "k": 0, "checks": {"X": hx, "Z": hz},
                "distance": {"d": 1,
                             "X": {"value": 1, "witness": [0]},
                             "Z": {"value": 1, "witness": [0]}}}
    # Use a real logical from a rank-deficient pair instead of assuming [0].
    import cpp_fast
    base = doc(hx0, hz0)
    target = doc(hx1, hz1)
    base["k"] = structural_validate(base)["computed_k"]
    target["k"] = structural_validate(target)["computed_k"]
    result = cpp_fast.css_ris(*build_matrices(m, a, b), trials=32, seed=9)
    witness = result["z"]["witness"]
    base["distance"] = {"d": len(witness),
                        "X": {"value": len(witness), "witness": witness},
                        "Z": {"value": len(witness), "witness": witness}}
    target["distance"] = base["distance"]
    # Certificate validates structure before transferring, so construct valid
    # independent witnesses for both documents.
    for item, matrices in ((base, build_matrices(m, a, b)),
                           (target, build_matrices(m, a, shifted_b))):
        ris = cpp_fast.css_ris(*matrices, trials=32, seed=11)
        wx, wz = ris["x"]["witness"], ris["z"]["witness"]
        item["distance"] = {"d": min(len(wx), len(wz)),
                            "X": {"value": len(wx), "witness": wx},
                            "Z": {"value": len(wz), "witness": wz}}
    certificate = equivalence_certificate(base, target,
                                          base["distance"]["Z"]["witness"], "Z")
    assert certificate["ok"]
    assert certificate["independent_block_shifts"] == [0, 4]


def test_cached_systematic_logicals_match_canonical_construction():
    group = semidirect_cyclic_group(11, 4, 10)
    A = ((6, 26, 33), (0, 25, 26))
    B = ((27, 34, 36), (0, 15, 30))
    ga = systematic_seed_generator(A, group, left=True)
    gb = systematic_seed_generator(B, group, left=False)
    lx0, lz0 = logicals_from_systematic_generators(ga, gb, group.order)
    lx1, lz1 = __import__("inverse_design_core").canonical_1x2_logicals(A, B, group)
    assert np.array_equal(lx0, lx1)
    assert np.array_equal(lz0, lz1)


def test_elite_mutations_preserve_support_weight_and_anchor():
    group = semidirect_cyclic_group(11, 4, 10)
    pair = ((6, 26, 33), (0, 25, 26))
    mutants = _mutate_pairs([pair], group, __import__("random").Random(7), 24)
    assert mutants
    assert all(len(a) == 3 and len(b) == 3 for a, b in mutants)
    assert all(len(set(a)) == 3 and len(set(b)) == 3 for a, b in mutants)
    assert all(0 in b for _, b in mutants)
    assert len(mutants) == len(set(mutants))


def test_radius_two_mutations_preserve_support_weight_and_anchor():
    group = semidirect_cyclic_group(11, 4, 10)
    pair = ((6, 26, 33), (0, 25, 26))
    mutants = _mutate_pairs([pair], group, __import__("random").Random(7), 8,
                            radius=2)
    assert mutants
    assert all(len(a) == 3 and len(b) == 3 for a, b in mutants)
    assert all(0 in b for _, b in mutants)
    assert len(mutants) == len(set(mutants))


def test_deep_repeat_combiner_keeps_best_sector_witnesses():
    estimates = [
        {"d_upper": 22, "dx_upper": 22, "dz_upper": 22,
         "x": {"best_weight": 22}, "z": {"best_weight": 22}},
        {"d_upper": 20, "dx_upper": 20, "dz_upper": 21,
         "x": {"best_weight": 20}, "z": {"best_weight": 21}},
        {"d_upper": 21, "dx_upper": 21, "dz_upper": 19,
         "x": {"best_weight": 21}, "z": {"best_weight": 19}},
    ]
    combined = _combine_css_estimates(estimates)
    assert combined["dx_upper"] == 20
    assert combined["dz_upper"] == 19
    assert combined["d_upper"] == 19
    assert combined["repeat_count"] == 3


def test_native_exact_tanner_finds_boundary_witness_when_built():
    import cpp_fast
    if not cpp_fast.available():
        return
    H = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    D = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    result = cpp_fast.exact_logical(H, D, 2)
    assert result["status"] == "found"
    assert result["witness_weight"] == 2


def test_systematic_seed_generator_is_kernel_basis():
    group = cyclic_group(4)
    pair = ((0, 1, 2), (0,))
    generator = systematic_seed_generator(pair, group)
    check = seed_check(pair, group)
    assert generator is not None
    assert check is not None
    assert np.array_equal((check @ generator.T) & 1, np.zeros((4, 4), dtype=np.uint8))


def test_odd_weight_shapes_cover_max_check_21_branch():
    shapes = infer_product_shapes(TargetSpec(350, 70, 21))
    assert any(shape.entry_weight == 7 and shape.carrier_order == 70 for shape in shapes)


def test_odd_weight_shapes_cover_max_check_27_branch():
    shapes = infer_product_shapes(TargetSpec(350, 70, 27))
    assert any(shape.entry_weight == 9 and shape.carrier_order == 70 for shape in shapes)


def test_over_cap_check_weight_is_rejected():
    shapes = infer_product_shapes(TargetSpec(350, 70, 33))
    assert shapes == []


def test_packed_logical_basis_has_expected_dimension():
    h = np.ones((1, 4), dtype=np.uint8)
    basis = gf2_logical_basis_packed(h, h)
    # dim ker(H) - dim row(H) = 3 - 1.
    assert len(basis) == 2


def test_fast_refuter_finds_lighter_logical():
    checks = [[0, 1, 2, 3]]
    doc = {"n": 4, "checks": {"X": checks, "Z": checks}, "distance": {"d": 3}}
    result = fast_refute_submission(doc, seed=17, trials=8, max_seconds=None)
    assert result["refuted"]
    assert result["d_found"] == 2
    assert len(result["witness"]) == 2


def test_fast_refuter_support_api_matches_submission_api():
    checks = [[0, 1, 2, 3]]
    result = fast_refute_supports(checks, checks, 4, 3, seed=17, trials=8,
                                  max_seconds=None)
    assert result["refuted"]
    assert result["d_found"] == 2


def test_upgma_tree_and_phylogenetic_elites_are_deterministic():
    rows = [
        {"semantic_hash": "a", "group_order": 11, "group_index": 0,
         "A": [[0, 1], [0, 2]], "B": [[0, 3], [0, 4]],
         "screen": {"d_upper": 12}},
        {"semantic_hash": "b", "group_order": 11, "group_index": 0,
         "A": [[0, 1], [0, 5]], "B": [[0, 3], [0, 4]],
         "screen": {"d_upper": 11}},
        {"semantic_hash": "c", "group_order": 11, "group_index": 0,
         "A": [[0, 6], [0, 7]], "B": [[0, 8], [0, 9]],
         "screen": {"d_upper": 10}},
    ]
    tree = upgma(rows)
    assert tree["algorithm"] == "UPGMA"
    assert tree["leaf_count"] == 3
    assert tree["newick"].endswith(";")
    chosen, summary = select_phylogenetic_elites(rows, 2)
    assert [r["semantic_hash"] for r in chosen] == ["a", "c"]
    assert summary["algorithm"] == "UPGMA-stratified-direct-PD"
    assert summary["selected_pairwise_distance"]["max"] > 0
    many = [dict(rows[i % len(rows)], semantic_hash=f"{i:064x}") for i in range(80)]
    chosen, _ = select_phylogenetic_elites(many, 10)
    assert len(chosen) == 10


def test_elites_keep_deep_rank_when_selecting_confirmation_pool():
    rows = [
        {"semantic_hash": "cheap-only", "group_order": 11, "group_index": 0,
         "A": [[0, 1], [0, 2]], "B": [[0, 3], [0, 4]],
         "screen": {"d_upper": 30}, "distance_estimate": {"d_upper": 12}},
        {"semantic_hash": "deep-best", "group_order": 11, "group_index": 0,
         "A": [[0, 5], [0, 6]], "B": [[0, 7], [0, 8]],
         "screen": {"d_upper": 20}, "distance_estimate": {"d_upper": 20}},
        {"semantic_hash": "deep-next", "group_order": 11, "group_index": 0,
         "A": [[0, 9], [0, 10]], "B": [[0, 1], [0, 3]],
         "screen": {"d_upper": 19}, "distance_estimate": {"d_upper": 19}},
    ]
    chosen, summary = select_phylogenetic_elites(rows, 2)
    assert [row["semantic_hash"] for row in chosen] == ["deep-best", "deep-next"]
    assert summary["selection_distance"] == "direct_genome_distance"


def test_elites_reserve_half_of_large_budget_for_quality():
    rows = [
        {"semantic_hash": label, "group_order": 11, "group_index": 0,
         "A": [[0, first], [0, 1]], "B": [[0, 2], [0, 3]],
         "screen": {"d_upper": score}}
        for label, first, score in (("a", 4, 20), ("b", 5, 19),
                                    ("c", 6, 18), ("d", 7, 17))
    ]
    chosen, summary = select_phylogenetic_elites(rows, 4)
    assert [row["semantic_hash"] for row in chosen[:2]] == ["a", "b"]
    assert summary["exploitation_slots"] == 2
    assert summary["exploration_slots"] == 2


def test_compact_history_keeps_same_target_unique_genomes():
    rows = [{"n": 300, "k": 60, "group_order": 11, "group_index": 0,
             "A": [[0, 1], [0, 2]], "B": [[0, 3], [0, 4]],
             "d_upper": 12, "semantic_hash": "a"}]
    assert len(compact_history(rows + rows, 300, 60)) == 1
    assert compact_history(rows, 350, 70) == []
    rejected = dict(rows[0], regulation={"official_gate_status": "refuted"})
    assert compact_history([rejected], 300, 60) == []


def test_literature_taxonomy_keeps_reference_nodes_out_of_score_tree():
    catalog = {
        "taxonomy": [{"id": "qldpc", "parent": None, "kind": "root"}],
        "families": [{"id": "lifted-product", "parent": "qldpc",
                       "label": "lifted-product", "repo_fit": "near"}],
    }
    tree = _literature_tree(catalog)
    assert tree["root"] == "qldpc"
    assert {edge["child"] for edge in tree["edges"]} == {"lifted-product"}
    assert tree["policy"].startswith("literature nodes are reference-only")


def test_ml_branch_selector_is_reproducible(tmp_path):
    plan = {
        "recommended_missing_branches": [
            {"n": 300, "k": 60, "max_check_weight": 27,
             "group_index": 3, "priority_score": 22.3},
            {"n": 300, "k": 60, "max_check_weight": 27,
             "group_index": 1, "priority_score": 22.2},
            {"n": 350, "k": 70, "max_check_weight": 27,
             "group_index": 9, "priority_score": 22.1},
        ]
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert ml_group_indices(path, ((300, 60),), 27, limit=2) == [3, 1]


def test_serial_concatenation_preserves_css_k_and_commutation():
    code = css_concatenate(repetition3_code(), steane_code())
    specs = validate_css_code(code)
    assert specs["n"] == 21
    assert specs["k"] == 1
    assert specs["commutation_violations"] == 0
    assert specs["max_check_weight"] == 6


def test_multilogical_block_concatenation_preserves_css_k_and_commutation():
    inner = four_two_two_code()
    code = css_block_concatenate(inner, inner)
    specs = validate_css_code(code)
    assert specs["n"] == 8
    assert specs["k"] == 2
    assert specs["commutation_violations"] == 0
    assert specs["max_check_weight"] == 4


def test_heterogeneous_even_parity_blocks_preserve_css_k_and_commutation():
    outer = css_block_concatenate(four_two_two_code(), four_two_two_code())
    code = css_heterogeneous_block_concatenate(
        outer, [even_parity_css_code(2)] * 4)
    specs = validate_css_code(code)
    assert specs["n"] == 16
    assert specs["k"] == 2
    assert specs["commutation_violations"] == 0
    assert specs["max_check_weight"] == 8


def test_inner_catalog_shor_is_valid_css_code():
    specs = validate_css_code(shor9_code())
    assert specs["n"] == 9
    assert specs["k"] == 1
    assert specs["min_check_weight"] == 2


def test_concatenation_campaign_enforces_blocklength_cap(tmp_path):
    outer_path = tmp_path / "outer.npz"
    outer = repetition3_code()
    np.savez_compressed(outer_path, hx=outer.hx, hz=outer.hz,
                        lx=outer.lx, lz=outer.lz)
    try:
        run_concatenation(outer_path, inner_name="steane7", trials=1,
                          seed=1, backend="cpp", max_n=20)
    except ValueError as exc:
        assert "exceeds challenge cap" in str(exc)
    else:
        raise AssertionError("out-of-cap concatenation was not rejected")


def test_public_generalized_bicycle_seed_has_expected_parameters():
    assert common_gcd_degree(REFERENCE_M, REFERENCE_A, REFERENCE_B) == 91
    hx, hz = build_matrices(REFERENCE_M, REFERENCE_A, REFERENCE_B)
    assert hx.shape == (341, 682)
    assert hz.shape == (341, 682)
    assert not np.any((hx @ hz.T) & 1)


def test_gf2_factorization_recovers_z341_factor_degrees():
    factors = factor_xm_plus_one(341)
    assert len(factors) == 38
    assert sorted(gf2_degree(item) for item in factors) == [1] + [5] * 6 + [10] * 31


def test_divisor_escape_samples_exact_degree_and_excludes_known_branch():
    catalog = factor_catalog()
    branches = sample_divisors(catalog, count=4, seed=19, exclude=0)
    assert len(branches) == 4
    assert all(item["degree"] == 91 for item in branches)
    assert all(item["factor_count"] >= 1 for item in branches)


def test_tree_breakout_adds_sparse_crt_torus_divisor_branches():
    branches = structured_bivariate_divisors(factor_catalog())
    assert len(branches) == 6
    assert all(item["degree"] == 91 for item in branches)
    assert all(item["coordinate_model"] == "C11_x_C31 via CRT"
               for item in branches)
    assert sorted({item["polynomial_weight"] for item in branches}) == [8, 12]
    assert all(len(item["support"]) <= 12 for item in branches)


def test_generalized_bicycle_word_miner_keeps_low_weight_words():
    words = mine_low_words(REFERENCE_M, [REFERENCE_A, REFERENCE_B],
                           iterations=8, seed=7, max_words=32)
    assert words
    assert all(10 <= len(word) <= 18 for word in words)


def test_native_cyclic_ideal_miner_accepts_ragged_bases_when_built():
    import cpp_fast
    if not cpp_fast.available():
        return
    words = cpp_fast.cyclic_ideal_mine(
        [(0, 1, 2), (0, 3)], m=7, iterations=8,
        min_weight=1, max_weight=5, seed=23, max_words=64)
    assert words
    assert all(1 <= len(word) <= 5 for word in words)
    assert all(all(0 <= value < 7 for value in word) for word in words)


def test_native_classical_ris_can_require_outside_subcode_detector():
    import cpp_fast
    if not cpp_fast.available():
        return
    generator = np.eye(3, dtype=np.uint8)
    # Nonzero pairing excludes span(e1,e2), forcing e0 content.
    detector = np.array([[1, 0, 0]], dtype=np.uint8)
    result = cpp_fast.classical_ris(
        generator, detectors=detector, trials=8, seed=17,
        pair_depth=3, target=2, stop_on_target=True)
    assert result["best_weight"] == 1
    assert 0 in result["witness"]


def test_resumable_ris_runner_rejects_mismatched_checkpoint(tmp_path):
    from gb_ris_validation_runner import _load_or_start
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}")
    checkpoint = tmp_path / "checkpoint.json"
    state = _load_or_start(checkpoint, candidate, 80, 20_000, 17)
    checkpoint.write_text(json.dumps(state))
    try:
        _load_or_start(checkpoint, candidate, 79, 20_000, 17)
    except ValueError as exc:
        assert "target" in str(exc)
    else:
        raise AssertionError("mismatched target was accepted")


def test_hyper_validator_checks_css_structure_and_witnesses():
    h = [[0, 1, 2, 3]]
    doc = {"n": 4, "k": 2, "checks": {"X": h, "Z": h},
           "distance": {"d": 2,
                        "X": {"value": 2, "witness": [0, 1]},
                        "Z": {"value": 2, "witness": [0, 1]}}}
    result = structural_validate(doc)
    assert result["ok"]
    assert result["computed_k"] == 2
    assert result["commutation_violations"] == 0


def test_parallel_ris_and_adaptive_validator_find_small_logical():
    import cpp_fast
    if not cpp_fast.available():
        return
    h = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    result = cpp_fast.css_ris_parallel(h, h, trials=8, seed=17, target=3,
                                       combo_depth=3, stop_on_target=True, threads=2)
    assert result["d_upper"] == 2
    assert result["screen_refuted"]
    doc = {"n": 6, "k": 4, "checks": {"X": [[0, 1, 2, 3, 4, 5]],
                                        "Z": [[0, 1, 2, 3, 4, 5]]},
           "distance": {"d": 4,
                        "X": {"value": 4, "witness": [0, 1, 2, 3]},
                        "Z": {"value": 4, "witness": [0, 1, 2, 3]}}}
    refined = refine_candidate(doc, stages=(8,), seed=17, threads=2)
    assert refined["status"] == "ok"
    assert refined["candidate"]["distance"]["d"] == 2


def test_validation_receipt_keeps_lightest_same_chunk_refutation():
    from gb_ris_validation_runner import _lightest_refutation
    state = {
        "target": 80,
        "best_observed": {
            "X": {"weight": 52, "witness": [1, 2], "chunk": 1, "seed": 7},
            "Z": {"weight": 76, "witness": [3, 4], "chunk": 1, "seed": 7},
        },
        "chunks": [{"index": 1, "witnesses": {
            "X": {"check": {"ok": True, "weight": 52}},
            "Z": {"check": {"ok": True, "weight": 76}},
        }}],
    }
    result = _lightest_refutation(state)
    assert result["side"] == "X"
    assert result["weight"] == 52


def test_frobenius_support_is_weight_preserving_and_periodic():
    from univariate_frobenius_campaign import frobenius_support
    word = (0, 3, 7, 10)
    assert frobenius_support(word, 2, 11) == (0, 3, 6, 9)
    assert frobenius_support(word, 4, 11) == (0, 1, 6, 7)


def test_stabilizer_multipliers_keep_the_exact_cyclic_ideal():
    from univariate_frobenius_campaign import stabilizer_multipliers
    # For m=5, A=1+x has the degree-one shared divisor.  At least the
    # Frobenius map x->x^2 must preserve it; all returned maps are exact.
    maps = stabilizer_multipliers((0, 1), 5)
    assert ("stabilizer_2", 2) in maps
    assert all(common_gcd_degree(5, (0, 1),
                                 tuple((p * x) % 5 for x in (0, 1))) == 1
               for _, p in maps)


def test_projection_specs_round_odd_score_floor_to_even_k():
    from gf2_factor import factor_xm_plus_one
    from projection_breakout_campaign import target_specs
    specs = target_specs([337], {337: factor_xm_plus_one(337)})
    assert [row["k"] for row in specs] == [126, 128, 168, 170]


def test_score_admission_uses_same_conservative_projection_as_target_gate():
    from projection_breakout_campaign import score_admission_distance
    # At the current board line, a [[682,182]] candidate needs raw d<=97
    # under the established 0.8 planning factor, not the d=125 research goal.
    assert score_admission_distance({"n": 682, "k": 182}) == 97


def test_score_admission_validator_filters_each_code_at_its_own_gate():
    from score_admission_validate import admission_rows
    document = {"proxy_top": [
        {"n": 682, "k": 182, "semantic_hash": "a", "screen": {"d_upper": 97}},
        {"n": 682, "k": 182, "semantic_hash": "b", "screen": {"d_upper": 96}},
    ]}
    assert [row["semantic_hash"] for row in admission_rows(document, 1582.226, 8)] == ["a"]


def test_raw_score_gate_targets_the_actual_board_line():
    from score_admission_validate import raw_score_distance
    assert raw_score_distance({"n": 630, "k": 182}, 1582.226) == 75


def test_branch_cap_prevents_one_proxy_clade_from_consuming_deep_beam():
    from projection_breakout_campaign import _branch_capped_rows
    rows = [
        {"branch_id": "a", "n": 10, "k": 2, "semantic_hash": f"a{i}",
         "screen": {"d_upper": 20 - i}}
        for i in range(4)
    ] + [{"branch_id": "b", "n": 10, "k": 2, "semantic_hash": "b",
           "screen": {"d_upper": 16}}]
    assert [row["semantic_hash"] for row in _branch_capped_rows(rows, 1)] == ["a0", "b"]


def test_frobenius_imported_parent_extracts_exact_common_divisor(tmp_path):
    import json
    from univariate_frobenius_campaign import _seed_branch
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"n": 10, "code_type": "CSS",
                                  "checks": {"X": [[0, 1, 5, 6]]}}))
    branch = _seed_branch(parent)
    assert branch["m"] == 5
    assert branch["k"] == 2
    assert branch["seed_support"] == [0, 1]


def test_block_swap_reverse_is_involution_and_weight_preserving():
    support = [0, 3, 5, 7]
    mapped = block_swap_reverse_support(support, 4)
    assert len(mapped) == len(support)
    assert block_swap_reverse_support(mapped, 4) == sorted(support)


def test_collapse_guard_tracks_orbit_pair_and_phase():
    rows = [
        {"A": [[0, 1, 2]], "B": [[0, 1, 3]], "d_proxy": 80,
         "proxy_repeat_scores": [80, 81]},
        {"A": [[1, 2, 3]], "B": [[1, 2, 4]], "d_proxy": 79,
         "proxy_repeat_scores": [79, 80]},
    ]
    annotated, meta = classify_rows(rows)
    assert meta["orbit_count"] == 2
    assert annotated[0]["orbit_pair"] == annotated[1]["orbit_pair"]
    assert annotated[0]["relative_phase"] == annotated[1]["relative_phase"]
    assert len(select_guard_beam(annotated, 1)) == 1


def test_collapse_guard_pair_cap_preserves_independent_lineages():
    rows = [
        {"orbit_pair": "a>b", "phase_bin": i, "semantic_hash": f"a{i}",
         "guard_scores": [90 - i], "dz_proxy": 90 - i, "dx_proxy": 90 - i}
        for i in range(3)
    ] + [
        {"orbit_pair": "c>d", "phase_bin": 0, "semantic_hash": "c0",
         "guard_scores": [80], "dz_proxy": 80, "dx_proxy": 80},
    ]
    selected = select_guard_beam(rows, 3, per_pair=1)
    assert [row["semantic_hash"] for row in selected] == ["a0", "c0"]


def test_collapse_blacklist_persists_only_refuted_signatures(tmp_path):
    path = tmp_path / "blacklist.json"
    payload = _update_blacklist(path, [
        {"collapse_signature": "0>1@3", "d": 76,
         "candidate_path": "bad.json"},
        {"collapse_signature": "0>1@4", "d": 78,
         "candidate_path": "survivor.json"},
    ])
    assert sorted(payload["entries"]) == ["0>1@3"]
    assert json.loads(path.read_text())["entries"]["0>1@3"]["observed_d"] == 76
