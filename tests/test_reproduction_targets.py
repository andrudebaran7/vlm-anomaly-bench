"""The pre-registered reproduction targets must stay internally consistent with their source.

These are not tests of our code — they are tests of a transcription. A typo in a published
number silently moves a pre-registered gate, which is exactly the kind of drift protocol §2
exists to prevent.
"""
from pathlib import Path

import yaml

TARGETS = Path(__file__).resolve().parents[1] / "configs" / "reproduction" / "patchcore_ref.yaml"


def _load():
    with open(TARGETS) as fh:
        return yaml.safe_load(fh)


def test_the_per_category_numbers_average_to_the_published_mean():
    """Table S1 prints both the per-category row and its Avg. If our transcription of the 15
    values does not reproduce the printed 99.0, one of the 16 numbers was copied wrong."""
    cfg = _load()
    per_cat = cfg["published_per_category"]
    assert len(per_cat) == 15, "MVTec AD has 15 categories"
    mean = sum(per_cat.values()) / len(per_cat)
    assert abs(mean - cfg["gate"]["published_mean"]) < 0.1, (
        f"per-category mean {mean:.3f} does not reproduce the published "
        f"{cfg['gate']['published_mean']}"
    )


def test_the_gate_matches_the_coreset_ratio_the_repo_actually_runs():
    """`PatchCore-10` means 10% coreset subsampling. A gate taken from that row while the
    method runs a different ratio would compare two different methods."""
    with open(TARGETS.parent.parent / "methods" / "patchcore_ref.yaml") as fh:
        method = yaml.safe_load(fh)
    assert method["coreset_sampling_ratio"] == 0.1
    assert "PatchCore-10" in _load()["gate"]["source"]


def test_the_gate_is_mvtec_ad_classic_and_visa_is_only_secondary():
    """Protocol §2 v0.2.12. Swapping these would restore the gate the amendment removed."""
    cfg = _load()
    assert cfg["gate"]["dataset"] == "mvtec_ad"
    assert cfg["secondary"]["dataset"] == "visa"
    assert cfg["gate"]["metric"] == cfg["secondary"]["metric"] == "i_auroc"
    assert cfg["gate"]["tolerance"] == 1.0


def test_every_gate_number_carries_its_source():
    cfg = _load()
    assert "arXiv:2106.08265" in cfg["gate"]["source"]
    assert "arXiv:2207.14315" in cfg["secondary"]["source"]
    assert cfg["secondary"]["caveat"].strip()


def test_the_number_of_seeds_is_pre_registered():
    """Protocol §6 asks for three seeds reported as mean ± std. The gate enforces that count,
    so the count has to be written down before the run rather than read off it."""
    assert _load()["n_seeds"] == 3


# --- WinCLIP -----------------------------------------------------------------------------
# Two gating blocks rather than one gate and one secondary check, because WinCLIP's own paper
# (arXiv:2303.14814, CVPR 2023) postdates the VisA dataset and reports both benchmarks at the
# configuration this repo runs. Protocol §2 v0.2.14.

WINCLIP = TARGETS.parent / "winclip.yaml"


def _winclip():
    with open(WINCLIP) as fh:
        return yaml.safe_load(fh)


def test_winclip_per_category_numbers_average_to_their_published_means():
    """Tables 10 and 16 print the per-category values; Table 1 prints their means. If our
    transcription of the 15 + 12 values does not reproduce 91.8 and 78.1, a cell was copied
    wrong -- and a mistyped gate is one nobody would notice from the verdict."""
    cfg = _winclip()
    for block, n, published in (("gate_mvtec_ad", 15, 91.8), ("gate_visa", 12, 78.1)):
        per_cat = cfg[block]["per_category"]
        assert len(per_cat) == n, f"{block} should hold {n} categories"
        mean = sum(per_cat.values()) / len(per_cat)
        assert abs(mean - published) < 0.1, (
            f"{block}: per-category mean {mean:.3f} does not reproduce the published {published}"
        )
        assert cfg[block]["published_mean"] == published


def test_both_winclip_blocks_gate():
    """The whole point of the WinCLIP entry. Demoting either to a non-gating check would
    silently remove a criterion its own paper supports (protocol §2 v0.2.14)."""
    cfg = _winclip()
    assert cfg["gate_mvtec_ad"]["gates"] is True
    assert cfg["gate_visa"]["gates"] is True
    assert cfg["gate_mvtec_ad"]["dataset"] == "mvtec_ad"
    assert cfg["gate_visa"]["dataset"] == "visa"
    assert cfg["gate_mvtec_ad"]["metric"] == cfg["gate_visa"]["metric"] == "i_auroc"
    assert cfg["gate_mvtec_ad"]["tolerance"] == cfg["gate_visa"]["tolerance"] == 1.0


def test_winclip_targets_are_the_zero_shot_ones_at_the_backbone_the_repo_runs():
    """Table 1 has 0-shot, 1-, 2- and 4-shot blocks for the same method name. Taking a k-shot
    row would gate our zero-shot adapter against WinCLIP+."""
    cfg = _winclip()
    for block in ("gate_mvtec_ad", "gate_visa"):
        assert "0-shot" in cfg[block]["source"]
        assert "arXiv:2303.14814" in cfg[block]["source"]
    with open(WINCLIP.parent.parent / "methods" / "winclip.yaml") as fh:
        method = yaml.safe_load(fh)
    assert method["k_shot"] == 0
    assert method["zero_shot"] is True
    # The paper's default is LAION-400M CLIP with ViT-B/16+ at 240^2 (Section 5, Appendix A).
    assert method["backbone"] == "ViT-B-16-plus-240"
    assert method["scales"] == [2, 3]          # small-scale 2x2, mid-scale 3x3 in ViT patches


def test_the_specific_states_ablation_is_recorded_as_not_the_target():
    """Table 7's VisA 78.9 beats the headline 78.1 and would be the tempting number to chase.
    It comes from per-category defect words, which protocol §3 forbids us to write."""
    cfg = _winclip()
    assert cfg["not_the_target"]["visa_specific_states"] == 78.9
    assert cfg["gate_visa"]["published_mean"] < cfg["not_the_target"]["visa_specific_states"]


def test_winclip_pre_registers_how_the_seed_question_gets_settled():
    """Protocol §6 is conditional ("where any stochasticity exists") and whether this adapter
    has any is unknown. The config has to carry the deciding procedure, not just a count, or
    the answer becomes a choice made after seeing a result."""
    cfg = _winclip()
    assert cfg["n_seeds"] == 3
    text = WINCLIP.read_text()
    assert "bit-identical" in text and "--all-seeds" in text


# --- AnomalyCLIP -------------------------------------------------------------------------
# Two gates again, but each against a DIFFERENT checkpoint: the paper fine-tunes on the
# auxiliary dataset's test split and swaps which one per target (arXiv:2310.18961v12 §4.1).

ANOMALYCLIP = TARGETS.parent / "anomalyclip.yaml"


def _anomalyclip():
    with open(ANOMALYCLIP) as fh:
        return yaml.safe_load(fh)


def test_anomalyclip_per_category_numbers_average_to_their_published_means():
    cfg = _anomalyclip()
    for block, n, published in (("gate_mvtec_ad", 15, 91.5), ("gate_visa", 12, 82.1)):
        per_cat = cfg[block]["per_category"]
        assert len(per_cat) == n, f"{block} should hold {n} categories"
        mean = sum(per_cat.values()) / len(per_cat)
        assert abs(mean - published) < 0.1, (
            f"{block}: per-category mean {mean:.3f} does not reproduce the published {published}"
        )
        assert cfg[block]["published_mean"] == published


def test_each_anomalyclip_gate_names_the_checkpoint_it_is_the_target_for():
    """The paper's two numbers come from two different checkpoints -- VisA 82.1 from the
    MVTec-AD-trained model, MVTec AD 91.5 from the VisA-trained one -- because it fine-tunes on
    the OTHER dataset's test split. A gate that did not name its checkpoint would fail a correct
    implementation that happened to load the other one."""
    cfg = _anomalyclip()
    assert cfg["gate_visa"]["checkpoint"] == "mvtec_ad_trained"
    assert cfg["gate_mvtec_ad"]["checkpoint"] == "visa_trained"
    assert cfg["gate_visa"]["checkpoint"] != cfg["gate_mvtec_ad"]["checkpoint"]


def test_the_anomalyclip_gate_checkpoints_agree_with_the_overlap_audit():
    """The audit designates which checkpoint is clean for which test set (protocol §3.1). If the
    targets file and the audit drifted apart, one of them would be authorising leakage."""
    cfg = _anomalyclip()
    with open(ANOMALYCLIP.parent.parent / "methods" / "anomalyclip.yaml") as fh:
        method = yaml.safe_load(fh)
    assert method["aux_trained"] is True
    assert method["checkpoint_for_visa_reproduction"] == cfg["gate_visa"]["checkpoint"]
    # MVTec AD classic's gate runs the same checkpoint as the MVTec AD 2 primary evaluation,
    # which is what makes that gate worth more than its verdict.
    assert method["checkpoint_for_mvtec_ad2"] == cfg["gate_mvtec_ad"]["checkpoint"]


def test_the_published_configuration_is_recorded_with_the_targets():
    """A target is only a criterion at the configuration it was measured at. The 518x518 input
    into a 336px backbone is the one most likely to be silently 'fixed' by an implementer."""
    cfg = _anomalyclip()["published_configuration"]
    assert cfg["input_resolution"] == 518
    assert cfg["test_time_map_smoothing"].startswith("Gaussian")
    with open(ANOMALYCLIP.parent.parent / "methods" / "anomalyclip.yaml") as fh:
        method = yaml.safe_load(fh)
    assert method["input_resolution"] == 518, "the method config must run what the target assumes"
    assert method["test_time_map_smoothing_sigma"] == 4


def test_the_arxiv_version_the_targets_were_read_from_is_pinned():
    """Twelve revisions exist and the PDF carries no ICLR banner. "ICLR 2024" alone cannot
    answer "does the version we read print these numbers?"."""
    cfg = _anomalyclip()
    for block in ("gate_mvtec_ad", "gate_visa"):
        assert "arXiv:2310.18961v12" in cfg[block]["source"]
    with open(ANOMALYCLIP.parent.parent / "methods" / "anomalyclip.yaml") as fh:
        assert "v12" in yaml.safe_load(fh)["paper_version"]


# --- AdaCLIP -----------------------------------------------------------------------------
# Two gates and two checkpoints again (arXiv:2407.15795v1 §5.1), but with three wrinkles none of
# the earlier methods had: each checkpoint carries a medical auxiliary dataset as well as an
# industrial one, the paper reports the same two datasets a SECOND time under a different setup,
# and the repo ships a third checkpoint that must never be loaded.

ADACLIP = TARGETS.parent / "adaclip.yaml"


def _adaclip():
    with open(ADACLIP) as fh:
        return yaml.safe_load(fh)


def _adaclip_method():
    with open(ADACLIP.parent.parent / "methods" / "adaclip.yaml") as fh:
        return yaml.safe_load(fh)


def test_adaclip_per_category_numbers_average_to_their_published_means():
    cfg = _adaclip()
    for block, n, published in (("gate_mvtec_ad", 15, 89.2), ("gate_visa", 12, 85.8)):
        per_cat = cfg[block]["per_category"]
        assert len(per_cat) == n, f"{block} should hold {n} categories"
        mean = sum(per_cat.values()) / len(per_cat)
        assert abs(mean - published) < 0.1, (
            f"{block}: per-category mean {mean:.3f} does not reproduce the published {published}"
        )
        assert cfg[block]["published_mean"] == published


def test_both_adaclip_blocks_gate():
    cfg = _adaclip()
    assert cfg["gate_mvtec_ad"]["gates"] is True
    assert cfg["gate_visa"]["gates"] is True
    assert cfg["gate_mvtec_ad"]["dataset"] == "mvtec_ad"
    assert cfg["gate_visa"]["dataset"] == "visa"
    assert cfg["gate_mvtec_ad"]["metric"] == cfg["gate_visa"]["metric"] == "i_auroc"
    assert cfg["gate_mvtec_ad"]["tolerance"] == cfg["gate_visa"]["tolerance"] == 1.0


def test_each_adaclip_gate_names_the_full_checkpoint_it_is_the_target_for():
    """Every published checkpoint is trained on an industrial AND a medical dataset. A block
    naming only the industrial half would name a checkpoint that does not exist, and the repo
    publishes its weights under the two-dataset labels asserted here."""
    cfg = _adaclip()
    assert cfg["gate_mvtec_ad"]["checkpoint"] == "visa_colondb_trained"
    assert cfg["gate_visa"]["checkpoint"] == "mvtec_ad_clinicdb_trained"
    assert cfg["gate_mvtec_ad"]["checkpoint"] != cfg["gate_visa"]["checkpoint"]


def test_the_adaclip_gate_checkpoints_agree_with_the_overlap_audit():
    """Protocol §3.1, same invariant as AnomalyCLIP's: if the targets file and the method config
    drifted apart, one of them would be authorising leakage."""
    cfg = _adaclip()
    method = _adaclip_method()
    assert method["aux_trained"] is True
    assert method["checkpoint_for_visa_reproduction"] == cfg["gate_visa"]["checkpoint"]
    # The MVTec AD gate runs the same checkpoint as the MVTec AD 2 primary evaluation.
    assert method["checkpoint_for_mvtec_ad2"] == cfg["gate_mvtec_ad"]["checkpoint"]


def test_the_all_datasets_checkpoint_is_forbidden_in_both_files():
    """The repo's third weight trains on 14 datasets including both mvtec and visa. Loading it
    would void the zero-shot claim, and nothing in a filename would reveal it -- which is exactly
    why the refusal has to live in the config rather than in someone's memory."""
    cfg = _adaclip()
    assert cfg["forbidden_checkpoint"] == "all_datasets"
    assert _adaclip_method()["forbidden_checkpoint"] == "all_datasets"
    for block in ("gate_mvtec_ad", "gate_visa"):
        assert cfg[block]["checkpoint"] != cfg["forbidden_checkpoint"]


def test_the_anomalyclip_setting_numbers_are_recorded_as_not_the_target():
    """Appendix §4 reports the same two datasets under AnomalyCLIP's single-auxiliary-dataset
    setup. Its MVTec figure is HIGHER than the gate, so it is the number a narrow miss would
    tempt someone toward -- and the repo publishes no weights for it."""
    cfg = _adaclip()
    assert cfg["not_the_target"]["anomalyclip_setting_mvtec_ad"] == 89.6
    assert cfg["not_the_target"]["anomalyclip_setting_visa"] == 83.9
    assert cfg["gate_mvtec_ad"]["published_mean"] < cfg["not_the_target"][
        "anomalyclip_setting_mvtec_ad"
    ]


def test_the_maintainers_warning_about_the_released_weights_is_pre_registered():
    """The repo states its released weights do not reproduce its published table, by an
    unquantified amount. Written down before the run so it cannot be produced afterwards as an
    excuse for a miss -- and flagged to be reported with a PASS too."""
    cfg = _adaclip()["released_weights_caveat"]
    assert cfg["quantified"] is False
    assert cfg["report_with_verdict"] is True
    assert "caoyunkang/AdaCLIP" in cfg["source"]


def test_the_adaclip_published_configuration_matches_what_the_method_config_runs():
    """A target is only a criterion at the configuration it was measured at. As with AnomalyCLIP,
    the 518x518 input into a 336px backbone is the one an implementer would silently 'fix'."""
    cfg = _adaclip()["published_configuration"]
    method = _adaclip_method()
    assert cfg["input_resolution"] == method["input_resolution"] == 518
    assert cfg["feature_layers"] == method["feature_layers"] == [6, 12, 18, 24]
    assert cfg["prompting_depth"] == method["prompting_depth"] == 4
    assert cfg["prompting_length"] == method["prompting_length"] == 5
    assert method["backbone"] == "ViT-L-14-336"


def test_the_adaclip_arxiv_version_the_targets_were_read_from_is_pinned():
    cfg = _adaclip()
    for block in ("gate_mvtec_ad", "gate_visa"):
        assert "arXiv:2407.15795v1" in cfg[block]["source"]
    assert "2407.15795v1" in _adaclip_method()["paper_version"]


def test_adaclip_pre_registers_how_the_seed_question_gets_settled():
    """Same conditional as WinCLIP's and AnomalyCLIP's, and AdaCLIP has a clustering step (HSF,
    K=20) where a seed can hide."""
    cfg = _adaclip()
    assert cfg["n_seeds"] == 3
    text = ADACLIP.read_text()
    assert "bit-identical" in text and "--all-seeds" in text
