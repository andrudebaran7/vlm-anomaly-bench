import numpy as np
import pytest
from PIL import Image

from mvtec_tree import CONDITIONS, build_category
from prepare_data import check_category, main, summarize_category


def test_a_well_formed_category_has_no_problems(tmp_path):
    build_category(tmp_path, "vial")
    assert check_category(tmp_path, "vial") == []


def test_a_missing_split_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    for p in sorted((cat / "validation" / "good").glob("*.png")):
        p.unlink()
    (cat / "validation" / "good").rmdir()
    (cat / "validation").rmdir()
    problems = check_category(tmp_path, "vial")
    assert any("validation" in p for p in problems)


def test_an_orphan_mask_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    orphan = cat / "test_public" / "ground_truth" / "bad" / "999_regular_mask.png"
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L").save(orphan)
    problems = check_category(tmp_path, "vial")
    assert any("999_regular" in p for p in problems)


def test_a_missing_mask_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    next((cat / "test_public" / "ground_truth" / "bad").glob("*.png")).unlink()
    problems = check_category(tmp_path, "vial")
    assert any("mask" in p.lower() for p in problems)


def test_a_truncated_mask_is_reported(tmp_path):
    """Masks were only matched by stem, so a mask that cannot be opened passed as OK.

    That failure then surfaced hours later inside `aggregate._load_pair`, which is exactly
    the mid-run death this script exists to prevent.
    """
    cat = build_category(tmp_path, "vial")
    mask = next((cat / "test_public" / "ground_truth" / "bad").glob("*.png"))
    mask.write_bytes(mask.read_bytes()[: len(mask.read_bytes()) // 2])
    problems = check_category(tmp_path, "vial")
    assert any(mask.name in p for p in problems), problems


def test_a_mask_that_is_not_a_png_at_all_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    mask = next((cat / "test_public" / "ground_truth" / "bad").glob("*.png"))
    mask.write_bytes(b"not a real png")
    problems = check_category(tmp_path, "vial")
    assert any(mask.name in p for p in problems), problems


def test_a_mask_whose_dimensions_differ_from_its_image_is_reported(tmp_path):
    """`_load_pair` raises when mask.shape != amap.shape, and the amap has the image's shape.

    A mask of the wrong size is therefore a guaranteed mid-run failure, and stem matching
    alone cannot see it.
    """
    cat = build_category(tmp_path, "vial")
    mask = next((cat / "test_public" / "ground_truth" / "bad").glob("*.png"))
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8), mode="L").save(mask)  # images are 6x8
    problems = check_category(tmp_path, "vial")
    assert any(mask.name in p for p in problems), problems
    assert any("4x3" in p or "(3, 4)" in p for p in problems), problems


def test_main_returns_nonzero_for_a_mask_that_does_not_match_its_image(tmp_path, capsys):
    cat = build_category(tmp_path, "vial")
    mask = next((cat / "test_public" / "ground_truth" / "bad").glob("*.png"))
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8), mode="L").save(mask)
    assert main(["--root", str(tmp_path)]) == 1
    assert mask.name in capsys.readouterr().out


def test_a_corrupt_image_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    corrupt = next((cat / "train" / "good").glob("*.png"))
    corrupt.write_bytes(b"not a real png")
    problems = check_category(tmp_path, "vial")
    assert any(corrupt.name in p for p in problems)


def test_an_empty_split_directory_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    for p in (cat / "test_private").glob("*.png"):
        p.unlink()
    problems = check_category(tmp_path, "vial")
    assert any("test_private" in p and "empty" in p.lower() for p in problems)


def test_summary_counts_files_per_split_and_condition(tmp_path):
    build_category(tmp_path, "vial", n_good=2, n_bad=3)
    summary = summarize_category(tmp_path, "vial")
    assert summary["train"] == {"regular": 2}
    assert summary["validation"] == {"regular": 2}
    assert summary["test_public"]["regular"] == 5  # 2 good + 3 bad
    assert set(summary["test_public"]) == set(CONDITIONS)
    assert summary["test_private"] == {"regular": 3}
    assert summary["test_private_mixed"] == {"mixed": 3}


def test_main_returns_zero_for_a_good_tree(tmp_path, capsys):
    build_category(tmp_path, "vial")
    assert main(["--root", str(tmp_path)]) == 0
    assert "vial" in capsys.readouterr().out


def test_main_returns_nonzero_and_names_the_problem(tmp_path, capsys):
    cat = build_category(tmp_path, "vial")
    next((cat / "test_public" / "ground_truth" / "bad").glob("*.png")).unlink()
    assert main(["--root", str(tmp_path)]) == 1
    assert "mask" in capsys.readouterr().out.lower()


def test_main_can_check_one_category(tmp_path):
    build_category(tmp_path, "vial")
    build_category(tmp_path, "can")
    assert main(["--root", str(tmp_path), "--category", "vial"]) == 0


def test_main_reports_no_categories_found(tmp_path, capsys):
    assert main(["--root", str(tmp_path)]) == 1
    assert str(tmp_path) in capsys.readouterr().out
