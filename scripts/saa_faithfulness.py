#!/usr/bin/env python3
"""Check that our SAA+ configuration still matches the repo it claims to be verbatim from.

    python scripts/saa_faithfulness.py --repo /content/Segment-Any-Anomaly

SAA+ has no reproduction gate, because its paper publishes no image-level number to gate against
(protocol §2 v0.2.17, `configs/reproduction/saa.yaml`). This script is the safeguard that takes
the gate's place for the half of its job that still applies.

**Why this is the right shape of check for THIS method.** SAA+ is the one method this repo does
not reimplement: `anomaly_map_source: official_repo_final_map` stores the repo's final map
verbatim, so no arithmetic of ours can be wrong. What can be wrong is the configuration we drive
their code with — the wrong branch, prompts that silently fell back, a property constraint that
got reformatted. Those are checked here directly, against the source, rather than inferred from
whether a summary statistic landed near a published one.

It covers the STATIC half only — everything checkable from a clone with no GPU. The runtime half
(400x400, K=5, the map stored unmodified, the fallback recorded when it is used) is owed by the
backend and is listed in `configs/reproduction/saa.yaml` under `integration_faithfulness.runtime`.
This script says so on every run rather than letting a green tick imply more than it checked.

**The repo's Python is parsed, never imported.** `ast.literal_eval` on the assigned dict, not
`runpy`: this reads a third-party clone, and a provenance check that executes the thing it is
checking is not a check. It also means the script works on a repo whose dependencies are absent.

Exit codes: **0 all checks pass, 1 a check failed, 2 the check could not be run** — so a missing
clone is never mistaken for a clean bill of health.
"""
import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "methods"
#: The five token positions SAA/model.py reads out of a property prompt.
POSITIONS = {5: int, 6: str, 7: str, 12: int, 19: float}
#: What general_prompts.py must still say, since every MVTec AD 2 category falls back to it.
EXPECTED_FALLBACK = ["defect on {}", "damage on {}", "flaw on {}"]


def _literal(path: Path, name: str) -> Any:
    """Return the literal assigned to `name` at module level, without importing the module."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{path} has no module-level assignment to {name!r}")


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def ok(self, label: str) -> None:
        print(f"  PASS  {label}")

    def fail(self, label: str, detail: str) -> None:
        print(f"  FAIL  {label}\n        {detail}")
        self.failures.append(label)

    def note(self, text: str) -> None:
        self.notes.append(text)


def check_branch_and_commit(repo: Path, method: dict, c: Checks) -> None:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    if branch == method["branch"]:
        c.ok(f"branch is {branch}")
    else:
        c.fail("branch", f"clone is on {branch!r}, config says {method['branch']!r}. `master` "
                         "holds the vanilla SAA demo, whose published VisA Fp is 12.76 "
                         "against SAA+'s 27.07 — a different method, silently.")

    pinned = method["commit"]
    if pinned == "unpinned_until_first_colab_run":
        c.note(f"commit is still unpinned. This clone is at {head}. Record it in "
               f"configs/methods/saa.yaml and in the overlap-free provenance before scoring.")
    elif head == pinned:
        c.ok(f"HEAD matches the pinned commit {pinned[:12]}")
    else:
        c.fail("commit", f"clone is at {head}, config pins {pinned}. Everything below was "
                         "verified against the pinned one.")


def check_visa_prompts(repo: Path, prompts: dict, c: Checks) -> None:
    src = repo / "SAA" / "prompts" / "visa_parameters.py"
    manual = _literal(src, "manual_prompts")
    prop = _literal(src, "property_prompts")
    recorded = prompts["visa_categories"]

    if set(recorded) != set(manual):
        c.fail("visa objects", f"recorded {sorted(recorded)} vs repo {sorted(manual)}")
        return

    drift = []
    for cat, entry in recorded.items():
        pairs = [list(p) for p in entry["prompt_pairs"]]
        if pairs != [list(p) for p in manual[cat]]:
            drift.append(f"{cat}: prompt pairs differ")
        if entry["property_constraints"] != prop[cat]:
            drift.append(f"{cat}: property constraint differs BYTE-FOR-BYTE")
    if drift:
        c.fail("visa prompts are verbatim", "; ".join(drift))
    else:
        c.ok(f"all {len(recorded)} VisA prompt entries are byte-identical to the repo")


def check_positional_format(repo: Path, c: Checks) -> None:
    prop = _literal(repo / "SAA" / "prompts" / "visa_parameters.py", "property_prompts")
    model = (repo / "SAA" / "model.py").read_text()

    for index in POSITIONS:
        if f"split(' ')[{index}]" not in model:
            c.fail("positional format", f"SAA/model.py no longer reads token [{index}]. The "
                                        "property-prompt format has changed and every recorded "
                                        "constraint must be re-read.")
            return

    for cat, text in prop.items():
        tokens = text.split(" ")
        for index, parse in POSITIONS.items():
            try:
                parse(tokens[index])
            except (ValueError, IndexError):
                c.fail("positional format", f"{cat}: token {index} is "
                                            f"{tokens[index] if index < len(tokens) else '<absent>'!r}")
                return
    c.ok("every property constraint still parses at tokens 5, 6, 7, 12, 19")


def check_prompt_tables_in_use(repo: Path, prompts: dict, c: Checks) -> None:
    hybrid = (repo / "SAA" / "hybrid_prompts.py").read_text()
    dead = prompts["unused_table_in_source"]
    if dead in hybrid:
        c.fail("dead prompt table", f"SAA/hybrid_prompts.py now references {dead!r}, which we "
                                    "recorded as unused. The method's inputs have changed.")
    elif "manual_prompts" in hybrid and "property_prompts" in hybrid:
        c.ok(f"hybrid_prompts.py still uses manual_prompts + property_prompts, not {dead}")
    else:
        c.fail("prompt tables", "SAA/hybrid_prompts.py references neither manual_prompts nor "
                                "property_prompts; the wiring has changed.")


def check_fallback(repo: Path, c: Checks) -> None:
    src = repo / "SAA" / "prompts" / "general_prompts.py"
    found = _literal(src, "general_anomaly_description")
    if found == EXPECTED_FALLBACK:
        c.ok("the generic fallback is the three lines we recorded for MVTec AD 2")
    else:
        c.fail("fallback", f"general_anomaly_description is {found!r}, recorded as "
                           f"{EXPECTED_FALLBACK!r}. Every MVTec AD 2 category runs on this, so a "
                           "change here changes 8 of 8 categories.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True,
                        help="path to the Segment-Any-Anomaly clone")
    parser.add_argument("--method-config", type=Path, default=CONFIGS / "saa.yaml")
    parser.add_argument("--prompts-config", type=Path, default=CONFIGS / "saa_prompts.yaml")
    args = parser.parse_args(argv)

    if not (args.repo / "SAA" / "prompts").is_dir():
        print(f"cannot check: {args.repo}/SAA/prompts does not exist. Is this the SAA-plus "
              "branch? `master` has no SAA/ package at all.", file=sys.stderr)
        return 2

    with open(args.method_config) as fh:
        method = yaml.safe_load(fh)
    with open(args.prompts_config) as fh:
        prompts = yaml.safe_load(fh)

    c = Checks()
    print("SAA+ integration faithfulness — static checks\n")
    try:
        check_branch_and_commit(args.repo, method, c)
        check_visa_prompts(args.repo, prompts, c)
        check_positional_format(args.repo, c)
        check_prompt_tables_in_use(args.repo, prompts, c)
        check_fallback(args.repo, c)
    except (KeyError, SyntaxError, ValueError) as exc:
        print(f"\ncannot check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    for text in c.notes:
        print(f"\n  NOTE  {text}")

    print("\nThis covers the STATIC half only. The runtime half — 400x400 input, K=5, saliency "
          "N=400,\nthe stored map unmodified, and the fallback recorded when MVTec AD 2 uses it "
          "— is owed by\nthe backend; see integration_faithfulness.runtime in "
          "configs/reproduction/saa.yaml.")

    if c.failures:
        print(f"\nFAILED: {len(c.failures)} check(s): {', '.join(c.failures)}")
        return 1
    print("\nAll static checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
