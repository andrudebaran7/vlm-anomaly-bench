# PatchCore Antialiased Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PatchCore's `score()` use the same antialiased pre-processing as its fit (protocol v0.2.21). Keep every new result apart from the old ones. Prepare the Colab session that re-runs the MVTec AD classic gate and the eight MVTec AD 2 categories.

**Architecture:** WinCLIP's `use_antialiased_forward` moves to a shared module, and `PatchCoreBackend` calls it after building the model. `resize_antialias` goes from the backend to the adapter and into every shard, the same way `preprocess` already does. The gate and the MVTec AD 2 summary refuse a set of shards that mixes values. A per-method results label (`patchcore_ref_v0.2.21`) sends new shards, Drive copies and reports to their own directories. Without it, `restore_from_drive` would copy the old shards back and the runner would skip every seed as already done. The Colab session itself is not part of this plan: the plan ends when the notebook is ready to run it.

**Tech Stack:** Python 3.11/3.13, pytest, anomalib 2.6.0 (GPU only, imported lazily), torchvision v2 transforms, Jupyter notebook JSON.

**Spec:** `docs/superpowers/specs/2026-10-19-patchcore-antialiased-anchor-design.md`

## Global Constraints

- anomalib pinned at `"2.6.0"`; no module-scope `import torch`/`anomalib` anywhere in `src/` or `scripts/` (CI has neither).
- The corrected configuration is hard-wired: no `resize_antialias: false` code path.
- `preprocess="classic"` is refused under v0.2.21.
- No old result file or shard is overwritten, moved or deleted. Old reports get **one dated header line** and nothing else.
- Hyperparameters unchanged: `wide_resnet50_2`, `[layer2, layer3]`, coreset 0.1, 9 neighbours, `Resize([256, 256])`.
- New results live under the label `patchcore_ref_v0.2.21`. The method name in shards stays `patchcore_ref`.
- Seeds 0/1/2. MVTec AD 2 split `test_public`.
- Feature branch `patchcore-antialiased-anchor` (already exists, holds the spec). No worktree: the editable install makes a worktree's tests exercise the main checkout. Merge with `--no-ff`. Never commit to master.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Full suite: `python -m pytest -q`. It must be green at the end of every task.

## Review Focus

1. **A resumed MVTec AD 2 run on a VM that still holds old Drive shards.** With the new label, `restore_from_drive` must not bring back any old shard (pinned in Task 4).
2. **Old and new shards copied into one root** (by hand, as happened on 2026-09-16). The gate and `summarise` must refuse mixed `resize_antialias`, and they must treat "absent" (old shards) as a value distinct from `True` (Tasks 3, 5).
3. **The grid report regenerated before the gate has run.** The flag must say "verdict pending", never "reproduces" (Task 5).
4. **`PatchCoreBackend(preprocess="classic")` called from an old notebook cell** (3b.x, 3c). It must fail before anomalib is imported, with a message naming v0.2.21 (Task 2).
5. **anomalib's `transform` losing its antialiasing Resize.** The shared helper must refuse, and its message must name the calling method's protocol entry, not WinCLIP's (Task 1).

---

### Task 1: Protocol v0.2.21, config, shared antialias helper

**Files:**
- Create: `src/vlmab/methods/anomalib_preprocess.py`
- Modify: `src/vlmab/methods/winclip_backend.py:99-121` (delete the function body, import it from the new module, keep the name importable from `winclip_backend`)
- Modify: `docs/protocol.md` (§3: new bullet after "WinCLIP resizes with antialiasing (v0.2.20)"; the v0.2.20 bullet's last two sentences get "superseded for PatchCore by v0.2.21"; Changelog entry `2026-10-19 — v0.2.21`)
- Modify: `configs/methods/patchcore_ref.yaml` (add `resize_antialias: true` with a comment citing v0.2.21 and check (b))
- Test: `tests/test_anomalib_preprocess.py` (new), `tests/test_patchcore_backend.py`

**Interfaces:**
- Produces: `use_antialiased_forward(pre, protocol_ref: str) -> None` in `vlmab.methods.anomalib_preprocess`. It sets `pre.export_transform = pre.transform`. It raises `RuntimeError` mentioning "Resize" when `transform` has no Resize, and mentioning "antialias" when any Resize lacks `antialias=True`. Both messages include `protocol_ref`. `winclip_backend` calls it with `protocol_ref="protocol §3 v0.2.20"`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_anomalib_preprocess.py`, reuse the fake `Resize`/`Normalize`/`Compose`/`_fake_pre` from `tests/test_winclip_backend.py:150-176`, copied rather than imported.
  - `test_forward_is_routed_to_the_antialiased_transform`: after the call, `pre.export_transform is pre.transform`.
  - `test_refusal_names_the_callers_protocol_entry`: `pytest.raises(RuntimeError, match="v0.2.21")` for `_fake_pre(antialias_in_transform=False)` with `protocol_ref="protocol §3 v0.2.21"`.
  - `test_routing_refuses_a_transform_with_no_resize`: matches `"Resize"`.

  In `tests/test_patchcore_backend.py`, add `test_the_config_records_the_antialiased_resize`: `cfg["resize_antialias"] is True`.
- [ ] **Step 2: Run** `python -m pytest tests/test_anomalib_preprocess.py tests/test_patchcore_backend.py -q`. Expected: FAIL (module missing, key missing).
- [ ] **Step 3: Implement** the module (lazy-free: it only inspects attributes), the WinCLIP import change, and the yaml key. Write the protocol §3 bullet and the Changelog entry. The Changelog entry must state:
  - the check (b) measurement (forward 50.480389 = export path; train path 32.635822; classic forward map 256x256 against train 224x224);
  - that the author decided it 2026-10-19 before any corrected number existed;
  - the commitment: unconditional anchor; a failed gate is flagged non-reproducing per §2 and the eight MVTec AD 2 categories still run; old numbers kept as evidence;
  - that `classic` is refused;
  - "No metric definition changed; no existing number moves."

  Provenance line: `docs/next-steps.md, "MEASURED 2026-10-19 — check (b)"`.
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: all pass, including the existing WinCLIP routing tests unchanged.
- [ ] **Step 5: Commit** `feat: protocol v0.2.21 — PatchCore forward resizes with antialiasing; shared helper`

### Task 2: PatchCoreBackend routes forward and refuses classic

**Files:**
- Modify: `src/vlmab/methods/patchcore_backend.py` (`__init__`, module docstring, `PREPROCESS_CHOICES` comment)
- Test: `tests/test_patchcore_backend.py`

**Interfaces:**
- Consumes: `use_antialiased_forward(pre, protocol_ref)` from Task 1.
- Produces: public attribute `PatchCoreBackend.resize_antialias: bool` (always `True`). `PatchCoreBackend(preprocess="classic")` raises `ValueError` matching `"v0.2.21"` **before** anomalib is imported. `preprocess_spec("classic")` itself is unchanged, because the probe and the spec tests still use it.

- [ ] **Step 1: Write the failing tests.**
  - `test_classic_is_refused_before_anomalib_is_imported`: mirror the existing `test_an_invalid_preprocess_is_refused_before_anomalib_is_imported`. Expect `ValueError` matching `"v0.2.21"`.
  - `test_the_backend_routes_forward_to_the_antialiased_transform`:
    - `monkeypatch.setitem(sys.modules, ...)` fake `anomalib`, `anomalib.models` (with a `Patchcore` class whose instances have `pre_processor = _fake_pre()`) and `anomalib.engine` (with a no-op `Engine`).
    - Construct `PatchCoreBackend(device="cpu")`.
    - Assert `b._model.pre_processor.export_transform is b._model.pre_processor.transform` and `b.resize_antialias is True`.
- [ ] **Step 2: Run** `python -m pytest tests/test_patchcore_backend.py -q`. Expected: FAIL on both.
- [ ] **Step 3: Implement.** Put the classic refusal right after `preprocess_spec(preprocess)`. The message must say why: with the cropped transform the map covers only the central 224 px and is stretched over the frame. Call `use_antialiased_forward(self._model.pre_processor, "protocol §3 v0.2.21")` after `Patchcore(...)` is built, and set `self.resize_antialias = True` next to `self.preprocess`. Leave `_apply_classic_preprocessing` in place (unreachable, documented as kept for a later decision). Update the `score()` comment that says the pre-processor does "exactly as it does for every image during fit": that is now true *because of* the routing, so say so.
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: pass.
- [ ] **Step 5: Commit** `feat: PatchCore backend scores through the antialiased transform; classic refused (v0.2.21)`

### Task 3: `resize_antialias` reaches every shard, and the gate refuses a mix

**Files:**
- Modify: `src/vlmab/methods/patchcore_ref.py` (declare `resize_antialias` the way `_init_preprocess` declares `preprocess`)
- Modify: `scripts/run_mvtec_ad.py:56-70`, `scripts/run_mvtec_ad2.py:141-157` (stamp the column and include it in the `run_meta` cfg, beside `preprocess`)
- Modify: `scripts/reproduction_gate.py:84-101` (generalise `_check_single_preprocess`)
- Test: `tests/test_patchcore_ref.py`, `tests/test_reproduction_gate.py`, `tests/test_run_mvtec_ad2.py`, `tests/test_run_mvtec_ad_script.py`

**Interfaces:**
- Produces:
  - Read-only property `PatchCoreRef.resize_antialias -> bool | None`: the backend's value via `getattr(backend, "resize_antialias", None)`, or `None` with no backend.
  - Shard column `resize_antialias` (bool, or None for methods without it).
  - Shard columns `anomalib_version` and `torch_version`. The spec requires runtime versions in every new result file. Read them with `importlib.metadata.version(...)` inside `run_one_seed` (lazy; `None` if not installed). They are stamped only, never part of `config_hash`.
  - The gate report's Provenance section lists the distinct values of both columns when present.
  - `_check_single_config(df, results: Path) -> None` in `reproduction_gate.py`. It replaces `_check_single_preprocess`, keeps its call site, and checks the columns `("preprocess", "resize_antialias")`. A column absent from some shards counts as the value `"absent"`, so old+new pooled is refused. A column absent from all shards is skipped.

- [ ] **Step 1: Write the failing tests.**
  - `test_the_adapter_declares_the_antialiasing_its_backend_applied` (fake backend with `resize_antialias = True` → adapter `True`; without the attribute → `None`).
  - `test_the_gate_refuses_shards_mixing_antialiased_and_old` (a DataFrame with two rows of the same preprocess: one with `resize_antialias=True`, one with NaN → `ValueError` matching `"resize_antialias"`).
  - `test_the_gate_still_refuses_mixed_preprocess` (the existing test renamed if needed).
  - For each runner, extend its existing `run_one_seed` meta test (or add one, using the fake method pattern already in those test files) to assert that the shard meta carries `resize_antialias`, `anomalib_version` and `torch_version`.
  - `test_the_gate_report_lists_runtime_versions` (shards carrying `torch_version="2.11.0+cu128"` → the substring appears in the report).
- [ ] **Step 2: Run** the four test files. Expected: FAIL.
- [ ] **Step 3: Implement** as specified in Interfaces. Do not put `resize_antialias` on the shared `BackendSeeded` contract (same reasoning as `preprocess`, `patchcore_ref.py:44-46`).
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: pass.
- [ ] **Step 5: Commit** `feat: stamp resize_antialias on every shard; the gate refuses a pooled old+new root`

### Task 4: New results label, so old shards can never be restored into a new run

**Files:**
- Modify: `src/vlmab/methods/gpu.py` (add the label table and function)
- Modify: `scripts/run_mvtec_ad2.py:52-67` (`default_results`, `default_drive_results`), `scripts/run_mvtec_ad.py:40-43` (`default_results`)
- Test: `tests/test_gpu_factory.py`, `tests/test_run_mvtec_ad2.py`, `tests/test_run_mvtec_ad_script.py`

**Interfaces:**
- Produces: `results_label(method: str) -> str` in `vlmab.methods.gpu`. It returns `"patchcore_ref_v0.2.21"` for `"patchcore_ref"` and the method name unchanged otherwise. It lives in a module-level dict `_RESULTS_LABELS = {"patchcore_ref": "patchcore_ref_v0.2.21"}`, with a comment saying a label changes only with a new protocol version that changes the method's scoring path.
- The defaults become:
  - `results/mvtec_ad2/<label>/<category>` (report `results/mvtec_ad2/<label>/<category>.md`: one method level, so the existing `.gitignore` negation `!results/mvtec_ad2/*/*.md` still tracks it);
  - `DRIVE_RESULTS/<label>/<category>`;
  - `results/reproduction/<label>/mvtec_ad`.

- [ ] **Step 1: Write the failing tests.**
  - `test_patchcore_results_go_under_the_v0221_label` and `test_winclip_keeps_its_method_name_as_label`, for `results_label`.
  - `test_default_results_never_point_at_the_pre_v0221_patchcore_tree`: `default_results("patchcore_ref", "vial") != Path("results/mvtec_ad2/patchcore_ref/vial")`, same for the Drive default.
  - `test_a_new_run_restores_nothing_from_the_old_drive_tree`, using `tmp_path` and the existing `restore_from_drive`:
    - place a shard under `<drive>/patchcore_ref/vial/shards/`;
    - call `restore_from_drive(default_results(...) rebased on tmp, default_drive_results(...) rebased on tmp)`;
    - assert the new local shards dir is empty or absent.
  - In `tests/test_results_are_tracked.py`, add `test_a_report_under_the_v0221_label_is_tracked`: `git check-ignore` returns 1 for `results/mvtec_ad2/patchcore_ref_v0.2.21/vial.md` and for `results/reproduction/patchcore_mvtec_ad_v0.2.21.md` (check-ignore works on non-existent paths).
- [ ] **Step 2: Run** them. Expected: FAIL (no `results_label`).
- [ ] **Step 3: Implement.** Update the `default_results` docstring in `run_mvtec_ad2.py`: the "pre-2026-10-01 shards are not migrated" paragraph gains one sentence on why v0.2.21 needs its own label (restore + `is_done`).
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: pass.
- [ ] **Step 5: Commit** `feat: v0.2.21 results label keeps corrected PatchCore shards and reports apart from the old`

### Task 5: The MVTec AD 2 report states the configuration and the gate verdict

**Files:**
- Modify: `scripts/run_mvtec_ad2.py:223-400` (`summarise`: refuse mixed config; new caveats)
- Test: `tests/test_run_mvtec_ad2.py`

**Interfaces:**
- Consumes: shard column `resize_antialias` (Task 3); `_check_single_config` (Task 3). `summarise` imports it from `reproduction_gate` if the script layout allows (both are in `scripts/`, so check how the existing tests import them), otherwise a local copy with the same rule.
- Produces:
  - `GATE_REPORT = Path("results/reproduction/patchcore_mvtec_ad_v0.2.21.md")`.
  - `gate_verdict(report: Path) -> str | None`: returns `"PASS"` or `"FAIL"` parsed from the line `## Verdict: **PASS**` / `**FAIL**` that `reproduction_gate.py` writes, or `None` when the file is absent.
  - `summarise(results, max_bytes=None, gate_report: Path = GATE_REPORT)`.
- The caveat copy (exact). Emitted only for `patchcore_ref` shards whose `resize_antialias` is `True`:
  - always: `- **Protocol v0.2.21 configuration:** forward resizes with antialiasing, the same pre-processing the memory bank was built with. Numbers from before 2026-10-19 used a non-antialiased scoring path and are a different configuration; they are not comparable entries.`
  - FAIL: `- ⚠️ **NON-REPRODUCING ANCHOR.** Under this configuration PatchCore failed its MVTec AD classic gate (\`results/reproduction/patchcore_mvtec_ad_v0.2.21.md\`). Per protocol §2 every number here is flagged as non-reproducing.`
  - None: `- ⚠️ **Gate verdict pending.** \`results/reproduction/patchcore_mvtec_ad_v0.2.21.md\` does not exist yet. Regenerate this report with \`--summarise-only\` once it does; until then these numbers are not reportable.`
  - PASS: `- The anchor passed its MVTec AD classic gate under this configuration (\`results/reproduction/patchcore_mvtec_ad_v0.2.21.md\`).`
  - The existing `Resize([256, 256])` caveat stays, with "the reproduction gate was passed at" changed to "the reproduction gate was run at".

- [ ] **Step 1: Write the failing tests.**
  - `test_gate_verdict_reads_pass_fail_and_absent` (three tmp files, one missing).
  - `test_summary_refuses_mixed_antialiasing` (two shards, `True` and absent → `ValueError` matching `"resize_antialias"`).
  - `test_report_flags_a_non_reproducing_anchor`, `test_report_says_pending_without_a_gate_report`, `test_report_states_the_v0221_configuration`: build shards with the existing test helpers in `tests/test_run_mvtec_ad2.py`, call `summarise(..., gate_report=tmp)`, and assert on substrings `"NON-REPRODUCING ANCHOR"`, `"Gate verdict pending"`, `"Protocol v0.2.21 configuration"`.
  - `test_an_old_configuration_report_gets_none_of_the_v0221_caveats` (shards without the column).
  - `test_report_provenance_lists_runtime_versions` (Provenance section shows `anomalib_version` and `torch_version` values from the shards).
- [ ] **Step 2: Run** `python -m pytest tests/test_run_mvtec_ad2.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** Add `--gate-report` to `main` (default `GATE_REPORT`) and pass it through.
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: pass.
- [ ] **Step 5: Commit** `feat: MVTec AD 2 reports state the v0.2.21 configuration and flag the gate verdict`

### Task 6: Probe stage 15 — the standing guard

**Files:**
- Modify: `notebooks/probe_patchcore.py` (new stage after `_resize_path`; `_resize_path`'s docstring gains "Superseded 2026-10-19 by stage 15; it builds a `classic` backend, which v0.2.21 refuses, so it no longer runs.")
- Create: `tests/test_probe_stage15.py` (mirror `tests/test_probe_stage14.py`)

**Interfaces:**
- Produces: stage registered as `"15. score() resizes with antialiasing (v0.2.21 guard)"`, expecting `"forward == train path"`. Its body:
  - builds `synthetic_images(ctx)` **before** `import torch`, as stage 14 does, so the stage gets this far on CPU;
  - fits `PatchCoreBackend(seed=0, num_workers=2)` (anomalib preprocess);
  - scores the same 480x640 query as stage 14 (`np.random.default_rng(1)`) via `forward`, the export path, the train path and `score()`, printing each;
  - **asserts** forward == train path, and score() == forward (the stage's `same()` tolerance, 1e-5 relative);
  - asserts `b.resize_antialias is True`;
  - asserts `PatchCoreBackend(preprocess="classic")` raises `ValueError`;
  - returns `"forward == train path (delta <x>)"`.

- [ ] **Step 1: Write the failing tests:** registration under the exact name; `run(only=STAGE)` builds 20 images and prints no `KeyError`.
- [ ] **Step 2: Run** `python -m pytest tests/test_probe_stage15.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the stage.
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: pass.
- [ ] **Step 5: Commit** `feat: probe stage 15 — the v0.2.21 forward/train equality guard`

### Task 7: Notebook phase 6 — the corrected-anchor session

**Files:**
- Modify: `notebooks/patchcore_colab.ipynb`:
  - append phase 6 after cell 52;
  - add a one-line "superseded by v0.2.21; `classic` is refused" note at the top of the markdown cells for 3b and 3c;
  - change the cell 51 markdown's archive line to say that archives may have been deleted after M3 and must be re-uploaded one at a time.
- Test: `tests/test_notebook_run_cells.py`

**Interfaces:**
- Consumes: probe stage 15 name (Task 6); `scripts/run_mvtec_ad.py` and `scripts/run_mvtec_ad2.py` defaults (Task 4); `GATE_REPORT` path (Task 5).
- Cells (markdown + code, in the existing style: self-contained, session path stated, "paste the output back"):
  - **6.0** Markdown: the session path is phase 0 → 1.1 → 6.0, the v0.2.21 commitment in three lines, and "phase 0 output is pasted back too: the versions go in every result file". Code: `run(only="15. ...")`, and assert that `ctx` reports a pass.
  - **6.1** Gate data: reuse the cell-28 extraction logic (MVTec AD classic `.tar.xz` from `MyDrive/mvtec_ad`).
  - **6.2** `!python scripts/run_mvtec_ad.py` with no `--results` (the default is now the label). Then `!python scripts/reproduction_gate.py --results results/reproduction/patchcore_ref_v0.2.21/mvtec_ad/shards --targets configs/reproduction/patchcore_ref.yaml --out results/reproduction/patchcore_mvtec_ad_v0.2.21.md`, then print it. Check the real flag names in `reproduction_gate.py`, including how it selects seeds, against how cell 47 called it for three seeds.
  - **6.3** Copy the gate report and shards to Drive (`MyDrive/mvtec_ad_results/patchcore_ref_v0.2.21/`) immediately.
  - **6.4** Markdown: the grid does not wait on the verdict; one category per run, smallest first (the cell-51 table); re-upload each archive by hand; after the gate exists, re-run `--summarise-only` for every category so each report carries the verdict. Code: as cell 52 (`CATEGORY = "vial"`, `!python scripts/run_mvtec_ad2.py --category {CATEGORY}`).
  - **6.5** Code: a loop over the eight categories running `--summarise-only`, then `git status --short results/` so the operator sees exactly which reports to bring back.

- [ ] **Step 1: Write the failing tests** in `tests/test_notebook_run_cells.py`:
  - `test_the_v0221_session_cells_exist`: cells whose source contains `"15. score() resizes with antialiasing"`, `"patchcore_mvtec_ad_v0.2.21.md"` and `"--summarise-only"`.
  - `test_the_v0221_gate_cell_writes_to_the_new_report_not_the_old`: no phase-6 cell contains `--out results/reproduction/patchcore_mvtec_ad.md`.
  - Extend the existing "imports everything it uses" parametrisation to the phase-6 code cells, following how the M3 cell is collected.
- [ ] **Step 2: Run** `python -m pytest tests/test_notebook_run_cells.py -q`. Expected: FAIL.
- [ ] **Step 3: Edit the notebook** with a short Python script that loads and dumps the JSON (`indent=1`, matching the file's current formatting; check with `git diff --stat` that only the intended cells changed). Do not hand-edit the JSON.
- [ ] **Step 4: Run** `python -m pytest -q`. Expected: pass.
- [ ] **Step 5: Commit** `feat: notebook phase 6 — the v0.2.21 corrected-anchor session (gate + MVTec AD 2 grid)`

### Task 8: Old reports annotated, next-steps updated, pre-v0.2.21 tag

**Files:**
- Modify (one line under the H1 title, nothing else):
  - `results/reproduction/patchcore_mvtec_ad.md`
  - `results/reproduction/patchcore_mvtec_ad_3seed.md`
  - `results/reproduction/patchcore_mvtec_ad_classic.md`
  - `results/reproduction/patchcore_ref_visa.md`
  - `results/mvtec_ad2/patchcore_ref/{can,fabric,fruit_jelly,rice,sheet_metal,vial,wallplugs,walnuts}.md`
- Modify: `docs/next-steps.md` (top banner + the check (b) section's "Owed" sentence)
- Test: none new. Verification is by diff.

- [ ] **Step 1: Add the header lines, exact copy:**
  - Re-run files (the two gate reports and the eight MVTec AD 2 reports): `> **2026-10-19 — superseded configuration.** Scored through anomalib's non-antialiased export path against an antialiased memory bank (check (b), docs/next-steps.md). Kept as evidence; the protocol v0.2.21 re-run is \`<new path>\`.`
    - `<new path>` for the two gate reports: `results/reproduction/patchcore_mvtec_ad_v0.2.21.md`.
    - `<new path>` for each MVTec AD 2 report: `results/mvtec_ad2/patchcore_ref_v0.2.21/<category>.md`.
  - VisA: the same sentence ending `Affected and not re-run (author's scope, 2026-10-19); cite only as old-configuration evidence, never as an anchor number.`
  - Classic: the same, adding `It was also crop-mismatched (H2), so it does not refute CenterCrop.`
- [ ] **Step 2: Verify** `git diff --stat results/` shows 12 files, each `1 insertion` plus a blank line at most, and `git diff results/ | grep '^-' | grep -v '^---'` prints nothing.
- [ ] **Step 3: Update next-steps.** The banner gets: v0.2.21 implemented on this branch; the next step is the Colab session (phase 0 → 1.1 → 6.0–6.5); archives may need re-uploading. The check (b) section's "Owed" line gets: decided 2026-10-19, see protocol v0.2.21.
- [ ] **Step 4: Tag** the last pre-v0.2.21 code state: `git tag -a patchcore-pre-v0.2.21 f57c9a3 -m "Last code state whose PatchCore backend scored through the non-antialiased export path"`. **Ask the author before `git push origin patchcore-pre-v0.2.21`.**
- [ ] **Step 5: Run** `python -m pytest -q`. Expected: pass. Commit `docs: annotate superseded PatchCore reports; next-steps points at the v0.2.21 session`.

---

## After the plan (not tasks here)

1. Whole-branch review, then a `--no-ff` merge to master and a push, so cell 1.1 picks it up.
2. The Colab session. The author runs it and pastes the output back. The verified numbers go into the result files **before** anything else (memory: record provenance first).
3. Paper branch `descope-c1-c3-rewrite`: anchor numbers and `docs/protocol-sync.md` get the v0.2.21 re-check.
