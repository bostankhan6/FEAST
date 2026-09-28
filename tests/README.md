# tests/

Unit tests. Two baseline-fidelity suites plus one FEAST-own-mechanism
correctness suite, following the pattern established for HeteroFL/ScaleFL.

## Files

- **`test_heterofl.py`** — 7 pytest tests for the HeteroFL baseline: per-tier
  channel scaling, forward-pass output shapes, parameter-index slicing,
  `budget_to_tier` boundary conditions, `distribute()` value correctness,
  and server-side aggregation (`combine()`, single-client identity and
  multi-tier weighted average).
- **`test_scalefl.py`** — 7 pytest tests for the ScaleFL baseline:
  split-config MAC computation, model output shapes, parameter-index
  slicing, `budget_to_level` boundary conditions, `distribute()` value
  correctness, `combine()` identity, and the `scalefl_loss` self-distillation
  objective.
- **`test_fiarse.py`** — 7 pytest tests for the FIARSE baseline, tailored to
  its unstructured-masking design rather than mirroring the other two:
  `generate_mask()` kept-fraction accuracy, forward-pass shapes across model
  sizes, `budget_to_model_size` clipping, mask nesting (a smaller
  model_size's active parameter set is a subset of any larger model_size's,
  since TopK-by-magnitude selection is monotonic for a fixed weight
  snapshot), `distribute()` returning an independent deep copy, and
  `combine()`'s partial-averaging formula (single-client identity and
  multi-client per-position averaging).
- **`test_sub_supernet.py`** — tests FEAST's own routed sub-supernet
  mechanism directly: `GenericOFAResNet` construction with
  `per_position_max_d`, `compute_sub_supernet_bounds()`,
  `create_sub_supernet()` (param reduction and per-stage structure), weight
  copying into the sub-supernet, `add_sub_supernet()` (sparse aggregation),
  and a forward pass through the routed model. Runnable directly
  (`python tests/test_sub_supernet.py`, via its own `main()`) or under
  pytest (`server`/`sub_supernet`/`sub_info` are real `@pytest.fixture`s).
- **`test_sub_supernet_zipf_validation.py`** — pytest wrapper exercising the
  same correctness/headroom/forward-pass checks as `test_sub_supernet.py`,
  but under Zipf-sampled client budgets against a fast synthetic cache (no
  dependency on the real GA-searched cache CSV, so it's fast enough for
  CI). The full validation against the real canonical cache — which also
  reproduces the paper's cited sub-supernet communication-reduction number —
  lives in `scripts/reproduce_sub_supernet_communication_reduction.py`, not
  here; this file only imports and re-runs its correctness-check functions
  at a smaller scale.
