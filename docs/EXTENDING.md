# Extending TrainWeave

## Extension points

### 1. New `dataset_source` types

Today only `dpo` is implemented in `app.handler`. To add another source:

1. Add conversion logic (prefer a module under `src/orchestrator/`).
2. Branch in `handler` after validating `dataset_source["type"]`.
3. Write materialised files under `datasets/{job_id}/` in the artifacts bucket.
4. Unit-test with a fake S3 client (see `tests/test_app_dpo_source.py`).

Keep IAM in `template.yaml` in sync: new sources may need extra `s3:GetObject` ARNs.

### 2. Training objectives

`objective` is `sft` (default) or `dpo`. DPO requires audit metadata. Extend
`train.py` and `bootstrap.sh` env `TRAIN_OBJECTIVE`; gate new objectives in the
Lambda the same way DPO is gated.

### 3. Adapter catalogue metadata

`register_adapter.py::build_catalog_item` accepts provenance fields. Add keys
in `app.py` when mapping `provenance` into UserData env vars (uppercased).

### 4. New sample datasets

Add `data/<name>/train.jsonl` and a workflow step in `.github/workflows/deploy.yaml`
(or invoke Lambda manually). No SAM change required if the dataset lives in an
existing bucket.

### 5. Post-training hooks

`bootstrap.sh` is the integration point after training. Fork-friendly pattern:

```bash
# After adapter upload succeeds
if [[ -n "${POST_TRAIN_HOOK:-}" ]]; then
  bash -c "$POST_TRAIN_HOOK"
fi
```

Not implemented in stock TrainWeave — add via your fork or a small PR.

## Plugin format (planned)

A stable **plugin** ABI is not defined yet. Recommended convention for
contributors:

- Pure Python transforms: functions `convert_*` returning `dpo_dataset.Conversion`
  or Alpaca rows.
- Side effects: only via AWS APIs explicitly granted in SAM.

Document new plugins in `docs/INTEGRATIONS.md`.

## Offline development

Import paths for tests:

```ini
# pytest.ini
pythonpath = src/orchestrator src/training
```

Run `python3 scripts/convert_dpo_local.py` to exercise DPO conversion without AWS.
