# Good first issues (draft for maintainers)

Concrete, scoped tasks derived from the codebase. Maintainers can paste these
into GitHub issues.

---

### 1. Add pytest workflow without deploy

**Files:** `.github/workflows/test.yml` (new)  
**Acceptance:** On pull request, `pip install -r requirements.txt pytest boto3` and `pytest -q` pass; no AWS credentials.  
**Why:** Contributors need CI that does not launch GPU jobs.

---

### 2. Parameterise shared stack name in deploy workflow

**Files:** `.github/workflows/deploy.yaml` (lines resolving `tarun-teamweave-shared`)  
**Acceptance:** Stack name comes from `vars.SHARED_STACK_NAME` or input, defaulting to current name; README documents the variable.  
**Why:** OSS adopters use their own networking stacks.

---

### 3. Redact account-specific ARN from workflow example in docs only

**Files:** `README.md`, `CONTRIBUTING.md`  
**Acceptance:** Documentation uses `arn:aws:iam::ACCOUNT_ID:role/...` placeholders; workflow file unchanged if still needed for owner's CI.  
**Why:** Safer forks without leaking account ids in copy-paste docs.

---

### 4. Hugging Face Hub upload flag (optional post-train)

**Files:** `src/training/bootstrap.sh`, `src/training/train.py` or new `upload_hf.py`, `template.yaml` (optional parameter)  
**Acceptance:** When `HF_REPO_ID` env set, adapter uploads to HF after S3 success; unit tests mock HF API; documented in `docs/INTEGRATIONS.md`.  
**Why:** High-value integration from INTEGRATIONS.md.

---

### 5. Pluggable judge backend interface

**Files:** `scripts/audit_dpo_labels.py`, `scripts/eval_reliability.py`  
**Acceptance:** `--judge-backend bedrock|anthropic|mock` with mock default in tests; existing Bedrock path unchanged.  
**Why:** Easier local testing and non-AWS adopters.

---

### 6. Document EC2 instance type matrix

**Files:** `docs/design/architecture.md`, `template.yaml` `AllowedValues`  
**Acceptance:** Table maps instance type → GPU memory → max practical model size; no invented benchmark numbers.  
**Why:** Operators choose `g4dn` vs `g5` without trial and error.

---

### 7. `dataset_source.type: local_s3_manifest` (stretch)

**Files:** `src/orchestrator/app.py`, new module, tests  
**Acceptance:** Event lists explicit S3 keys for JSONL paths; Lambda copies to artifacts bucket; tests with RecordingS3.  
**Why:** Large DPO sets where list-by-prefix is slow.

---

### 8. Improve spot interruption logging

**Files:** `src/training/bootstrap.sh`, `src/training/train.py`  
**Acceptance:** On SIGTERM or spot notice, flush checkpoint and log structured JSON line; no change to default success path tests.  
**Why:** Spot reclaim is common; operators need visibility.

---

### 9. Validate Alpaca JSONL in `data/` at test time

**Files:** `tests/test_sample_datasets.py` (new), `data/**/*.jsonl`  
**Acceptance:** Test loads each bundled dataset line as JSON with required keys `instruction`, `output`.  
**Why:** Broken sample data fails early.

---

### 10. Export catalogue item JSON schema

**Files:** `docs/schemas/adapter-catalog-item.json` (new), `tests/test_register_adapter.py`  
**Acceptance:** JSON Schema matches `build_catalog_item` output; test validates fixture against schema via `jsonschema` (dev dep).  
**Why:** DeployWeave and third-party resolvers get a contract.
