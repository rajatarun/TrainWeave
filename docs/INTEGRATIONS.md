# Integration opportunities

Where TrainWeave can plug into ecosystems developers already use. **Fit** is
relative to this repository today (Lambda + EC2 LoRA orchestration).

| Integration | Fit | What an adapter would do | Effort | Implemented |
|-------------|-----|--------------------------|--------|-------------|
| **AWS SAM / CloudFormation** | High | Package and deploy stack; already the source of truth | Low | Yes (`template.yaml`, `sam deploy`) |
| **GitHub Actions** | High | OIDC deploy + async Lambda invokes | Low | Yes (`.github/workflows/deploy.yaml`) |
| **DeployWeave adapter catalogue** | High | Post-train DynamoDB registration + S3 URI | Medium | Yes (`register_adapter.py`, template parameter) |
| **TeamWeave DPO bucket** | High | `dataset_source.type: dpo` materialisation | Medium | Yes (`dpo_dataset.py`, orchestrator) |
| **Amazon Bedrock custom models** | Med | Map finished adapters to Bedrock import / custom model APIs instead of only external resolver | High | No — catalogue item is generic; Bedrock attach is downstream |
| **Amazon SageMaker Training** | Med | Alternative backend implementing same job event schema | High | No — design chose EC2 Spot |
| **Hugging Face Hub** | High | Push adapters as HF repos; pull base models (bootstrap already uses HF) | Medium | Partial (download at train time; no Hub upload) |
| **OpenTelemetry** | Med | Trace Lambda invoke → instance launch → train steps via OTEL on Lambda + EC2 | Medium | No |
| **MLflow / W&B** | Med | Log hyperparameters and metrics from `train.py` | Medium | No — only S3 run config artifact |
| **LangChain / LangGraph** | Low | Chain nodes that trigger training after eval; not a natural in-process library | Medium | No |
| **MCP servers** | Low | Expose "start training job" as MCP tool wrapping Lambda invoke | Small | No |
| **Anthropic / OpenAI batch APIs** | Med | Use judge APIs in `audit_dpo_labels.py` / `eval_reliability.py` with pluggable providers | Medium | Partial (Bedrock/Anthropic via scripts, not abstracted) |
| **Kubernetes / Argo Workflows** | Med | Replace Lambda launcher with Workflow that schedules GPU pods | High | No |

## Recommended next adapters (top 3)

### 1. Hugging Face Hub upload (fit: **high**)

After `bootstrap.sh` uploads to S3, a optional step could `huggingface-cli upload`
the adapter folder using a repo id from the Lambda event. **Effort:** ~2–3 days
with tests mocked; needs HF token handling consistent with existing `hf_token`.

### 2. OpenTelemetry spans (fit: **medium**)

Emit spans for `handler`, `_materialise_dpo_dataset`, `RunInstances`, and bootstrap
phases. **Effort:** ~2 days for Lambda + document EC2 sidecar/agent pattern.

### 3. MCP job trigger (fit: **medium** for teams already on MCP)

Thin MCP server: `start_lora_job(dataset_name, ...)` → `lambda.invoke`. **Effort:**
~1 day; no change to core orchestrator.

## Local offline integration

`scripts/convert_dpo_local.py` integrates **without AWS** for dataset prep —
useful in notebooks or CI before uploading JSONL to S3.
