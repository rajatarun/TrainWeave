# TrainWeave examples

## Local DPO → dataset (no AWS)

Convert preference records to Alpaca SFT JSONL offline:

```bash
cd /path/to/trainweave
python3 scripts/convert_dpo_local.py \
  --input examples/dpo-records \
  --output /tmp/trainweave-demo-out
```

Inspect output:

```bash
head -n 1 /tmp/trainweave-demo-out/train.jsonl | python3 -m json.tool
```

## Lambda invoke payloads

See `lambda-payloads.json` for copy-paste event bodies:

- Standard SFT job (dataset in S3)
- DPO-derived SFT job (`dataset_source.type: dpo`)
- DPO objective after a passing label audit

Replace bucket names, prefixes, and `audit.key` with your environment.

## Sample training data

The repository ships small Alpaca JSONL files under `data/`:

| Path | Use case |
|------|----------|
| `data/customer-support/train.jsonl` | Support-style instructions |
| `data/api-json/train.jsonl` | Structured JSON outputs |
| `data/tool-calling/train.jsonl` | Tool-use style prompts |

After deploy, these sync to the stack's code bucket (see root README).
