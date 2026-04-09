#!/bin/bash
# TrainWeave — EC2 Spot training bootstrap
#
# This script is pulled from S3 and run by the EC2 UserData stub injected by
# the orchestrator Lambda. All environment variables (JOB_ID, EFFECTIVE_MODEL,
# DATASET_BUCKET, DATASET_KEY, ARTIFACTS_BUCKET, CODE_BUCKET,
# AWS_DEFAULT_REGION, HF_TOKEN) are exported by the UserData script before
# this file is executed.
#
# Flow:
#   1. Set up logging (tee to file + logger → CloudWatch via journald)
#   2. Install only the LoRA/PEFT stack (Deep Learning AMI has PyTorch+CUDA)
#   3. Pull dataset from S3 (free — routes through S3 VPC gateway endpoint)
#   4. Pull train.py from S3 (also free via VPC endpoint)
#   5. Run LoRA fine-tuning
#   6. Sync final adapter to S3
#   7. shutdown -h now → InstanceInitiatedShutdownBehavior=terminate kicks in
#
# On ANY error: ERR trap fires, logs the failure, and shuts down the instance
# so it does not idle and accrue costs.

set -euo pipefail

# ── Logging ───────────────────────────────────────────────────────────────────
LOGFILE="/var/log/trainweave-bootstrap.log"
exec > >(tee -a "$LOGFILE") 2>&1

log() { echo "[trainweave $(date -u +%H:%M:%SZ)] $*"; }

# ── Terminate on any error ────────────────────────────────────────────────────
# Uses the EC2 API for termination — more reliable than shutdown -h now on
# newer AMIs where systemd may defer the halt. Falls back to shutdown if the
# metadata service or CLI is unavailable.
# Always uploads the log to S3 first so failures are diagnosable post-mortem.
_terminate() {
    local exit_code=${1:-0}
    IIDENT=$(curl -sf --max-time 5 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)

    # Upload log to S3 before terminating so it survives instance deletion
    if [[ -n "${ARTIFACTS_BUCKET:-}" && -n "${JOB_ID:-}" ]]; then
        local log_key="logs/${JOB_ID}/bootstrap.log"
        aws s3 cp "$LOGFILE" "s3://${ARTIFACTS_BUCKET}/${log_key}" \
            --region "${AWS_DEFAULT_REGION:-us-east-1}" 2>/dev/null || true
        log "Log uploaded to s3://${ARTIFACTS_BUCKET}/${log_key}"
    fi

    if [[ -n "$IIDENT" ]]; then
        log "Terminating instance $IIDENT via EC2 API (exit_code=$exit_code)"
        aws ec2 terminate-instances --instance-ids "$IIDENT" \
            --region "${AWS_DEFAULT_REGION:-us-east-1}" 2>/dev/null || true
    fi
    shutdown -h now
}
trap 'log "ERROR at line $LINENO (exit $?). Terminating to avoid idle cost."; _terminate 1' ERR

log "Bootstrap started | JOB_ID=${JOB_ID}"
log "Instance: $(curl -sf http://169.254.169.254/latest/meta-data/instance-id)"
log "Instance type: $(curl -sf http://169.254.169.254/latest/meta-data/instance-type)"
log "Availability zone: $(curl -sf http://169.254.169.254/latest/meta-data/placement/availability-zone)"

WORKDIR="/opt/trainweave"
mkdir -p "$WORKDIR"

# ── 1. Python package installation ────────────────────────────────────────────
# The AWS Deep Learning AMI (Amazon Linux 2) ships with:
#   - PyTorch (matching CUDA version)
#   - CUDA + cuDNN drivers
#   - Correct Python version
# We only install the LoRA/fine-tuning stack on top.
#
# pip traffic goes through the NAT instance (internet egress).
# This is unavoidable for PyPI packages; keep the list minimal.

log "Installing fine-tuning dependencies..."

# Activate the pre-installed PyTorch conda environment if available.
# Deep Learning AMIs ship with named environments like pytorch, pytorch_p310.
CONDA_ENV=""
if command -v conda &>/dev/null; then
    # Find the first pytorch-named environment
    CONDA_ENV=$(conda env list 2>/dev/null | awk '/pytorch/{print $1; exit}')
    if [[ -n "$CONDA_ENV" ]]; then
        log "Activating conda env: $CONDA_ENV"
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    fi
fi

pip install --quiet --upgrade pip

# Pin versions for reproducibility.
# bitsandbytes: required for 4-bit QLoRA quantization.
# trl: provides SFTTrainer which wraps Trainer with instruction-tuning helpers.
pip install --quiet \
    "peft>=0.10.0" \
    "transformers>=4.40.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.29.0" \
    "bitsandbytes>=0.43.0" \
    "trl>=0.8.6" \
    "boto3>=1.34.0"

log "Dependencies installed."

# ── 2. Pull training script from S3 ──────────────────────────────────────────
# S3 traffic stays within the VPC via the gateway endpoint — zero NAT cost.
log "Pulling train.py from s3://${CODE_BUCKET}/training/train.py"
aws s3 cp "s3://${CODE_BUCKET}/training/train.py" "$WORKDIR/train.py"

# ── 3. Pull dataset from S3 ──────────────────────────────────────────────────
# Also free via S3 VPC gateway endpoint.
DATASET_PATH="$WORKDIR/dataset.jsonl"
log "Pulling dataset from s3://${DATASET_BUCKET}/${DATASET_KEY}"
aws s3 cp "s3://${DATASET_BUCKET}/${DATASET_KEY}" "$DATASET_PATH"
log "Dataset size: $(wc -l < "$DATASET_PATH") lines"

# ── 4. HuggingFace authentication (gated models) ─────────────────────────────
# HF_TOKEN is only set for gated/private models; skip otherwise.
# Model weights download through the NAT instance (internet egress).
if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN present — authenticating with HuggingFace Hub"
    pip install --quiet "huggingface_hub>=0.22.0"
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
fi

# ── 5. Run LoRA fine-tuning ───────────────────────────────────────────────────
OUTPUT_DIR="$WORKDIR/output"
mkdir -p "$OUTPUT_DIR"

log "Starting training | model=${EFFECTIVE_MODEL}"
python "$WORKDIR/train.py" \
    --model-id        "${EFFECTIVE_MODEL}" \
    --dataset-path    "$DATASET_PATH" \
    --output-dir      "$OUTPUT_DIR" \
    --job-id          "${JOB_ID}" \
    --artifacts-bucket "${ARTIFACTS_BUCKET}"

log "Training complete."

# ── 6. Upload final adapter to S3 ────────────────────────────────────────────
# train.py also syncs checkpoints periodically during training.
# This final sync ensures the complete adapter is uploaded even if the last
# checkpoint sync was mid-epoch.
ADAPTER_S3_PATH="s3://${ARTIFACTS_BUCKET}/adapters/${JOB_ID}/"
log "Uploading adapter to ${ADAPTER_S3_PATH}"
aws s3 sync "$OUTPUT_DIR/" "$ADAPTER_S3_PATH" --exclude "checkpoint-*"

log "Adapter uploaded. Job ${JOB_ID} finished successfully."

# ── 7. Self-terminate ─────────────────────────────────────────────────────────
# _terminate uploads the log to S3 then kills the instance.
log "Job ${JOB_ID} complete. Uploading log and terminating."
_terminate 0
