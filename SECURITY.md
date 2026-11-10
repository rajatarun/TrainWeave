# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| `main` branch | Yes — security fixes land here first |
| Tagged releases | Only if the repository publishes tags; otherwise use `main` |

TrainWeave is infrastructure software (Lambda + ephemeral EC2). Supported
versions mean the SAM template and orchestrator code on `main`, not every
historical commit.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately by opening a **GitHub Security Advisory** (Repository →
Security → Advisories → Report a vulnerability), or contact the repository
maintainers through the channel listed in the repository profile if advisories
are not enabled.

Include:

- Affected component (orchestrator Lambda, `train.py`, bootstrap, scripts)
- Steps to reproduce
- Impact assessment (credential exposure, cross-tenant data access, etc.)

We aim to acknowledge reports within **7 days**. We may ask for a proof of
concept or suggested fix. Please allow reasonable time for a patch before
public disclosure.

## Scope notes

- **AWS credentials**: Training instances use IAM instance profiles; users must
  scope policies to their buckets and tables. Misconfigured `PassRole` or overly
  broad S3 ARNs are deployment risks, not bugs in application logic alone.
- **HF tokens**: Optional `hf_token` in Lambda events is passed to EC2 UserData;
  treat events and logs as sensitive when gated models are used.
- **DPO buckets**: The orchestrator reads arbitrary keys under a configured
  prefix; use bucket policies to restrict who can write preference records.

See `docs/design/security-notes.md` for a short threat model.
