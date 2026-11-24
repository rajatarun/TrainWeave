"""Public package namespace for TrainWeave."""

from trainweave.adapter_bundle import (
    compute_components_checksum,
    evaluate_promotion_gate,
    export_adapter_bundle,
    verify_adapter_bundle,
)
from trainweave.wapp_adapter import (
    AdapterCompatibilityResult,
    build_wapp_adapter_sidecar,
    validate_adapter_for_wapp,
)

__all__ = [
    "AdapterCompatibilityResult",
    "build_wapp_adapter_sidecar",
    "compute_components_checksum",
    "evaluate_promotion_gate",
    "export_adapter_bundle",
    "validate_adapter_for_wapp",
    "verify_adapter_bundle",
]
