"""Compatibility re-export of trainweave.adapter_bundle under training namespace."""

from trainweave.adapter_bundle import (
    NODE_SNAPSHOT_SCHEMA,
    TRAINWEAVE_VERSION,
    compute_components_checksum,
    evaluate_promotion_gate,
    export_adapter_bundle,
    verify_adapter_bundle,
)

__all__ = [
    "NODE_SNAPSHOT_SCHEMA",
    "TRAINWEAVE_VERSION",
    "compute_components_checksum",
    "evaluate_promotion_gate",
    "export_adapter_bundle",
    "verify_adapter_bundle",
]
