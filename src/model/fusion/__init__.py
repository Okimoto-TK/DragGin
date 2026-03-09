"""Fusion modules for Module 5."""

from src.model.fusion.cross_scale_attention import CrossScaleAttention
from src.model.fusion.gated_fusion import GatedFusion, MultiScaleFusion
from src.model.fusion.micro_self_attention import MicroSelfAttention

__all__ = [
    "CrossScaleAttention",
    "MicroSelfAttention",
    "GatedFusion",
    "MultiScaleFusion",
]
