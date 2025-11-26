from __future__ import annotations

"""
videomask package

Core public API surface is the VideoSegmenter.
"""

from .pipeline.segmenter import VideoSegmenter

__all__ = ["VideoSegmenter"]