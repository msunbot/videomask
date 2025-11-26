# Changelog  
All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- SAM-3 inference prototype (Colab)
- Safe empty-output mask handling
- Initial ConceptOps roadmap

---

## [0.1.0] — Nov 25, 2025  
**VideoMask SDK (v0.1) initial release**

### Added
- Project scaffolding (`videomask/` package structure)
- Frame extraction via ffmpeg (`extract_frames_ffmpeg`)
- Backend abstraction layer (`BaseSegmentationBackend`)
- Dummy segmentation backend (`DummyBackend`)
- End-to-end segmentation pipeline (`VideoSegmenter`)
- IoU-based temporal smoothing (`smooth_masks_sequence`)
- Folder-format dataset exporter  
- CLI command: `videomask segment <input> --out <dir>`
- Colab-based SAM-3 prototyping path

### Notes
- SAM-3 backend is designed for GPU environments and is not expected to run on CPU-only machines.
- COCO export and advanced tracking are deliberately out of scope for v0.1.