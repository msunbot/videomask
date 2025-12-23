# VideoMask SDK Progress  
*ConceptOps Phase 1 — SAM-3 powered programmatic video segmentation*  
*Started: Nov 25, 2025*

## Week 1: Foundations & Model Integration (Days 0–4 Complete)

### Day 0 (Nov 25) — Repo & Environment Setup ✅
- [x] Created full Python package structure (`videomask/`)
- [x] Added modules: `core/`, `backends/`, `pipeline/`, `exporters/`, `examples/`, `tests/`)
- [x] Added `pyproject.toml` and enabled editable install
- [x] Installed ffmpeg on macOS  
**Result:** Clean SDK skeleton ready

---

### Day 1 (Nov 25) — Frame Extraction + Dummy Backend + Segmenter Pipeline ✅
- [x] Implemented `extract_frames_ffmpeg()` (reliable across OS)
- [x] Added `DummyBackend` (fast dev mask generator)
- [x] Implemented `VideoSegmenter` orchestration pipeline
- [x] Validated with `day1_test.py`  
**Result:** Video → frames → dummy masks → dataset (end-to-end working)

---

### Day 2 (Nov 25) — Temporal Smoothing + Dataset Exporter + CLI Tooling ✅
- [x] Added IoU-based temporal smoothing (`smooth_masks_sequence`)
- [x] Added folder-format dataset exporter (`frames_raw/`, `masks/`, `metadata.json`)
- [x] Implemented CLI (`videomask segment ...`)
- [x] Verified pipeline correctness  
**Result:** Usable SDK + CLI for dataset generation

---

### Day 3 (Nov 25) — SAM-3 Backend Scaffolding + Colab Setup Plan ✅
- [x] Implemented `SAM3Backend` skeleton with lazy loading
- [x] Integrated backend selection into `VideoSegmenter`
- [x] Established Colab GPU workflow for SAM-3 integration  
**Result:** Architecture ready for real SAM-3 inference

---

### Day 4 (Nov 25) — SAM-3 GPU Prototype ✅
- [x] Installed CUDA PyTorch in Colab; resolved NCCL errors
- [x] Installed SAM-3 from GitHub
- [x] Logged into Hugging Face; weights download works
- [x] Confirmed `build_sam3_image_model()` loads successfully
- [x] Ran text-prompt inference (`"person"`)
- [x] Observed multiple masks returned; selected highest-score mask
- [x] Implemented empty-output safety guard
- [x] Generated first real binary mask + overlay visualization  
**In progress:** Packaging inference logic into SDK backend

---

## Week 1 Learnings
1. ffmpeg is robust for frame extraction across OS.
2. Backend abstraction enables fast iterations.
3. SAM-3 requires CUDA-compatible PyTorch; CPU-only builds fail.
4. Model returns multiple masks → backend must define selection strategy.
5. Edge cases (empty results) must return safe zero masks.
6. Colab is ideal for prototyping GPU-dependent model code.

---

## Technical Decisions
- Folder-format dataset export for v0.1 prioritizes flexibility and iteration speed.
- Backend abstraction (`BaseSegmentationBackend`) enables future ConceptOps features.
- Temporal smoothing kept simple (IoU-based) to stay within v0.1 scope.
- SAM-3 work done in Colab; SDK consumes stable inference wrapper.

---

## Week 2 (Days 5–10): SAM-3 Integration, Testing & Release Preparation

### Day 5 (Nov 27) — Implement Real SAM3Backend Logic & Helper Functions ✅
- [x] Implemented `_safe_select_binary_mask` (GPU-agnostic unit-tested logic)
- [x] Integrated Colab-tested pipeline into `segment_frame()`
- [x] Added safe fallback for device mismatch (CPU fallback messaging)
- [x] Added `SAM3NotAvailableError` for clean UX  
**Result:** Fully functional SAM-3 backend ready for GPU environments

---

### Day 6 (Nov 27) — SAM-3 Example Script + CLI Verification (GPU) ✅
- [x] Added `examples/sam3_example.py`
- [x] Verified example <-> CLI consistency on GPU
- [x] Generated `outputs/sam3_example/` dataset from real SAM-3 run  
**Result:** Demonstrated complete SAM-3 → dataset flow

---

### Day 7 (Nov 27) — Unit Tests + Selection Logic Tests (CPU) ✅
- [x] Added tests for mask-selection logic using fake tensors
- [x] Verified behavior: empty masks, thresholding, picking highest score
- [x] All tests passed locally (`pytest`)  
**Result:** Stable core logic independent of GPU availability

---

### Day 8 (Nov 27) — README + Docs Cleanup (Copy-Paste Safe MD) ✅
- [x] Created clean `README.md` (no nested code fences)
- [x] Updated `TECHNICAL_DESIGN.md` (Markdown-confirmed)
- [x] Updated `CHANGELOG.md` and `PROGRESS.md`
- [x] Standardized directory naming and examples  
**Result:** Public-facing documentation ready for v0.1

---

### Day 9 (Nov 27) — GitHub Cleanup, .gitignore, Repo Reset & Push Fixes ✅
- [x] Removed large mistakenly committed virtualenv
- [x] Reset `.git` to clean history
- [x] Added `.gitignore` for venv, outputs, OS artifacts
- [x] Successfully pushed clean repo to GitHub  
**Result:** Repo ready for public release and contributions

---

### Day 10 (Nov 27) — Final Validation & v0.1 Release Prep 🎉
- [x] Ran local smoke tests (dummy backend)
- [x] Ran GPU tests (SAM-3 backend)
- [x] Confirmed code style, comments, file structure
- [x] Verified outputs (frames, masks, metadata)
- [x] Prepared repo for launch announcement  
**Result:** VideoMask SDK v0.1 complete and ready to ship

---

# Week 2 Summary
**Completed:**  
- SAM-3 backend (functional)  
- Example scripts (dummy + SAM-3)  
- Full documentation suite (README, DESIGN, ROADMAP, CHANGELOG)  
- GPU & CPU testing  
- Repo cleanup and push  
- v0.1 release readiness  

**Next Week (Week 3):**  
- Public launch tweet/blog  
- COCO export (optional for v0.1.1)  
- Early adopter outreach  

---
## Week 3 (Days 11–20): ConceptOps MVP — Full Pipeline, Events, Concepts, Episode Export

### Day 11–12 (Dec 5) — ConceptOps CLI Integration + Manifest Flow ✅
- [x] Added unified `conceptops run` CLI: video → masks → events → concepts → episode  
- [x] Added tuning flags (`--event-iou-threshold`, `--event-min-length`, `--labels`, `--concept-top-k`)  
- [x] Integrated manifest-driven multi-stage pipeline (mask → event → concept → episode)  
**Result:** One-command end-to-end ConceptOps pipeline

---

### Day 13–15 (Dec 5) — Temporal Event Segmentation (IoU + Motion Heuristics) ✅
- [x] Implemented robust event segmentation (IoU + centroid + area motion)  
- [x] Added fallback: always ≥ 1 event when masks exist  
- [x] Implemented event normalization (contiguous event_ids: 0..N-1)  
- [x] Added duplicate segment removal + min-length filtering  
**Result:** Stable temporal event extraction suitable for MVP

---

### Day 16–17 (Dec 5) — CLIP Concept Tagging + Thumbnails + Uncertainty Scoring ✅
- [x] Integrated CLIP concept tagging with customizable vocab  
- [x] Added per-event thumbnails  
- [x] Added `top_score` + `uncertain=True` flag for low-confidence cases  
- [x] Linked concept stage into CLI + manifest updates  
**Result:** Semantic labels + visual thumbnails for each event

---

### Day 18 (Dec 5) — LeRobot Episode Exporter + Manifest Flow 🎉  
- [x] Implemented `episode.json` builder from frames, masks, events, concepts  
- [x] Normalized relative paths for portability  
- [x] Added episode stage to CLI + manifest updates  
**Result:** Robot-ready structured episode export

---

### Day 19 (Dec 5) — SAM-3 GPU Integration in Colab + HF Auth + Runtime Fixes 🎉  
- [x] Set up full SAM-3 backend in Colab (HF login, vocab fix, asset path fix)  
- [x] Ran full ConceptOps pipeline with `backend=sam3`  
- [x] Validated masks, IoUs, event splits, CLIP labels, thumbnails  
**Result:** First real ConceptOps hero run (SAM-3 powered)

---

### Day 20 (Dec 5) — Visualization Notebook + Analysis Workflow 🎉  
- [x] Built demo notebook for visual inspection  
- [x] Added event overview, label summaries, thumbnails, flipbooks  
- [x] Validated notebook workflow for hero clip  
**Result:** Polished MVP demo tooling for README + launch

---

## Week 3 Summary
**Completed:**  
- Full ConceptOps MVP pipeline  
- Event segmentation (IoU + centroid + area)  
- CLIP concept tagging (thumbnails + uncertainty)  
- LeRobot episode export  
- SAM-3 GPU flow (Colab)  
- Demo notebook (end-to-end visualization)

**Next:**  
- README upgrade + diagrams  
- Add example video(s)  
- Publish repo + tweet thread + demo notebook  
- Optional: small UX polish pre-launch

## Week 4 (Days 21–23): Phase 2 Integrated Pipeline (ConceptOps + VideoMask + Ego2Robot v0.5)

### Day 21 (Dec 9) — Ingestion Module + Integrated Entry Point ✅
- [x] Designed 4-layer architecture for Phase 2:
      1) Ingestion, 2) Perception (VideoMask), 3) Events (Ego2Robot),
      4) Orchestration & Export (ConceptOps).
- [x] Implemented `conceptops.ingestion.ingest_video()`:
      - ffprobe-based metadata probing
      - ffmpeg frame extraction → `frames_raw/`
      - returns `VideoIngestResult` (frame paths + canonical `VideoMetadata`)
- [x] Added `process_video_to_dataset(...)` orchestration entrypoint:
      - video path + out_dir → ingestion + VideoMask run
      - writes `frames_raw/`, `masks/`, `metadata.json`
- [x] Created initial pytest for ingestion + integrated pipeline
**Result:** Clean Layer 1 + Layer 4 integration with a single Python entrypoint.

---

### Day 22 (Dec 9) — Canonical Episode Schema + JSON Export ✅
- [x] Added `conceptops/types.py` with core dataclasses:
      - `VideoMetadata`, `FrameRecord`, `EventRecord`, `Episode`
- [x] Refactored ingestion to use shared `VideoMetadata`
- [x] Extended `process_video_to_dataset(...)` to:
      - read VideoMask `metadata.json`
      - build `FrameRecord` list (frame + mask + timestamps)
      - construct `Episode` and write `episode.json`
- [x] Implemented `Episode.to_dict()` / `Episode.to_json()` helpers
- [x] Updated tests to assert existence + basic structure of `episode.json`
**Result:** End-to-end `video → masks → episode.json` pipeline using canonical schemas.

---

### Day 23 (Dec 9) — Event Detectors + Episode Round-Trip + Export Hooks ✅
- [x] Implemented `SimpleEventDetector` (fixed-window segmentation) with `SimpleEventConfig`
- [x] Implemented `MotionEventDetector` (mean pixel diff–based motion) with `MotionEventConfig`
- [x] Wired event detectors into `process_video_to_dataset(...)` via `e2r_config`:
      - `mode="fixed"` → windowed segments
      - `mode="motion"` → motion-based segments
- [x] Added `Episode.from_dict()` / `Episode.from_json()` for round-trip loading
- [x] Wrote round-trip tests:
      - in-memory Episode → JSON → Episode
      - pipeline `episode.json` → `Episode.from_json`
- [x] Added LeRobot export surface (`episode_to_lerobot(...)`) for future RLDS integration
- [x] Confirmed:
      - `pytest` passes for ingestion + pipeline + round-trip tests
      - `scripts/run_integrated_pipeline.py` produces episodes with frames + events
**Result:** Phase 2 v0.5 integrated pipeline:
`video → frames → masks → events → Episode (JSON)` with loaders, tests, and an export hook.

---

## Next (Week 4 + Phase 3 Start)

**Week 4 — Perception Layer Polish (VideoMask)**  
- [ ] Temporal mask smoothing improvements (reduce flicker across frames)  
- [ ] Multi-object mask prototype:
      - multiple instances per frame
      - initial heuristics for object selection  
- [ ] Stable object IDs across frames:
      - simple IoU/centroid tracking over time  
- [ ] Basic mask quality metrics:
      - per-frame mask area / coverage
      - simple confidence heuristics for filtering bad masks  

**Phase 3 (first half) — Stronger Semantics & Multi-Object Support**  
- [ ] Upgrade segmentation backends:
      - plug in real SAM-3 / EfficientSAM / HQ-SAM under a common interface  
- [ ] Extend `FrameRecord` / Episode schema for multi-object:
      - support list of instance masks / IDs per frame  
- [ ] Improve event semantics:
      - refine motion-based detector with mask-aware features
      - prepare path toward primitive events (reach, grasp, move, place) via Ego2Robot  

**Integration & UX**  
- [ ] Add small visualization notebook:
      - show frames + masks + event spans from `episode.json`  
- [ ] Add CLI flag to select event mode:
      - `--event-mode fixed|motion`  
- [ ] Keep `process_video_to_dataset(...)` as the canonical Python entrypoint
      and ensure tests stay green as perception and events evolve.

# Week 4 (Days 24-28): Perception Polish & Phase 3 Foundations  
**Date: Dec 11, 2025**

## Day 24 (Dec 11) — Multi-Object Schema + Episode Infrastructure Upgrade ✅
- [x] Added `InstanceMask` dataclass (Phase 3 foundational type)  
- [x] Extended `FrameRecord` with `instances: List[InstanceMask]`  
- [x] Preserved backward compatibility with legacy `mask_path`  
- [x] Updated `Episode.from_dict` with deep nested dataclass reconstruction  
- [x] Verified `episode.json` round-trip load/save remains consistent  
**Result:** Episode schema now fully supports multi-object perception.

---

## Day 25 (Dec 11) — Mask Quality Metrics + Perception Enhancement Layer ✅
- [x] Created `MaskStats` + `compute_mask_stats(mask_path)`  
- [x] New module: `conceptops/perception/mask_metrics.py`  
- [x] Pipeline now computes `area_px` & `area_ratio` for every mask  
- [x] Per-frame `mask_quality` added to metadata  
- [x] Instances enriched with per-mask stats; multi-mask support generalized  
**Result:** Each frame now carries meaningful mask diagnostics for downstream ML.

---

## Day 26 (Dec 11) — Multi-Object Ready Frame Builder + Integration Tests ✅
- [x] `_build_frame_records` now handles:
  - `masks=[str, str, ...]` → single-object
  - `masks=[[m1, m2, ...], [...]]` → multi-object  
- [x] Per-frame `instances[...]` populated with metrics  
- [x] `mask_quality` aggregates: max/min/mean area ratios  
- [x] Updated integrated pipeline tests to ensure schema correctness  
**Result:** Perception layer is now structurally ready for multi-object SAM-3 outputs.

---

## Day 27 (Dec 11) — Export Formats (LeRobot / RLDS / COCO) + Testing Suite ✅
- [x] Implemented `episode_to_lerobot(episode)`  
- [x] Implemented `episode_to_rlds(episode)`  
- [x] Implemented full COCO exporter (`export_coco_from_episode`)  
- [x] Added bbox extraction from masks and COCO-style annotation builder  
- [x] Created pytest suite for:
  - Model detector  
  - Exporters  
  - COCO serialization  
**Result:** ConceptOps can now export robot datasets in three major formats.

---

## Day 28 (Dec 11) — ModelEventDetector Scaffold + Mode="model" Integration + Inspection Notebook ✅  
- [x] Added `ModelEventConfig` + `ModelEventDetector` scaffold  
- [x] Integrated new detector into `process_video_to_dataset` (`mode="model"`)  
- [x] Stub implementation delegates to motion detector + metadata tagging  
- [x] Added pytest confirming “model_stub” event metadata  
- [x] Created developer notebook template for:
  - loading episodes  
  - RLDS + LeRobot inspections  
  - COCO bbox visualization  
**Result:** Phase 3 model slot now implemented; export + inspection tools complete.

---

# Week 4 Summary  
**Completed:**
- Multi-object perception schema + mask metrics  
- Enhanced Episode & FrameRecord data model  
- Full export stack (LeRobot / RLDS / COCO)  
- Model-based event detector placeholder  
- Integrated tests + inspection notebook template  

**Next (Phase 3: Weeks 5-7):**
- Collect labeled clips for event taxonomy  
- Manual annotation via `event_labels.json`  
- Train event model v1  
- Add affordance classifier (graspable, pushable, rotateable)  
- Add Δpose / motion vectors  
- Strengthen COCO exporter with polygons once SAM-3 masks are real  
- Build dataset viewer notebook + CLI exports  

# Week 5 (Days 29-31): Phase 3 - "Pretrained Action Module + Export Formats"
**Date: Dec 15, 2025**
*Goal:* Move from heuristic-only events to a credible, research-grade action/event module with labeling, training, inference, and evaluation — fully integrated into the existing pipeline.

## Day 29 (Dec 15, 2025) — Data + Labeling Workflow (Foundation) ✅
- [x] Added a canonical labeling artifact: `event_labels.json` stored alongside each `episode.json`
- [x] Adopted a canonical taxonomy file (repo-local): `conceptops/config/event_taxonomy.json`
- [x] Implemented label I/O and conversion layer:
  - `conceptops/labeling/` (schemas + io)
  - Validates taxonomy version and frame bounds
  - Converts labeled spans → canonical `EventRecord` (schema-aligned: `event_id`, `label`, `start_frame`, `end_frame`, `score`, `metadata`)
- [x] Added a minimal labeling tool:
  - `scripts/label_episode_events.py`
  - Frame inspection (OpenCV), mark start/end, add label, save JSON
- [x] Added batch episode builder:
  - `scripts/batch_build_episodes.py` generates `data/episodes/<clip_id>/episode.json` etc.
**Result:** Labeling is now possible and understandable with portable, diffable artifacts.

---

## Day 30 (Dec 15, 2025) — Training + Inference Loop (Baseline) ✅
- [x] Implemented training scaffolding:
  - `conceptops/training/dataset.py` (load episodes + labels, resilient to bad/empty JSON)
  - `conceptops/training/features.py` (7-dim handcrafted features from mask stats + frame diffs)
  - `conceptops/training/model.py` (Tiny MLP classifier)
  - `scripts/train_event_model.py` produces:
    - `data/models/event_model_v0/model.pt`
    - `labels.json`
    - `feature_spec.json`
- [x] Implemented standalone inference runner:
  - `scripts/run_event_model_inference.py`
  - Sliding-window proposals + scoring → `pred_events.json`
- [x] Added single-episode evaluation:
  - `conceptops/eval/metrics.py` (temporal IoU, greedy 1-1 matching, P/R/F1)
  - `scripts/eval_event_model.py`
**Result:** End-to-end loop works: labeled spans → trained artifact → predictions → evaluatable metrics.

---

## Day 31 (Dec 15, 2025) — Pipeline Integration + Batch Evaluation + Robustness ✅
- [x] Resolved taxonomy path ambiguity:
  - scripts accept `config/event_taxonomy.json` and fall back to `conceptops/config/...` when needed
- [x] Fixed pipeline integration bug (no circular dependency):
  - event detection runs on `frame_records` (Episode not constructed yet)
  - `process_video_to_dataset` now builds `Episode` after `events` are computed
- [x] Implemented real `ModelEventDetector` inference (no stub):
  - Loads `model.pt + labels.json + feature_spec.json`
  - Generates proposals, extracts features, scores spans, outputs canonical `EventRecord[]`
  - Writes predictions into `episode.json.events`
- [x] Added inference controls:
  - `min_score` threshold (lowered to `0.0` temporarily for weak early models)
  - temporal NMS-style dedup (IoU-based) to reduce overlapping duplicate outputs
- [x] Upgraded proposal strategy:
  - kept sliding windows for coverage
  - added motion-guided proposals from mask area-change peaks (metadata-driven, cheap)
- [x] Added training-side overlap handling:
  - dominant-label-per-frame normalization to convert overlapping spans → disjoint spans for training
  - optional per-frame training mode scaffold (planned; enabled when label volume is sufficient)
- [x] Added batch evaluation:
  - `scripts/batch_eval_event_model.py` compares `episode.json.events` vs `event_labels.json`
  - micro-averaged P/R/F1 across labeled clips
- [x] Confirmed end-to-end Phase 3 wiring on a labeled clip:
  - `episode.json.events` contains model-produced event(s) with scores and metadata (`source: model`)
  - batch eval runs and produces metrics (example: P=1.0, R=0.5, F1=0.67 on 1 clip)
**Result:** Model-backed event detection runs through the canonical pipeline entrypoint and is evaluatable in batch.

---

## Phase 3 Status Checklist (Done Definition)

### Labeling ✅
- [x] Can label clips with a clear workflow (`event_labels.json`, taxonomy, tool)
- [ ] Label 8–12 clips (target: 30–60+ spans across 3–5 labels) to reach credible model performance

### Training ✅
- [x] Can train baseline model from labeled clips and save artifacts
- [x] Training tolerates messy / partial episode directories

### ModelEventDetector ✅ (Wired to real model)
- [x] Loads trained artifacts
- [x] Produces `EventRecord[]` with scores and provenance metadata
- [x] Integrated into `process_video_to_dataset(... e2r_config={"mode":"model", ...})`

### Evaluation ✅
- [x] Temporal IoU + greedy one-to-one matching
- [x] Per-episode evaluation script
- [x] Batch evaluation script (micro-averaged metrics)

### Remaining for “Phase 3 done” (quality + completeness)
- [ ] Label enough real clips to make evaluation meaningful (8–12 clips / 30–60+ spans)
- [ ] Improve model beyond baseline as needed:
  - optional per-frame training mode (once label volume grows)
  - better features / representations (future)
- [ ] Ensure tests remain green after new detector wiring (run full pytest suite)
- [ ] (Optional) Add a “label coverage report” (counts per label, overlaps, span lengths) to guide labeling

---

# PHASE 4 — “DEMO + DASHBOARD” (Weeks 11–13)
*Goal:* Turn the CLI pipeline into a publicly consumable demo that looks like a YC batch-ready product.  

## Week 11 
- [ ] Build lightweight dashboard (Streamlit/Gradio)
- [ ] Video upload → pipeline run → visualizations

## Week 12
- [ ] Add segmentation playback
- [ ] Add action timeline display
- [ ] Add episode slices
- [ ] Add dataset export button

## Week 13
- [ ] Polish UI
- [ ] Add loading states + confidence metrics
- [ ] Add “download dataset as zip”

**Result:** Prototype becomes a product-grade demo.

---

# PHASE 5 — “READY FOR LAUNCH” (Weeks 14–15)
## Deliverables
1) **Essay V2 (polished + backed by results)**
- [ ] Update early essay with:
  - real visuals
  - real examples
  - real dataset exports
  - real action sequences
- [ ] Use as definitive outreach artifact (labs, founders, NVIDIA, investors, social)

2) **Launch Video (30–45 seconds)**
- [ ] Screen recording demo
- [ ] Pipeline steps
- [ ] Visual “beauty shots”

3) **Launch GTM**
- [ ] Website landing page
- [ ] Tweet thread
- [ ] Open-source repos
- [ ] Demo link

---

## Day 32 (Dec 16, 2025) — Large-Scale Labeling + Dataset Maturation 🔁
- [x] Labeled **16 real video clips** using the manual labeling workflow:
  - `scripts/label_episode_events.py`
  - Canonical `event_labels.json` stored alongside each `episode.json`
- [x] Expanded labeled dataset to **39 action spans across 7 labels**:
  - `close, move, open, pick, place, pour, wipe`
- [x] Validated labeling integrity:
  - All labeled clips load correctly
  - No schema or frame-bound errors
- [x] Re-trained baseline event model on full labeled dataset:
  - `data/models/event_model_v1/`
  - Confirmed artifact generation: `model.pt`, `labels.json`, `feature_spec.json`
- [x] Re-ran full pipeline in `mode=model` using v1 artifacts:
  - Predictions written into `episode.json.events`
- [x] Ran batch evaluation (label-match + span-only):
  - Span-only micro F1 ≈ **0.32** → confirms proposal/timing signal exists
  - Label-match metrics low → classifier is current bottleneck (expected at this scale)

**Result:** Phase 3 data loop is fully exercised on real labeled data. System is now data-limited (not wiring-limited), with clear next levers identified for quality improvement.

---

## Day 33 (Dec 17, 2025) — Phase 3 Quality Polish: Demo-Clean Inference + Eval Loop ✅

- [x] Shipped Phase 3 “quality + credibility” improvements (no new parallel systems):
  - `scripts/report_label_coverage.py` now correctly reads `event_labels.json` schema (`labeled_events`, `start_frame_idx/end_frame_idx`)
  - `scripts/eval_dashboard_summary.py` produces stable batch metrics and supports:
    - **span-only** vs **label-match**
    - **label collapsing** via `--label_collapse demo3` (7 labels → 3 buckets)
- [x] Diagnosed the core failure mode: taxonomy mismatch + stale predictions:
  - GT labels: `close/move/open/pick/place/pour/wipe`
  - Early model artifacts were not aligned with GT → label-match was guaranteed to fail
  - Mixed historical predictions (`segment_*`) were polluting eval until predictions were regenerated consistently
- [x] Implemented demo taxonomy training (3 labels) and reproducible prediction generation:
  - Added label collapsing support to training data flow (`label_collapse=demo3`)
  - Trained demo3 model:
    - `data/models/event_model_demo3/` → labels: `manipulate/move/toggle`
  - Added `scripts/batch_predict_events.py`:
    - Regenerates `pred_events.json` for every episode using a single model directory
    - Enables consistent, reproducible batch eval
- [x] Improved classifier stability with weighted loss:
  - Trained weighted demo3 model:
    - `data/models/event_model_demo3_wt/`
    - class weights prevent “predict one class only” collapse
- [x] Finalized **demo-clean inference profile** and enforced non-overlapping spans:
  - Added `demo_clean_v2` inference profile in `conceptops/core/events.py`
  - Guarantee: predicted spans are **non-overlapping** (touching boundaries allowed), matching GT expectations
  - Demo-clean results (batch, 16 episodes / 42 GT spans):
    - Total predicted spans: **26** (demo-clean density)
    - Span-only micro F1 @ IoU=0.30: **0.235**
    - Collapsed label-match micro F1 (demo3): **0.059**
  - Interpretation:
    - Timing/proposal quality is now credible and demo-friendly
    - Coarse classification is improving but remains the main bottleneck

**Result:** Phase 3 is now “research-grade credible”: stable taxonomy + reproducible train/infer/eval loop, demo-clean outputs, and measurable improvements with a clear next lever (classification).

**Next (Phase 4 — Demo + Dashboard):**
- Build Streamlit/Gradio UI that runs the pipeline and visualizes:
  - segmentation playback
  - action timeline (using demo_clean_v2 outputs)
  - slice viewer (jump to predicted spans)
  - export buttons (LeRobot / RLDS / COCO)
  - download dataset as zip

---

## Day 34 (Dec 18, 2025) — Phase 4 Demo Foundations: Wiring the Real Pipeline 🧱

- [x] Bootstrapped **Phase 4 demo layer** directly on top of the real integrated pipeline:
  - Demo now calls `process_video_to_dataset` end-to-end
  - No mock paths, no parallel demo-only pipelines
- [x] Resolved multiple pipeline/UI mismatches uncovered during integration:
  - Normalized episode directory resolution (temp vs `data/episodes/*`)
  - Eliminated stale-episode bugs caused by filename reuse
  - Added strict provenance checks between uploaded video and `episode.json.video_path`
- [x] Verified segmentation stage correctness:
  - Confirmed `frames_raw/`, `masks/`, `metadata.json` generation
  - Debugged and fixed “no frames” rendering edge cases in UI
- [x] Added **run provenance + forensic visibility**:
  - `demo_run_manifest.json`: exact inputs, configs, timing, pipeline return
  - `event_stage_status.json`: stage-by-stage artifact verification
  - Removed ambiguity around “did this stage actually run?”

**Result:** Phase 4 demo is now wired to the *real* pipeline with deterministic artifact resolution and provable execution. All remaining issues are quality/UX, not wiring.

---

## Day 35 (Dec 18, 2025) — Event Inference Integration: From Invisible to Verifiable 🎯

- [x] Diagnosed root cause of “missing events” in demo:
  - Integrated pipeline **does not write `pred_events.json` by design**
  - Event outputs live in `episode.json.events`
- [x] Fixed demo visibility gap without refactoring Phase 3:
  - Materialized `pred_events.json` from `episode.json.events` post-run
  - Ensured demo UI timeline always reflects actual event inference results
- [x] Confirmed **event inference is truly running**:
  - Verified via `episode.json.extra.event_config.mode`
  - Verified proposal provenance:
    - `proposal_method: motion_guided_from_area`
    - `inference_profile: demo_clean_v2`
- [x] Added hard verification commands to workflow:
  - `jq '.events | length' episode.json`
  - `jq '.extra.event_config' episode.json`

**Result:** Event inference is no longer a “black box” in the demo. The UI now faithfully surfaces real model outputs, with clear provenance.

---

## Day 36 (Dec 18, 2025) — Phase 4 Demo Maturation: Model-First UX + Exports 🚀

- [x] Switched demo default from heuristic → **model-based events**:
  - Event mode selector: `model | motion | fixed`
  - Demo now reflects the *actual value* of ConceptOps
- [x] Exposed **model inference controls** in the UI:
  - `topk`, `min_score`, `nms_iou`, `window_size`, `stride`
  - Enables event density tuning for demo and debugging
- [x] Restored and stabilized **export surface**:
  - COCO / RLDS / LeRobot adapters fixed
  - JSON artifacts generated correctly from `episode.json`
  - One-click download buttons + persisted files on disk
- [x] Finalized core demo UX loop:
  - Segmentation playback
  - Event timeline (demo_clean_v2)
  - Click event → slice viewer
  - Export + dataset download

**Known Limitation (Accepted for Phase 4):**
- Local runs use **dummy segmentation**, producing static masks
- This limits motion signal and often collapses events to 1 span
- Root cause is environment (no SAM-3), not model correctness

**Result:** Phase 4 demo is **functionally complete and model-backed**. Remaining gap is visual fidelity, not system correctness.

---

## Phase 4 Status (End of Day 36)

**Phase 4 is now:**
- ✅ Fully wired to the real pipeline
- ✅ Model-first (not heuristic-first)
- ✅ Export-capable
- ✅ Provenance-safe and debuggable

**Next (Phase 4.5 — Demo Wow / Launch Prep):**
- Run Streamlit on a **GPU environment with SAM-3**
- Record launch demo with dynamic masks + richer event timelines
- Capture 30–45s launch video showing:
  - real segmentation
  - multi-event timelines
  - slice navigation
  - export artifacts