# VideoMask SDK Roadmap  
*From video-to-mask SDK → ConceptOps data engine*

---

## Phase 1 — SDK v0.1 (Current)
**Goal:** Build a functional, Python-first segmentation SDK.

### Milestones
- [x] Frame extraction with ffmpeg
- [x] Backend abstraction + dummy backend
- [x] Temporal smoothing
- [x] Dataset exporter
- [x] CLI tool
- [ ] SAM-3 backend integration
- [ ] Complete README + examples
- [ ] GitHub release

---

## Phase 2 — SDK v0.2: Advanced Backends & Formats
**Goal:** Improve segmentation quality and portability.

### Planned
- [ ] Full SAM-3 backend with configurable prompts  
- [ ] Mask selection strategies (`topk`, `union`, `threshold`)
- [ ] COCO-format export
- [ ] Optional HQ-SAM / EfficientSAM backends
- [ ] Simple visualization utilities

---

## Phase 3 — ConceptOps v1: Concept-centric Segmentation
**Goal:** Move from raw segmentation → concept-level understanding.

### Planned
- [ ] “Concept Backend” (concept detection + mask generation)
- [ ] Concept registry for labeling common industrial elements  
- [ ] Temporal “masklets” (linked concept segments across frames)
- [ ] Dataset builder for concept-supervised training

---

## Phase 4 — ConceptOps v2: Data Engine & Integration
**Goal:** Build a scalable data layer for robotics & Physical AI.

### Planned
- [ ] HF Hub integration (push datasets programmatically)
- [ ] Metadata-rich exports (concept occurrence, frequency, stability)
- [ ] Integration with LeRobot / VLA pipelines
- [ ] ConceptOps CLI (query, slice, visualize dataset recipes)

---

## Phase 5 — Platform Roadmap (Long-term)
**Goal:** Position VideoMask + ConceptOps as the “model-ready data layer” for world models & robotics.

### Directions
- Plugin architecture for community backends
- Concept graph construction (interactions, affordances)
- Autolabel pipelines for manipulation datasets
- Multi-modal support (depth, events, audio optional)

---

This roadmap is iterative — each release builds toward a fully concept-centric data stack powering Physical AI development.