# scripts/run_integrated_pipeline.py

from conceptops.pipelines.integrated_pipeline import process_video_to_dataset

# FIXED-WINDOW EVENTS
episode_path_fixed = process_video_to_dataset(
    video_path="examples/desk_demo.mp4",
    out_dir="outputs/run_fixed_events",
    vm_config={"fps": 2, "resize": 512, "backend": "dummy", "max_frames": 20},
    e2r_config={"mode": "fixed", "frames_per_event": 5, "base_label": "event"},
)
print("Fixed-window episode:", episode_path_fixed)

# MOTION-BASED EVENTS
episode_path_motion = process_video_to_dataset(
    video_path="examples/desk_demo.mp4",
    out_dir="outputs/run_motion_events",
    vm_config={"fps": 2, "resize": 512, "backend": "dummy", "max_frames": 20},
    e2r_config={"mode": "motion", "threshold_multiplier": 1.0, "min_event_length": 1},
)
print("Motion-based episode:", episode_path_motion)

# MOTION-MASK EVENTS
episode_path_motion_mask = process_video_to_dataset(
    video_path="examples/desk_demo.mp4",
    out_dir="outputs/run_motion_mask_events",
    vm_config={"fps": 2, "resize": 512, "backend": "dummy", "max_frames": 20},
    e2r_config={
        "mode": "motion_mask",
        "threshold_multiplier": 1.5,
        "min_event_length": 2,
        "min_area_ratio": 0.05,
    },
)
print("Motion-based episode:", episode_path_motion_mask)

# MODEL
episode_path_model = process_video_to_dataset(
    "examples/desk_demo.mp4",
    "outputs/run_model_events",
    vm_config={"fps": 2, "resize": 512, "backend": "dummy", "max_frames": 20},
    e2r_config={"mode": "model", "model_name": "stub_v1"},
)

print("Model-based episode:", episode_path_model)