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
    out_dir="outputs/run_motion_events_motion",
    vm_config={"fps": 2, "resize": 512, "backend": "dummy", "max_frames": 20},
    e2r_config={"mode": "motion", "threshold_multiplier": 1.0, "min_event_length": 1},
)
print("Motion-based episode:", episode_path_motion)