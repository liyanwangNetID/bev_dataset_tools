from pathlib import Path
from step2.clip_reader import CAMERA_NAMES, DrivingClipReader


def test_real_clip_3d_pose_and_calibration_smoke():
    clip = Path("/home/lab/data_from_alpasim/test_clip_001")
    if not clip.is_dir():
        return
    reader = DrivingClipReader(clip)
    assert tuple(reader.get_all_camera_calibrations()) == CAMERA_NAMES
    diagnostics = reader.get_camera_calibration_diagnostics()
    assert all((x.recorded_width, x.recorded_height) == (854, 480) for x in diagnostics)
    assert all((x.calibration_width, x.calibration_height) == (1920, 1080) for x in diagnostics)
    common = sorted(
        set.intersection(*(set(reader.camera_indexes[name].timestamps_ns) for name in CAMERA_NAMES))
    )
    poses = (reader.get_ego_pose3d_at(timestamp_ns) for timestamp_ns in common)
    pose = next((value for value in poses if value is not None), None)
    assert pose is not None
    assert pose.interpolation_span_ns <= 200_000_000
    assert pose.ego2global[3] == (0.0, 0.0, 0.0, 1.0)
