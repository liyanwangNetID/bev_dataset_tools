from __future__ import annotations
import math
from step2.calibration import calibration_from_dict
from step2.geometry3d import Pose3D, Quaternion, Vector3, interpolate_pose3d, quaternion_slerp


def test_slerp_and_pose_matrix() -> None:
    first = Quaternion(0, 0, 0, 1)
    second = Quaternion(0, 0, 1, 0)
    middle = quaternion_slerp(first, second, 0.5)
    assert math.isclose(middle.z, math.sqrt(0.5), abs_tol=1e-9)
    pose = interpolate_pose3d(Pose3D(Vector3(0,0,0), first), Pose3D(Vector3(2,4,6), second), 0.5)
    assert pose.position == Vector3(1,2,3)
    matrix = pose.matrix4x4()
    assert matrix[0][3] == 1 and matrix[1][3] == 2 and matrix[2][3] == 3


def test_calibration_resolution_mismatch_is_explicit() -> None:
    data = {"available_camera": {"logical_id": "camera", "intrinsics": {"resolution_w": 1920, "resolution_h": 1080, "shutter_type": "ROLLING_TOP_TO_BOTTOM", "ftheta_param": {"principal_point_x": 960, "principal_point_y": 540, "max_angle": 1.3, "reference_poly": "PIXELDIST_TO_ANGLE", "angle_to_pixeldist_poly": [0, 900], "pixeldist_to_angle_poly": [0, 0.001]}}, "rig_to_camera": {"vec": {"x": 1, "y": 2, "z": 3}, "quat": {"x": 0, "y": 0, "z": 0, "w": 1}}}}
    calibration = calibration_from_dict("front_wide", data)
    assert not calibration.recorded_resolution_matches(854, 480)
    sx, sy = calibration.recorded_scale(854, 480)
    assert math.isclose(sx, 854/1920)
    assert math.isclose(sy, 480/1080)
