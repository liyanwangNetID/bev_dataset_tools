import numpy as np
from bev_dataset_tools.step7.calibration import _rig_to_camera_matrix


def test_rig_to_camera_is_inverted_for_occstudio():
    raw = {"available_camera": {"rig_to_camera": {
        "quat": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        "vec": {"x": 2.0, "y": -1.0, "z": 0.5},
    }}}
    camera_to_ego = _rig_to_camera_matrix(raw)
    np.testing.assert_allclose(camera_to_ego[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(camera_to_ego[:3, 3], [-2.0, 1.0, -0.5], atol=1e-6)


def test_real_quaternion_produces_valid_rotation():
    raw = {"available_camera": {"rig_to_camera": {
        "quat": {"w": -0.49640864, "x": 0.504378, "y": -0.49575067, "z": 0.503401},
        "vec": {"x": 2.0162725, "y": -0.061019592, "z": 1.5914044},
    }}}
    matrix = _rig_to_camera_matrix(raw)
    rotation = matrix[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    assert matrix.dtype == np.float32
