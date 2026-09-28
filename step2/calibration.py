"""Read recorded F-theta camera calibration without guessing transform direction."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
from step2.geometry3d import Quaternion, Vector3, quaternion_from_mapping, vector3_from_mapping


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FThetaCalibration:
    camera_name: str
    logical_id: str
    resolution_width: int
    resolution_height: int
    shutter_type: str
    principal_point_x: float
    principal_point_y: float
    max_angle: float
    reference_poly: str
    angle_to_pixeldist_poly: tuple[float, ...]
    pixeldist_to_angle_poly: tuple[float, ...]
    rig_to_camera_translation: Vector3
    rig_to_camera_quaternion: Quaternion
    raw: dict[str, Any]

    def recorded_resolution_matches(self, width: int, height: int) -> bool:
        return self.resolution_width == width and self.resolution_height == height

    def recorded_scale(self, width: int, height: int) -> tuple[float, float]:
        return width / self.resolution_width, height / self.resolution_height


def _number_list(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise CalibrationError(f"{name} must be a non-empty list")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must contain numeric values") from exc


def calibration_from_dict(camera_name: str, data: Mapping[str, Any]) -> FThetaCalibration:
    try:
        available = data["available_camera"]
        intrinsics = available["intrinsics"]
        ftheta = intrinsics["ftheta_param"]
        transform = available["rig_to_camera"]
        result = FThetaCalibration(
            camera_name=camera_name,
            logical_id=str(available["logical_id"]),
            resolution_width=int(intrinsics["resolution_w"]),
            resolution_height=int(intrinsics["resolution_h"]),
            shutter_type=str(intrinsics["shutter_type"]),
            principal_point_x=float(ftheta["principal_point_x"]),
            principal_point_y=float(ftheta["principal_point_y"]),
            max_angle=float(ftheta["max_angle"]),
            reference_poly=str(ftheta["reference_poly"]),
            angle_to_pixeldist_poly=_number_list(ftheta["angle_to_pixeldist_poly"], "angle_to_pixeldist_poly"),
            pixeldist_to_angle_poly=_number_list(ftheta["pixeldist_to_angle_poly"], "pixeldist_to_angle_poly"),
            rig_to_camera_translation=vector3_from_mapping(transform["vec"]),
            rig_to_camera_quaternion=quaternion_from_mapping(transform["quat"]),
            raw=dict(data),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"invalid calibration for {camera_name}: {exc}") from exc
    if result.resolution_width <= 0 or result.resolution_height <= 0:
        raise CalibrationError("calibration resolution must be positive")
    return result


def read_camera_calibration(path: Path, camera_name: str) -> FThetaCalibration:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read calibration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError("calibration root must be an object")
    return calibration_from_dict(camera_name, data)
