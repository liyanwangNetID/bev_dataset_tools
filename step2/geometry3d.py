"""Three-dimensional rigid-pose utilities for camera-only occupancy data."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Mapping

_EPS = 1e-12


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def normalized(self) -> "Quaternion":
        norm = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w)
        if norm <= _EPS:
            raise ValueError("quaternion norm must be positive")
        return Quaternion(self.x/norm, self.y/norm, self.z/norm, self.w/norm)


@dataclass(frozen=True, slots=True)
class Pose3D:
    position: Vector3
    orientation: Quaternion

    def matrix4x4(self) -> tuple[tuple[float, float, float, float], ...]:
        q = self.orientation.normalized()
        x, y, z, w = q.x, q.y, q.z, q.w
        return (
            (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), self.position.x),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), self.position.y),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), self.position.z),
            (0.0, 0.0, 0.0, 1.0),
        )


@dataclass(frozen=True, slots=True)
class TimedPose3D:
    stamp_ns: int
    pose: Pose3D
    interpolation_span_ns: int
    exact: bool


def vector3_from_mapping(value: Mapping[str, Any]) -> Vector3:
    return Vector3(_finite(value["x"], "x"), _finite(value["y"], "y"), _finite(value["z"], "z"))


def quaternion_from_mapping(value: Mapping[str, Any]) -> Quaternion:
    return Quaternion(_finite(value["x"], "qx"), _finite(value["y"], "qy"), _finite(value["z"], "qz"), _finite(value["w"], "qw")).normalized()


def quaternion_slerp(first: Quaternion, second: Quaternion, ratio: float) -> Quaternion:
    t = _finite(ratio, "ratio")
    if not 0.0 <= t <= 1.0:
        raise ValueError("ratio must be within [0, 1]")
    a, b = first.normalized(), second.normalized()
    dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w
    if dot < 0.0:
        b = Quaternion(-b.x, -b.y, -b.z, -b.w)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return Quaternion(a.x+t*(b.x-a.x), a.y+t*(b.y-a.y), a.z+t*(b.z-a.z), a.w+t*(b.w-a.w)).normalized()
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    wa = math.sin((1.0-t)*theta) / sin_theta
    wb = math.sin(t*theta) / sin_theta
    return Quaternion(wa*a.x+wb*b.x, wa*a.y+wb*b.y, wa*a.z+wb*b.z, wa*a.w+wb*b.w).normalized()


def interpolate_pose3d(first: Pose3D, second: Pose3D, ratio: float) -> Pose3D:
    t = _finite(ratio, "ratio")
    if not 0.0 <= t <= 1.0:
        raise ValueError("ratio must be within [0, 1]")
    return Pose3D(
        Vector3(first.position.x+t*(second.position.x-first.position.x), first.position.y+t*(second.position.y-first.position.y), first.position.z+t*(second.position.z-first.position.z)),
        quaternion_slerp(first.orientation, second.orientation, t),
    )
