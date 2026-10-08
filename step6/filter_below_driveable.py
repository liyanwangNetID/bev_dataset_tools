from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from project_paths import MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step6.contract import OBSERVED_FREE_ID, UNKNOWN_ID

DRIVEABLE_SURFACE_ID = 11
FILTER_VERSION = "0.1.1"


def atomic_update_npz(
    destination: Path,
    semantics: np.ndarray,
    mask_camera: np.ndarray,
    mask_lidar: np.ndarray,
) -> None:
    """Atomically replace a label NPZ without importing Step 6C production."""
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        semantics=semantics.astype(np.uint8, copy=False),
        mask_camera=mask_camera.astype(bool, copy=False),
        mask_lidar=mask_lidar.astype(bool, copy=False),
    )
    temporary.replace(destination)


def below_driveable_surface_mask(semantics: np.ndarray) -> np.ndarray:
    """Return ID-17 voxels strictly below the lowest ID-11 voxel per XY column."""
    if semantics.ndim != 3:
        raise ValueError(f"expected a 3D semantics grid, got {semantics.shape}")
    road = semantics == DRIVEABLE_SURFACE_ID
    has_road = np.any(road, axis=2)
    lowest_road = np.argmax(road, axis=2)
    z = np.arange(semantics.shape[2], dtype=np.int16)[None, None, :]
    below = has_road[:, :, None] & (z < lowest_road[:, :, None])
    return below & (semantics == OBSERVED_FREE_ID)


def filter_below_driveable(
    semantics: np.ndarray,
    mask_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    clear = below_driveable_surface_mask(semantics)
    updated_semantics = semantics.copy()
    updated_mask = mask_camera.copy()
    updated_semantics[clear] = UNKNOWN_ID
    updated_mask[clear] = False
    stats = {
        "driveable_xy_columns": int(np.count_nonzero(np.any(semantics == DRIVEABLE_SURFACE_ID, axis=2))),
        "cleared_observed_free_voxels": int(np.count_nonzero(clear)),
        "remaining_observed_free_voxels": int(np.count_nonzero(updated_semantics == OBSERVED_FREE_ID)),
        "remaining_visible_observed_free_voxels": int(np.count_nonzero((updated_semantics == OBSERVED_FREE_ID) & updated_mask)),
    }
    return updated_semantics, updated_mask, stats


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-manifest", type=Path, default=MANIFEST_ROOT / "occupancy_labels_v0.1.jsonl")
    parser.add_argument("--outcome-root", type=Path, default=OUTCOME_ROOT)
    parser.add_argument("--report-output", type=Path, default=REPORT_ROOT / "observed_free_below_driveable_filter_v0.1.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.report_output.exists() and not args.force:
        raise FileExistsError(args.report_output)

    rows = list(read_jsonl(args.label_manifest))
    if args.limit is not None:
        rows = rows[:args.limit]
    aggregate = Counter()
    examples = []
    for index, row in enumerate(rows, 1):
        label_path = args.outcome_root / row["labels_path"]
        with np.load(label_path, allow_pickle=False) as arrays:
            semantics = arrays["semantics"]
            mask_camera = arrays["mask_camera"].astype(bool)
            mask_lidar = arrays["mask_lidar"].astype(bool)
        updated_semantics, updated_mask, stats = filter_below_driveable(semantics, mask_camera)
        if stats["cleared_observed_free_voxels"]:
            atomic_update_npz(label_path, updated_semantics, updated_mask, mask_lidar)
            if len(examples) < 20:
                examples.append({"sample_id": row["sample_id"], **stats})
        aggregate.update(stats)
        if index == 1 or index % args.progress_every == 0 or index == len(rows):
            print(
                f"[Below-road filter] {index}/{len(rows)}: {row['sample_id']} "
                f"cleared={stats['cleared_observed_free_voxels']} "
                f"total_cleared={aggregate['cleared_observed_free_voxels']}",
                flush=True,
            )

    report = {
        "version": FILTER_VERSION,
        "status": "pass",
        "policy": "ID 17 strictly below the lowest ID 11 voxel in the same XY column becomes ID 255; mask_camera is cleared there",
        "record_count": len(rows),
        "aggregate": dict(aggregate),
        "examples": examples,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Report:", args.report_output)
    print("Cleared observed free:", aggregate["cleared_observed_free_voxels"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
