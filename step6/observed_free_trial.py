from __future__ import annotations
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, OUTCOME_ROOT
from step1.clip_manifest import CAMERA_NAMES
from step6.camera_visibility import load_assignment_index, load_camera_contract, project_voxel_contract
from step6.contract import GRID, OBSERVED_FREE_ID, UNKNOWN_ID
from step6.apply_camera_visibility import atomic_update_npz


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def palette():
    result = np.random.default_rng(7).integers(35, 240, (256, 3), dtype=np.uint8)
    result[OBSERVED_FREE_ID] = (35, 210, 90)
    result[UNKNOWN_ID] = (245, 245, 245)
    return result


def save_bev(path, semantics, mask=None):
    values = semantics if mask is None else np.where(mask, semantics, UNKNOWN_ID)
    bev = np.full(values.shape[:2], UNKNOWN_ID, np.uint8)
    for z in range(values.shape[2]):
        layer = values[:, :, z]
        occupied = (layer != UNKNOWN_ID) & (layer != OBSERVED_FREE_ID)
        bev[(bev == UNKNOWN_ID) & (layer == OBSERVED_FREE_ID)] = OBSERVED_FREE_ID
        bev[occupied] = layer[occupied]

    # Display convention: +x/front is up, +y/left is image-left.
    display = bev[::-1, ::-1]
    image = cv2.resize(palette()[display], (800, 800), interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    center = (400, 400)
    cv2.circle(image, center, 9, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.arrowedLine(image, center, (400, 320), (0, 0, 0), 7, cv2.LINE_AA, tipLength=0.28)
    cv2.putText(image, 'EGO', (414, 414), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'FRONT +x', (340, 302), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'LEFT +y', (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, 'RIGHT -y', (676, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # Explicit legend for the two greens under review.
    cv2.rectangle(image, (18, 748), (42, 772), (90, 210, 35), -1)
    cv2.putText(image, '17 observed free', (52, 769), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(image, (252, 748), (276, 772), (124, 213, 163), -1)
    cv2.putText(image, '11 driveable surface', (286, 769), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-manifest', type=Path, default=MANIFEST_ROOT / 'occupancy_labels_v0.1.jsonl')
    parser.add_argument('--outcome-root', type=Path, default=OUTCOME_ROOT)
    parser.add_argument('--dataset-root', type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument('--assignments', type=Path, default=MANIFEST_ROOT / 'rectification_assignments_v0.1.jsonl')
    parser.add_argument('--rectification-root', type=Path, default=OUTCOME_ROOT / 'calibration/rectification_v0.1')
    parser.add_argument('--output-root', type=Path, default=OUTCOME_ROOT / 'review/step6_observed_free_trial_v0.2')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.output_root.exists() and args.force:
        shutil.rmtree(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    assignments = load_assignment_index(args.assignments)
    active_scene = None
    contracts = None
    records = []
    aggregate = Counter()
    for index, row in enumerate(read_jsonl(args.label_manifest), 1):
        if index > args.limit:
            break
        scene = row['scene_id']
        if scene != active_scene:
            contracts = {
                camera: load_camera_contract(scene, camera, args.dataset_root, assignments, args.rectification_root)
                for camera in CAMERA_NAMES
            }
            active_scene = scene
        label_source = args.outcome_root / row['labels_path']
        sample_dir = args.output_root / scene / row['sample_id']
        sample_dir.mkdir(parents=True, exist_ok=True)
        label_target = sample_dir / 'labels.npz'
        shutil.copy2(label_source, label_target)
        with np.load(label_source) as arrays:
            original = arrays['semantics'].copy()
            mask_lidar = arrays['mask_lidar'].copy()
        semantics = original.copy()
        frustum = np.zeros(GRID.shape, dtype=bool)
        free = np.zeros(GRID.shape, dtype=bool)
        supervised = np.zeros(GRID.shape, dtype=bool)
        cameras = {}
        for camera, (extrinsic, intrinsic, valid, _) in contracts.items():
            cf, cfree, cmask, stats = project_voxel_contract(original, extrinsic, intrinsic, valid)
            frustum |= cf
            free |= cfree
            supervised |= cmask
            cameras[camera] = stats
        semantics[free & (semantics == UNKNOWN_ID)] = OBSERVED_FREE_ID
        mask_camera = supervised & (semantics != UNKNOWN_ID)
        atomic_update_npz(label_target, semantics, mask_camera, mask_lidar)
        before_occupied = int(np.count_nonzero(original < OBSERVED_FREE_ID))
        after_occupied = int(np.count_nonzero(semantics < OBSERVED_FREE_ID))
        stats = {
            'sample_id': row['sample_id'], 'scene_id': scene,
            'occupied_before': before_occupied, 'occupied_after': after_occupied,
            'observed_free': int(np.count_nonzero(semantics == OBSERVED_FREE_ID)),
            'visible_observed_free': int(np.count_nonzero((semantics == OBSERVED_FREE_ID) & mask_camera)),
            'unknown': int(np.count_nonzero(semantics == UNKNOWN_ID)),
            'mask_camera_true': int(mask_camera.sum()), 'camera_stats': cameras,
            'trial_labels_path': str(label_target),
        }
        if before_occupied != after_occupied:
            raise RuntimeError(f"occupied count changed: {row['sample_id']}")
        save_bev(sample_dir / 'before_semantics_bev.png', original)
        save_bev(sample_dir / 'after_semantics_bev.png', semantics)
        save_bev(sample_dir / 'supervised_semantics_bev.png', semantics, mask_camera)
        records.append(stats)
        aggregate.update({
            'occupied': after_occupied, 'observed_free': stats['observed_free'],
            'visible_observed_free': stats['visible_observed_free'], 'unknown': stats['unknown'],
        })
        print(f"[Observed Free Trial] {index}/{args.limit}: {row['sample_id']} free={stats['observed_free']} visible_free={stats['visible_observed_free']}")
    report = {
        'version': '0.2.0', 'record_count': len(records),
        'policy': 'valid_rectified_camera_frustum unknown-to-17; mask_camera only through first known surface',
        'aggregate': dict(aggregate), 'records': records,
    }
    report_path = args.output_root / 'trial_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Report:', report_path)
    print('Observed free:', aggregate['observed_free'])
    print('Visible observed free:', aggregate['visible_observed_free'])


if __name__ == '__main__':
    main()
