"""Diagnose zero/full camera-visibility edge cases before freezing Step 6."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, OUTCOME_ROOT, REPORT_ROOT
from step1.clip_manifest import CAMERA_NAMES
from step6.camera_visibility import load_assignment_index, load_camera_contract, project_known_voxels
from step6.contract import GRID, UNKNOWN_ID

COLORS = {0:(128,128,128),3:(255,128,0),4:(0,0,255),7:(255,0,255),9:(0,128,255),10:(0,255,255),11:(0,180,0),255:(18,18,18)}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def metric_bounds(mask: np.ndarray) -> dict[str, list[float]] | None:
    indices = np.argwhere(mask)
    if not len(indices):
        return None
    points = np.stack([
        GRID.x_min + (indices[:,0] + 0.5) * GRID.voxel_size,
        GRID.y_min + (indices[:,1] + 0.5) * GRID.voxel_size,
        GRID.z_min + (indices[:,2] + 0.5) * GRID.voxel_size,
    ], axis=1)
    return {axis:[float(points[:,i].min()), float(points[:,i].max())] for i,axis in enumerate("xyz")}


def quadrants(mask: np.ndarray) -> dict[str, int]:
    indices = np.argwhere(mask)
    if not len(indices):
        return {"front_left":0,"front_right":0,"rear_left":0,"rear_right":0}
    x = GRID.x_min + (indices[:,0] + 0.5) * GRID.voxel_size
    y = GRID.y_min + (indices[:,1] + 0.5) * GRID.voxel_size
    return {
        "front_left":int(np.count_nonzero((x>=0)&(y>=0))),
        "front_right":int(np.count_nonzero((x>=0)&(y<0))),
        "rear_left":int(np.count_nonzero((x<0)&(y>=0))),
        "rear_right":int(np.count_nonzero((x<0)&(y<0))),
    }


def bev(semantics: np.ndarray, visible: np.ndarray | None, title: str) -> np.ndarray:
    selected = semantics != UNKNOWN_ID
    if visible is not None:
        selected &= visible
    labels = np.full(semantics.shape[:2], UNKNOWN_ID, dtype=np.uint8)
    labels[np.any(selected & (semantics == 11), axis=2)] = 11
    for class_id in np.unique(semantics):
        if class_id in (11, UNKNOWN_ID):
            continue
        labels[np.any(selected & (semantics == class_id), axis=2)] = class_id
    image = np.zeros((*labels.T.shape,3), dtype=np.uint8)
    for class_id in np.unique(labels):
        image[labels.T == class_id] = COLORS.get(int(class_id),(255,255,255))
    image = cv2.flip(image,0)
    image = cv2.resize(image,(400,400),interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(image,(200,200),(255,255,255),cv2.MARKER_CROSS,14,2)
    header=np.zeros((42,400,3),dtype=np.uint8)
    cv2.putText(header,title,(8,27),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
    return np.vstack([header,image])


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--edge-cases",type=Path,default=REPORT_ROOT/"step6_visibility_edge_cases_v0.1.jsonl")
    parser.add_argument("--outcome-root",type=Path,default=OUTCOME_ROOT)
    parser.add_argument("--dataset-root",type=Path,default=ALPASIM_DATA_ROOT)
    parser.add_argument("--assignments",type=Path,default=MANIFEST_ROOT/"rectification_assignments_v0.1.jsonl")
    parser.add_argument("--rectification-root",type=Path,default=OUTCOME_ROOT/"calibration"/"rectification_v0.1")
    parser.add_argument("--report-output",type=Path,default=REPORT_ROOT/"step6_visibility_edge_diagnosis_v0.1.json")
    parser.add_argument("--visual-root",type=Path,default=OUTCOME_ROOT/"review"/"step6_visibility_edge_diagnosis_v0.1")
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    if args.visual_root.exists() and any(args.visual_root.iterdir()) and not args.force:
        raise FileExistsError(f"output exists: {args.visual_root}")
    args.visual_root.mkdir(parents=True,exist_ok=True)
    assignments=load_assignment_index(args.assignments)
    records=[]
    active_scene=None
    contracts=None
    for index,case in enumerate(read_jsonl(args.edge_cases),1):
        scene_id=case["scene_id"]
        if scene_id != active_scene:
            contracts={name:load_camera_contract(scene_id,name,args.dataset_root,assignments,args.rectification_root) for name in CAMERA_NAMES}
            active_scene=scene_id
        assert contracts is not None
        label_path=args.outcome_root/case["labels_path"]
        with np.load(label_path) as arrays:
            semantics=arrays["semantics"]
            saved=arrays["mask_camera"]
        known=semantics!=UNKNOWN_ID
        recomputed=np.zeros(GRID.shape,dtype=bool)
        camera_stats={}
        panels=[bev(semantics,None,"known semantics")]
        for name in CAMERA_NAMES:
            extrinsic,k,valid,asset=contracts[name]
            mask,stats=project_known_voxels(semantics,extrinsic,k,valid)
            recomputed|=mask
            camera_stats[name]={**stats,"asset_id":asset,"visible_quadrants":quadrants(mask)}
            panels.append(bev(semantics,mask,name))
        if not np.array_equal(recomputed,saved):
            raise RuntimeError(f"recomputed mask mismatch: {case['sample_id']}")
        top=np.hstack(panels[:3]); bottom=np.hstack(panels[3:]); bottom=cv2.copyMakeBorder(bottom,0,0,0,top.shape[1]-bottom.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0)); image=np.vstack([top,bottom])
        visual=args.visual_root/scene_id/f"{case['sample_id']}.jpg"; visual.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(visual),image,[int(cv2.IMWRITE_JPEG_QUALITY),94])
        records.append({
            **case,
            "known_bounds_ego_m":metric_bounds(known),
            "known_quadrants":quadrants(known),
            "camera_projection_counts":camera_stats,
            "recomputed_visible":int(recomputed.sum()),
            "visual_path":str(visual),
        })
        print(f"[Step 6C Edge] {index}: {case['sample_id']} known={int(known.sum())} visible={int(recomputed.sum())}")
    zero_nonempty=[row for row in records if row["known_voxels"]>0 and row["mask_camera_true"]==0]
    zero_empty=[row for row in records if row["known_voxels"]==0]
    full=[row for row in records if row["known_voxels"]>0 and row["mask_camera_true"]==row["known_voxels"]]
    report={
        "record_count":len(records),
        "zero_empty_count":len(zero_empty),
        "zero_nonempty_count":len(zero_nonempty),
        "full_visibility_count":len(full),
        "zero_nonempty_samples":[row["sample_id"] for row in zero_nonempty],
        "full_visibility_samples":[row["sample_id"] for row in full],
        "records":records,
    }
    args.report_output.parent.mkdir(parents=True,exist_ok=True)
    args.report_output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("Report:",args.report_output)
    print("Visuals:",args.visual_root)
    print("Zero empty:",len(zero_empty))
    print("Zero nonempty:",len(zero_nonempty))
    print("Full visibility:",len(full))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
