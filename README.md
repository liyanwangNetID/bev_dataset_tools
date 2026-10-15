# BEV AlpaSim Dataset Tools

## 1. Purpose and current objective

This repository builds a camera-only, two-frame, four-camera BEV semantic occupancy dataset from locally recorded AlpaSim Clips. The immediate objective is to finish the full-profile Step 6 build and audit, create leakage-free Map-ID splits, and start training smoke tests for ALOcc and GDFusion-compatible models.

The current path does not use virtual LiDAR, Scene-Fact, CoC, Reasoning, natural-language supervision, or fabricated static semantic pseudo-labels.

## 2. Current status

### Code and tests

- Steps 0 through 7 are implemented.
- All automated tests are located under `tests/stepN/` in this Git repository.
- The former project-level `../tests/` directory has been retired.
- Latest complete result on October 15, 2026: `65 passed, 2 warnings in 7.79s`.
- The two warnings are external `google._upb._message` Python 3.14 compatibility warnings; the current Python 3.12 test suite passes.

### Reference profile

```text
Raw root:      /home/lab/data_from_alpasim
Outcome root:  /home/lab/bev_alpasim_dataset_tools/outcome
Clips:         908
Keyframes:     24,019
Step 6 state:  v0.2 labels built, audited, and frozen
```

### Full profile

```text
Raw root:      /home/lab/data_all_alpasim
Outcome root:  /home/lab/bev_alpasim_dataset_tools/outcome_full
Clips:         2,311
Keyframes:     72,250
Step 6 state:  NPZ labels still need to be built and audited
```

Reference-profile counts, reports, and hashes must not be copied into the full profile.

## 3. Repository and output organization

```text
/home/lab/bev_alpasim_dataset_tools/
├── bev_dataset_tools/             # Git repository and all development code
│   ├── step1/ ... step7/
│   ├── tests/step1/ ... step7/
│   ├── tests/conftest.py
│   └── README.md                  # this authoritative README
├── outcome/                       # reference-profile formal products
└── outcome_full/                  # full-profile formal products
```

Rules:

- Step-specific source belongs in `stepN/`.
- Step-specific tests belong in `tests/stepN/`.
- Raw Clips are read-only.
- Formal products never belong in the Git repository.
- Temporary installers, audits, and review outputs belong under `/home/lab/Downloads` or `/tmp`.
- Git is maintained manually; automated work must not inspect, stage, or commit Git changes.
- This Git-repository README is the only maintained README.

## 4. Frozen data contract

### Camera and time order

```text
Camera order: cross_left, front_wide, cross_right, front_tele
Time order:   current, history
History gap:  approximately 0.5 seconds
Scene start:  current is repeated as history; history delta is zero
```

### Camera model

```text
Rectified pinhole image:  960 x 540
Initial training image:   480 x 270
Initial resize factor:    0.5
```

Image resizing must scale `fx`, `fy`, `cx`, and `cy` consistently.

### Occupancy grid

```text
Coordinate frame: current ego
Extent:           x,y in [-40.0,40.0) m; z in [-1.0,5.4) m
Voxel size:       0.4 m
Shape:            [200,200,16]
Axis order:       [x,y,z]
```

Each label NPZ contains:

```text
semantics    uint8 [200,200,16]
mask_camera bool  [200,200,16]
mask_lidar  bool  [200,200,16]
```

`mask_lidar` is always false.

## 5. Current Step 6 semantic and visibility logic

- IDs `0-16` are occupied semantic classes.
- ID `11` is physical `driveable_surface`.
- ID `17` is `observed_free_space`, meaning observed free 3D space rather than road or automatically drivable ground.
- ID `255` is unknown/ignore and must not contribute to training loss or metrics.
- Valid rectified camera frusta convert only unknown ID 255 voxels to ID 17; occupied IDs 0-16 are preserved.
- For each projected pixel, the first known surface is retained and supervision extends along the ray only through that surface.
- Voxels behind the first known surface remain unsupervised.
- Four-camera results are unioned.
- In an XY column containing ID 11, ID 17 strictly below the lowest driveable voxel is reset to ID 255 and the corresponding camera mask is cleared.
- Unknown ID 255 voxels may not have `mask_camera=True`.

Implementation:

```text
step6/camera_visibility.py
step6/apply_camera_visibility.py
step6/filter_below_driveable.py
step6/audit_visibility.py
step6/diagnose_visibility_edges.py
step6/finalize.py
```

### Latest audited reference statistics

```text
Checked label files:             24,019
ID 17 total voxels:       8,385,494,017
ID 17 camera-visible:        46,633,387
ID 255 total voxels:      6,647,743,604
mask_camera mean/sample:        8,774.409
mask_camera maximum:               29,076
zero-visibility samples:               58
full-visibility samples:                  0
Step 6 audit failures:                    0
```

Current contract:

```text
outcome/reports/step6_dataset_contract_v0.2.json
```

The complete contract hash must be read from that file. Any label mutation invalidates previous statistics, audits, edge diagnosis, and hashes.

## 6. Capability boundary

Reliable sources currently cover lane geometry and traffic-participant Actor boxes. They do not provide complete buildings, walls, trees, vegetation, terrain meshes, curbs, semantic depth, or complete static-world occupancy. Camera free-space rays may therefore pass through unmodeled static objects.

The separate AlpaSim/ROS 2 static-environment export is not part of this frozen training contract. Incorporating it requires an explicit contract revision, rebuild, audit, and training comparison.

## 7. Pipeline state

1. Step 0: schema and contracts, implemented.
2. Step 1: Clip manifest, implemented for both profiles.
3. Step 2: unified reader and geometry API, implemented.
4. Step 3: synchronized temporal candidates, implemented for both profiles.
5. Step 4: formal keyframes, implemented for both profiles.
6. Step 5: calibration census, rectification assets, assignment and audit, implemented in code; full-profile products must be used for the full build.
7. Step 6A-F: reference profile complete; full profile still needs labels, audit, diagnosis, and v0.2 freeze.
8. Step 7: DataLoader, calibration resolution, OccStudio bridge, delivery audit, and visualization, implemented and tested.
9. Next: full-profile Step 6, Map-ID split, training smoke tests, baseline, temporal model, evaluation, and release freeze.

## 8. Reproducible command book

The commands below are retained as the operational runbook. Set the active profile first. Use current CLI `--help` if an entry point changes.

### Environment setup

Reference dataset:

```bash
cd /home/lab/bev_alpasim_dataset_tools || exit 1
export BEV_ALPASIM_DATA_ROOT=/home/lab/data_from_alpasim
export ALPASIM_RAW_DATA_ROOT=/home/lab/data_from_alpasim
export BEV_DATASET_PROJECT_ROOT=/home/lab/bev_alpasim_dataset_tools
export BEV_DATASET_OUTCOME_ROOT=/home/lab/bev_alpasim_dataset_tools/outcome
export PYTHONPATH=/home/lab/bev_alpasim_dataset_tools/bev_dataset_tools:/home/lab/bev_alpasim_dataset_tools${PYTHONPATH:+:$PYTHONPATH}
```

Full dataset:

```bash
cd /home/lab/bev_alpasim_dataset_tools || exit 1
export BEV_ALPASIM_DATA_ROOT=/home/lab/data_all_alpasim
export ALPASIM_RAW_DATA_ROOT=/home/lab/data_all_alpasim
export BEV_DATASET_PROJECT_ROOT=/home/lab/bev_alpasim_dataset_tools
export BEV_DATASET_OUTCOME_ROOT=/home/lab/bev_alpasim_dataset_tools/outcome_full
export PYTHONPATH=/home/lab/bev_alpasim_dataset_tools/bev_dataset_tools:/home/lab/bev_alpasim_dataset_tools${PYTHONPATH:+:$PYTHONPATH}
```

The path helper reads `BEV_ALPASIM_DATA_ROOT`, not `ALPASIM_RAW_DATA_ROOT`. Both are exported because some Step 7 compatibility code may still read the older name.

### Complete production command sequence

Run from `/home/lab/bev_alpasim_dataset_tools`. Set the environment first to either the reference or full profile above. The commands below use path-helper defaults unless explicit paths are required.

#### Step 1: Clip manifest

```bash
python3 -m bev_dataset_tools.step1.clip_manifest --force
```

#### Step 2: Unified reader and geometry API

Step 2 is a shared API and has no independent formal production command. Validate it through the complete test suite.

#### Step 3: Occupancy Clip profile

```bash
python3 -m bev_dataset_tools.step3.profile \
  --dataset-root "$BEV_ALPASIM_DATA_ROOT" \
  --profile-output "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_clip_profile_v0.1.jsonl" \
  --summary-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_feasibility_v0.1.json" \
  --force
```

#### Step 4: Formal Keyframes

```bash
python3 -m bev_dataset_tools.step4.keyframes \
  --dataset-root "$BEV_ALPASIM_DATA_ROOT" \
  --step1-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/clips_v0.1.jsonl" \
  --step3-profile "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_clip_profile_v0.1.jsonl" \
  --output "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_keyframes_v0.1.jsonl" \
  --summary-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_keyframe_summary_v0.1.json" \
  --rejections-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_temporal_rejections_v0.1.jsonl" \
  --force
```

#### Step 5A: Calibration census and shared rectification assets

```bash
python3 -m bev_dataset_tools.step5.census --force
```

#### Step 5B: Rectification trial

```bash
python3 -m bev_dataset_tools.step5.trial --force
```

#### Step 5C: Rectification assignments and contract

```bash
python3 -m bev_dataset_tools.step5.finalize --force
```

#### Step 6A: Compact occupancy source manifest

```bash
python3 -m bev_dataset_tools.step6.build_sources \
  --keyframe-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_keyframes_v0.1.jsonl" \
  --dataset-root "$BEV_ALPASIM_DATA_ROOT" \
  --output "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_sources_v0.1.jsonl" \
  --summary-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_source_summary_v0.1.json" \
  --contract-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_source_contract_v0.1.json" \
  --force
```

#### Step 6B: Occupied semantic voxelization

```bash
python3 -m bev_dataset_tools.step6.voxelize \
  --source-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_sources_v0.1.jsonl" \
  --keyframe-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_keyframes_v0.1.jsonl" \
  --dataset-root "$BEV_ALPASIM_DATA_ROOT" \
  --output-root "$BEV_DATASET_OUTCOME_ROOT/occupancy/labels_v0.1" \
  --manifest-output "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_labels_v0.1.jsonl" \
  --summary-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_label_summary_v0.1.json" \
  --force
```

#### Step 6C: Observed-free generation and conservative camera mask

```bash
python3 -m bev_dataset_tools.step6.apply_camera_visibility \
  --label-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_labels_v0.1.jsonl" \
  --outcome-root "$BEV_DATASET_OUTCOME_ROOT" \
  --dataset-root "$BEV_ALPASIM_DATA_ROOT" \
  --assignments "$BEV_DATASET_OUTCOME_ROOT/manifests/rectification_assignments_v0.1.jsonl" \
  --rectification-root "$BEV_DATASET_OUTCOME_ROOT/calibration/rectification_v0.1" \
  --summary-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_camera_visibility_summary_v0.1.json" \
  --audit-output "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_camera_visibility_audit_v0.1.jsonl" \
  --force
```

This production command now invokes the below-driveable filter internally. For labels generated before that integration, run the standalone migration once:

```bash
python3 -m bev_dataset_tools.step6.filter_below_driveable \
  --label-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_labels_v0.1.jsonl" \
  --outcome-root "$BEV_DATASET_OUTCOME_ROOT" \
  --report-output "$BEV_DATASET_OUTCOME_ROOT/reports/observed_free_below_driveable_filter_v0.1.json" \
  --progress-every 500 \
  --force
```

Do not run the standalone migration after a fresh Step 6C unless auditing idempotency; a second run should clear zero additional voxels.

#### Step 6D: Visibility and semantic audit

```bash
python3 -m bev_dataset_tools.step6.audit_visibility \
  --label-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_labels_v0.1.jsonl" \
  --outcome-root "$BEV_DATASET_OUTCOME_ROOT" \
  --visibility-summary "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_camera_visibility_summary_v0.1.json" \
  --visibility-audit "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_camera_visibility_audit_v0.1.jsonl" \
  --summary-output "$BEV_DATASET_OUTCOME_ROOT/reports/step6_final_audit_v0.1.json" \
  --edge-cases-output "$BEV_DATASET_OUTCOME_ROOT/reports/step6_visibility_edge_cases_v0.1.jsonl" \
  --force
```

#### Step 6E: Edge diagnosis

```bash
python3 -m bev_dataset_tools.step6.diagnose_visibility_edges \
  --edge-cases "$BEV_DATASET_OUTCOME_ROOT/reports/step6_visibility_edge_cases_v0.1.jsonl" \
  --outcome-root "$BEV_DATASET_OUTCOME_ROOT" \
  --dataset-root "$BEV_ALPASIM_DATA_ROOT" \
  --assignments "$BEV_DATASET_OUTCOME_ROOT/manifests/rectification_assignments_v0.1.jsonl" \
  --rectification-root "$BEV_DATASET_OUTCOME_ROOT/calibration/rectification_v0.1" \
  --report-output "$BEV_DATASET_OUTCOME_ROOT/reports/step6_visibility_edge_diagnosis_v0.1.json" \
  --visual-root "$BEV_DATASET_OUTCOME_ROOT/review/step6_visibility_edge_diagnosis_current" \
  --force
```

#### Step 6F: Freeze dataset contract

```bash
python3 -m bev_dataset_tools.step6.finalize \
  --label-manifest "$BEV_DATASET_OUTCOME_ROOT/manifests/occupancy_labels_v0.1.jsonl" \
  --labels-root "$BEV_DATASET_OUTCOME_ROOT/occupancy/labels_v0.1" \
  --label-summary "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_label_summary_v0.1.json" \
  --visibility-summary "$BEV_DATASET_OUTCOME_ROOT/reports/occupancy_camera_visibility_summary_v0.1.json" \
  --final-audit "$BEV_DATASET_OUTCOME_ROOT/reports/step6_final_audit_v0.1.json" \
  --edge-diagnosis "$BEV_DATASET_OUTCOME_ROOT/reports/step6_visibility_edge_diagnosis_v0.1.json" \
  --output "$BEV_DATASET_OUTCOME_ROOT/reports/step6_dataset_contract_v0.2.json" \
  --force
```

#### Step 7A: Complete adapter tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest tests/step7 -q
```

#### Step 7B: Delivery audit

List real sample IDs instead of reusing IDs from another raw dataset:

```bash
python3 -m bev_dataset_tools.step7.list_samples \
  --scene-id test_clip_001 \
  --limit 5
```

Then audit one returned sample ID:

```bash
python3 -m bev_dataset_tools.step7.audit_delivery \
  --sample-id SAMPLE_ID_FROM_LIST \
  --output /tmp/step7_delivery_audit.json
```

#### Step 7C: DataLoader and visualization audit

```bash
rm -rf /tmp/step7_review
python3 -m bev_dataset_tools.step7.validate_dataloader_sample \
  --sample-id SAMPLE_ID_FROM_LIST \
  --output-dir /tmp/step7_review
```

### Tests

Step 6 tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest tests/step6 -q
```

Step 7 targeted tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest tests/step7 -q
```

Complete project suite from the Git repository root:

```bash
cd /home/lab/bev_alpasim_dataset_tools/bev_dataset_tools || exit 1
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q
```

### Observed-free audits and visualizations

Full ID 17 audit tool, when available in `/home/lab/Downloads`:

```bash
python3 -u /home/lab/Downloads/audit_observed_free_space_v2.py \
  --outcome-root "$BEV_DATASET_OUTCOME_ROOT" \
  --output "$BEV_DATASET_OUTCOME_ROOT/reports/observed_free_space_audit_current.json" \
  --progress-every 500
```

Height-slice visualization for a selected `labels.npz`:

```bash
python3 /home/lab/Downloads/render_observed_free_height_slices.py \
  --labels /absolute/path/to/labels.npz \
  --output /tmp/observed_free_height_slices.png \
  --masked-output /tmp/observed_free_height_slices_masked.png
```

Display convention:

- Image up: ego front, `+x`
- Image left: ego left, `+y`
- Black dot: ego origin
- Black arrow: ego heading
- Bright green `#23D25A`: ID 17 observed free
- Light green `#A3D57C`: ID 11 driveable surface
- Near white `#F5F5F5`: ID 255 unknown

### Post-build mandatory consistency checks

After Step 6C or any label mutation:

1. Recompute actual NPZ class counts. Do not trust stale Step 6B `class_counts` fields.
2. Verify ID 17 exists globally and in `mask_camera`.
3. Verify no ID 17 remains below the lowest ID 11 in the same XY column.
4. Verify IDs 0-16 were not overwritten by ID 17.
5. Verify `mask_lidar` remains all False.
6. Rerun Step 6 audit, edge diagnosis, and finalize.
7. Rerun Step 7 tests and one real-sample delivery audit.
8. Any prior Step 6 contract hash becomes stale after label mutation.

### Known limitations and next development target

- Buildings, walls, trees, vegetation, terrain meshes, curbs, and complete static obstacles are not represented by the current raw annotation contract.
- Current class 17 can pass through unmodeled static obstacles because the raw inputs do not provide their geometry.
- The next development task is to inspect the AlpaSim export source and every non-camera raw artifact to determine whether complete static world geometry can be exported.
- Do not apply static pseudo-labels from RGB alone.
- The full 2,311-Clip dataset must not be re-frozen with the current class 17 change until the small reference dataset is re-audited after below-road filtering and the static-geometry limitation is explicitly accepted.

## 9. Current test baseline

Run the complete suite from the Git repository root:

```bash
cd /home/lab/bev_alpasim_dataset_tools/bev_dataset_tools || exit 1

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Current result:

```text
65 passed, 2 warnings in 7.79s
```

## 10. Next work

1. Activate the full profile.
2. Continue the command book from the first full-profile product that is not complete.
3. Build Step 6 labels using the current observed-free and below-driveable logic.
4. Run Step 6 audit, edge diagnosis, and v0.2 finalization against `outcome_full`.
5. Rerun the complete test suite and one real-sample Step 7 delivery/visualization audit.
6. Create leakage-free Map-ID-level train, validation, and test splits.
7. Run an 8-sample overfit test and a 64-train/16-validation smoke test.
8. Train a single-frame baseline before ALOcc plus GDFusion temporal training.

## 11. Maintenance rule

Update this README in place. Remove statements that are no longer true, revise changed commands and counts, and preserve still-valid contracts, operational commands, limitations, and handoff information. Do not replace it with a short summary and do not accumulate override notices above obsolete text.
