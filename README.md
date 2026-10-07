# BEV AlpaSim Dataset Tools

## 1. Project purpose

This repository prepares a minimal, reproducible camera-based dataset for fine-tuning a BEV semantic occupancy model, with the immediate target being ALOcc with GDFusion through OccStudio or an equivalent compatible training framework.

The current priority is deliberately narrow:

1. Use only the 908 AlpaSim Clips already stored locally.
2. Build stable temporal camera samples.
3. Generate partial Occ3D-compatible semantic occupancy labels.
4. Generate conservative camera-visibility masks.
5. Adapt the dataset to the BEV occupancy training framework.
6. Create leakage-free train, validation, and test splits.
7. Run smoke tests, fine-tuning, and evaluation.

The current minimal version does not include Scene-Fact, structured CoC, Reasoning, natural-language supervision, navigation reasoning, driving-decision labels, virtual LiDAR, USDZ assets, or remote scene downloads.

---

## 2. Important paths

### Development code

```text
/home/lab/bev_alpasim_dataset_tools/bev_dataset_tools
```

### Formal dataset outputs

```text
/home/lab/bev_alpasim_dataset_tools/outcome
```

### Raw input Clips

```text
/home/lab/data_from_alpasim/test_clip_001
...
/home/lab/data_from_alpasim/test_clip_908
```

### Environment

The active Python environment used during development is:

```text
/home/lab/alpasim/.venv
```

The environment contains a ROS-related pytest plugin that is incompatible with the installed pytest version. Run the full test suite with plugin autoload disabled:

```bash
cd /home/lab/bev_alpasim_dataset_tools/bev_dataset_tools

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q
```

Do not use Git operations in automated development. The user maintains Git manually.

---

## 3. Fixed project rules

### 3.1 Source boundary

The dataset input is strictly:

```text
908 locally stored Clips
```

The formal pipeline must not require:

```text
USDZ
mesh.ply
mesh_ground.ply
remote Hugging Face artifacts
online simulator services
virtual LiDAR
```

Three previously inspected USDZ files were used only to understand capability boundaries. USDZ is not part of the formal dataset contract.

### 3.2 Development organization

Step-specific code belongs under:

```text
stepN/
```

Step-specific tests belong under:

```text
tests/stepN/
```

Do not leave Step-specific scripts in the repository root.

### 3.3 Output organization

Formal products belong under:

```text
/home/lab/bev_alpasim_dataset_tools/outcome
```

Raw Clips are read-only. Formal products must not be written into the Raw Clip directories or the development-code directory.

### 3.4 Workflow

For each Step:

1. Implement a coherent batch of functionality.
2. Add or update tests.
3. Run the complete test suite once.
4. Build a small real-data trial when needed.
5. Compare and audit products.
6. Build the formal product.
7. Freeze the Step contract.
8. Move directly to the next task when no user decision or additional input is required.

Do not repeatedly ask the user to confirm routine next actions.

---

## 4. Current dataset scale

```text
Raw Clips:                   908
Unique Map IDs:              875
Formal Keyframes:         24,019
Formal occupancy labels:  24,019
Occupancy-label storage:  about 280 MB
```

---

## 5. Completed Steps

# Step 0: Foundation and contracts

## Purpose

Step 0 established the repository layout, path handling, versioning rules, schema foundations, and the separation between Raw Clips, development code, and formal products.

## Frozen decisions

```text
Development code root:
/home/lab/bev_alpasim_dataset_tools/bev_dataset_tools

Formal output root:
/home/lab/bev_alpasim_dataset_tools/outcome

Raw Clip root:
/home/lab/data_from_alpasim
```

All later Steps must use shared path helpers and formal manifests rather than hard-coded dataset sizes or duplicated directory scans.

---

# Step 1: Clip Manifest

## Purpose

Step 1 discovers and validates the 908 local Clips and creates the stable inventory used by later Steps.

## Responsibilities

- Discover `test_clip_001` through `test_clip_908`.
- Assign stable Clip IDs.
- Verify required Clip content.
- Record available actor, ego, map, camera, and calibration sources.
- Provide one canonical Clip discovery interface.

## Frozen decisions

- A Clip is the fundamental raw-data container.
- Later Steps must discover Clips through formal Step 1 products, not by using `range(1, 909)`.
- The current dataset consists of 908 local Clips.

---

# Step 2: Unified Clip reading API

## Purpose

Step 2 provides a shared API for reading one Clip and aligning records by timestamp. The main interface is `DrivingClipReader`.

## Responsibilities

- Read current, history, and future Actor records.
- Read ego poses.
- Read VectorMap.
- Read camera streams and image references.
- Read camera calibration.
- Find exact or nearest records by timestamp.
- Return explicit timing errors.
- Provide coordinate and rigid-transform utilities.

## Frozen decisions

- Later Steps should use the unified reader rather than parse Raw Clip JSON or JSONL independently.
- Timing alignment must expose `time_error_ns`.
- Schema drift must fail explicitly instead of being silently mapped.
- Actor alignment for all 24,019 current Keyframes was exact. The maximum Actor timing error was 0 ns.

---

# Step 3: Candidate Anchor and temporal-pair generation

## Purpose

Step 3 creates candidate temporal samples from timestamps shared exactly by all four cameras.

## Camera order

```text
cross_left
front_wide
cross_right
front_tele
```

## Candidate-generation logic

1. Find timestamps present in all four camera streams.
2. Use a shared timestamp as a candidate current time.
3. Find a shared history time approximately 0.5 seconds earlier.
4. Form a candidate `(history_timestamp, current_timestamp)` pair.

## Important clarification

Step 3 is primarily a temporal and camera-completeness filter. It does not select Anchors based on semantic importance, rare events, pedestrians, intersections, or driving difficulty.

A candidate Anchor is therefore:

```text
a four-camera-synchronized temporal pair
```

not:

```text
a scene selected because it is interesting
```

---

# Step 4: Formal Keyframes

## Purpose

Step 4 converts candidate temporal pairs into formal samples after validating the final sample contract.

## Relationship to Step 3

For ordinary temporal samples:

```text
formal Keyframe
=
candidate history-current pair
+
valid history ego pose
+
valid current ego pose
+
final schema and uniqueness checks
```

Most camera and timing filters were already completed in Step 3. The main actual rejection reason during Step 4 was `history_pose_boundary`, where the history image time was outside the usable ego-pose range.

## Scene-start Keyframes

Each Clip also receives one special scene-start sample:

```text
history_timestamp = current_timestamp
history image = current image
history pose = current pose
history_delta_ns = 0
history_is_repeated_current = true
is_scene_start = true
```

This sample is not an ordinary 0.5-second temporal pair.

## Current formal scale

```text
Temporal Keyframes: 23,111
Scene-start samples:    908
Total Keyframes:     24,019
```

## Stable sample ID

```text
<clip_id>_<current_timestamp_ns>
```

Example:

```text
test_clip_001_9300112661000
```

## Formal product

```text
/home/lab/bev_alpasim_dataset_tools/outcome/manifests/occupancy_keyframes_v0.1.jsonl
```

## Frozen decisions

- Occupancy coordinates use the current ego frame.
- Ordinary time separation is approximately 0.5 seconds.
- Four exact current frames and four exact history frames are required.
- Both current and history ego poses are required.
- Sample IDs must remain stable and reproducible.

---

# Step 5: Camera calibration and rectification

## Purpose

Step 5 converts the original distorted camera model into a uniform rectified pinhole-image contract.

This is not a plain image resize. It performs calibration-based reprojection.

## Responsibilities

- Read original intrinsics and F-Theta distortion parameters.
- Read each camera pose relative to the rig.
- Build reusable remap assets.
- Rectify distorted images.
- Produce a new rectified intrinsic matrix `K_rect`.
- Produce a rectification-valid-pixel mask.
- Assign each Clip and camera to a shared rectification asset.

## Output size

```text
960 x 540
```

## Four cameras

```text
cross_left
front_wide
cross_right
front_tele
```

## Key products

Rectification assets include the equivalent of:

```text
metadata.json
K_rect
valid_mask.png
remap information
```

## Frozen decisions

- Training images use the Step 5 rectification contract.
- Any later image resize must also scale `fx`, `fy`, `cx`, and `cy` in `K_rect`.
- Pixels outside the rectification valid mask are not valid camera observations.

---

# Step 6: Occ3D-compatible partial semantic occupancy

Step 6 generates semantic occupancy labels and conservative camera visibility.

## Formal grid contract

```text
Coordinate frame: current ego

x in [-40.0, 40.0) m
y in [-40.0, 40.0) m
z in [ -1.0,  5.4) m

Voxel size: 0.4 m
Shape: [200, 200, 16]
Axis order: [x, y, z]
```

## Semantic IDs

```text
0    others
1    barrier
2    bicycle
3    bus
4    car
5    construction_vehicle
6    motorcycle
7    pedestrian
8    traffic_cone
9    trailer
10   truck
11   driveable_surface
12   other_flat
13   sidewalk
14   terrain
15   manmade
16   vegetation
17   observed_free_space
255  unknown / ignore
```

## Step 6A: Source geometry

### Actor source

```text
actors/current.jsonl
```

Actors are converted from map coordinates into the current ego frame and represented as oriented 3D boxes.

### Actor mapping

```text
automobile          -> 4  car
person              -> 7  pedestrian
heavy_truck         -> 10 truck
trailer             -> 9  trailer
bus                 -> 3  bus

rider               -> 0  others
protruding_object   -> 0  others
other_vehicle       -> 0  others
stroller            -> 0  others
train_or_tram_car   -> 0  others
animal              -> 0  others
```

`others=0` means that occupancy is known but no direct requested Occ3D category can be assigned reliably.

### VectorMap source

```text
map/vector_map.json
```

The left and right lane boundaries are triangulated into a three-dimensional lane ribbon and mapped to:

```text
11 driveable_surface
```

### Compact-source manifest

Actor geometry is stored per sample. VectorMap geometry is referenced and loaded per scene instead of duplicating lane vertices into every sample.

Formal source manifest:

```text
/home/lab/bev_alpasim_dataset_tools/outcome/manifests/occupancy_sources_v0.1.jsonl
```

Current size is about 330 MB.

## Step 6B: Semantic voxelization

Each grid begins as:

```text
255 unknown
```

Then:

```text
lane ribbon -> 11 driveable_surface
Actor OBB   -> mapped Actor semantic ID
```

Priority is:

```text
Actor semantic > driveable_surface > unknown
```

The pipeline does not invent classes that are absent from the local Clip data.

## Step 6C: Camera visibility

Every `labels.npz` contains three independent tensors:

```text
semantics    uint8 [200, 200, 16]
mask_camera  bool  [200, 200, 16]
mask_lidar   bool  [200, 200, 16]
```

`mask_camera=False` does not change a known semantic label to 255. For example:

```text
semantics = 4
mask_camera = False
```

means that the voxel is known to be a car voxel from the Actor label but is not considered camera-visible by the conservative visibility rule.

### Conservative camera visibility rule

A voxel may be marked `mask_camera=True` only if:

1. `semantics != 255`.
2. The voxel center has positive camera depth.
3. The center projects inside the 960 x 540 rectified image.
4. The projected pixel passes the Step 5 rectification valid mask.
5. The voxel survives the per-camera z-buffer among known semantic voxels.
6. At least one of the four cameras marks the voxel visible.

Unknown voxels are never marked camera-visible.

### LiDAR policy

```text
mask_lidar = all False
```

The project does not generate virtual LiDAR.

### Observed free-space policy

```text
class 17 observed_free_space is not generated
```

The local Clips do not provide enough reliable first-hit geometry to label observed free space without inventing supervision.

## Formal labels

```text
/home/lab/bev_alpasim_dataset_tools/outcome/occupancy/labels_v0.1/
```

Layout:

```text
labels_v0.1/
  <scene_id>/
    <sample_id>/
      labels.npz
```

Formal label manifest:

```text
/home/lab/bev_alpasim_dataset_tools/outcome/manifests/occupancy_labels_v0.1.jsonl
```

## Formal counts

```text
Scenes:      908
Samples:  24,019
NPZ files: 24,019
Storage: about 280 MB
```

## Full semantic voxel counts

```text
0 others:              688,749
3 bus:                2,589,189
4 car:               61,912,299
7 pedestrian:           324,725
9 trailer:            1,516,622
10 truck:            15,726,369
11 driveable:       256,164,426
255 unknown:      15,033,237,621
```

## Camera-visibility statistics

```text
mask_camera True minimum:      0
P01:                        1,275
Median:                     6,435
P99:                       17,490
Maximum:                   21,882
Mean:                    6,832.89

Visible-known ratio minimum: 0.0
P01:                       0.2466
Median:                    0.4807
P99:                       0.6608
Maximum:                   1.0
Mean:                      0.4794
```

## Edge-case resolution

```text
25 samples: no known semantic geometry
33 samples: all known geometry was rearward or outside every valid camera FOV
4 samples: a very small known set was fully visible
Visibility violations: 0
```

## Performance improvements

The initial voxelizer retained all VectorMaps and reached about 11.1 GiB RSS. It was changed to an active-scene-only cache.

A 200-sample memory trial after the fix used about 91,884 KB RSS.

The full camera-visibility update used:

```text
Elapsed: 6:22.21
Maximum RSS: 108,120 KB
```

## Step 6 final contract

```text
/home/lab/bev_alpasim_dataset_tools/outcome/reports/step6_dataset_contract_v0.1.json
```

Status:

```text
frozen
```

Contract SHA256:

```text
993f6b7cc5429aac3c849df2948755d84043421dd731e7c5fc26f8a2311b5a70
```

## Step 6 formal reports

```text
outcome/reports/occupancy_label_summary_v0.1.json
outcome/reports/occupancy_camera_visibility_summary_v0.1.json
outcome/reports/occupancy_camera_visibility_audit_v0.1.jsonl
outcome/reports/step6_final_audit_v0.1.json
outcome/reports/step6_visibility_edge_cases_v0.1.jsonl
outcome/reports/step6_visibility_edge_diagnosis_v0.1.json
outcome/reports/step6_dataset_contract_v0.1.json
```

---

## 6. Current capability boundary

### Reliably supervised classes

```text
others
bus
car
pedestrian
trailer
truck
driveable_surface
unknown
```

### Classes without a reliable local source in the current dataset

```text
barrier
bicycle
construction_vehicle
motorcycle
traffic_cone
other_flat
sidewalk
terrain
manmade
vegetation
observed_free_space
```

These classes must not be fabricated merely to fill the Occ3D taxonomy.

### Important training interpretation

```text
unknown=255 is not free space
```

Any training loss must ignore semantic ID 255.

---

## 7. Immediate project objective

The immediate objective is now:

> Fine-tune a camera-based BEV semantic occupancy model as quickly as possible, using ALOcc with GDFusion through OccStudio or an equivalent compatible codebase.

Step 7 onward has been simplified. Do not resume the old Scene-Fact, CoC, or Reasoning plan unless the user explicitly changes the objective.

---

## 8. Remaining minimal roadmap

# Step 7: OccStudio / ALOcc dataset adapter

## Goal

Make the BEV training framework load one AlpaSim sample correctly, then scale to all 24,019 samples.

## Step 7A: Audit the training framework

Find the local OccStudio, ALOcc, or GDFusion source tree. If the source is not present locally, ask the user where the repository is located before editing anything.

Audit:

```text
Dataset classes
DataLoader output contract
Configuration system
Camera-count assumptions
Image preprocessing
Temporal-frame organization
Intrinsics format
Extrinsics direction
Ego-motion format
Occupancy-label loading
mask_camera use
mask_lidar use
ignore_index
point_cloud_range
voxel_size
occ_size
num_classes
evaluation implementation
```

Pay special attention to code that assumes six nuScenes cameras. This dataset has four cameras and must not duplicate fake cameras to reach six.

## Step 7B: Minimal AlpaSim occupancy dataset

Each sample must provide the framework-equivalent of:

```text
images or current_images/history_images
4 cameras
2 temporal frames for ordinary samples
K_rect for every image
camera extrinsics
either current/history ego poses or history_to_current transform
semantics [200, 200, 16]
mask_camera [200, 200, 16]
mask_lidar [200, 200, 16]
sample_id
scene_id
is_scene_start
```

Preferred logical image tensor:

```text
[time=2, camera=4, channel=3, height, width]
```

Adapt this layout if the framework expects another ordering.

## Image preparation

For the minimal version, do not pre-generate all rectified images.

Use online loading:

1. Read the original image.
2. Apply the Step 5 rectification remap.
3. Resize for training.
4. Scale `K_rect` consistently.
5. Normalize and convert to tensor.

Cache the small set of shared rectification profiles in Dataset memory.

## Initial training size

Use:

```text
Rectified source: 540 x 960
Training input:    270 x 480
Scale:             0.5
```

The scaled intrinsic matrix must use:

```text
fx *= 0.5
fy *= 0.5
cx *= 0.5
cy *= 0.5
```

## Step 7 acceptance tests

- Read 100 random samples without failure.
- Confirm four current and four history images.
- Confirm all tensor shapes and dtypes.
- Confirm scene-start samples repeat current as history.
- Confirm `K_rect` scaling.
- Confirm camera extrinsic direction with a projection test.
- Confirm history-to-current motion.
- Confirm semantic and mask arrays match the frozen Step 6 files.
- Confirm deterministic repeated loading of the same sample when random augmentation is disabled.
- Generate a visual audit showing the four current images, four history images, occupancy GT, metadata, and projected known geometry.

---

# Step 8: Leakage-free dataset split

## Goal

Create train, validation, and test sets without temporal or map leakage.

## Split grouping

Split by Map ID, not by individual sample.

All Clips and samples sharing a Map ID must remain in the same split. This also prevents the 33 repeated Map IDs from leaking across subsets.

## Initial ratio

```text
Train: 80%
Val:   10%
Test:  10%
```

Use a fixed seed:

```text
20260929
```

## Required products

```text
outcome/splits/train_v0.1.txt
outcome/splits/val_v0.1.txt
outcome/splits/test_v0.1.txt

outcome/manifests/train_samples_v0.1.jsonl
outcome/manifests/val_samples_v0.1.jsonl
outcome/manifests/test_samples_v0.1.jsonl

outcome/reports/split_summary_v0.1.json
outcome/reports/split_contract_v0.1.json
```

## Step 8 acceptance tests

- All 24,019 samples appear exactly once.
- No Map ID crosses splits.
- No Clip crosses splits.
- No sample ID is duplicated.
- All splits are non-empty.
- Each split has usable occurrences of the main supervised classes.
- Freeze seed, hashes, and allocation statistics.

---

# Step 9: Training smoke tests

## Goal

Prove that Dataset, model, loss, backpropagation, checkpointing, and evaluation work before long training.

## Test 1: Overfit eight samples

```text
8 fixed train samples
200 to 500 iterations
strong augmentation disabled
```

Required result:

```text
training loss clearly decreases
small-set occupancy metrics clearly improve
```

Do not begin full training if eight samples cannot be overfitted.

## Test 2: 64-sample smoke training

```text
64 train samples
16 validation samples
1 to 2 epochs
```

Verify:

- stable DataLoader;
- stable memory;
- no NaN or Inf loss;
- checkpoint save and resume;
- validation evaluator;
- per-class IoU output;
- ID 255 ignored by loss and metrics.

## Test 3: Full DataLoader traversal

Traverse the complete training split and record:

```text
load failures
throughput
image shapes
label shapes
class counts
worker stability
```

## Initial hardware-safe settings

The current GPU is an NVIDIA RTX 4070 Ti Super with 16 GB VRAM.

Start with:

```text
Input: 270 x 480
Cameras: 4
Temporal frames: 2
Batch size: 1
Mixed precision: on
Gradient accumulation: 4 or 8
Workers: 4
Task: semantic occupancy only
```

Increase batch size only after measuring memory.

## Loss

Minimum version:

```text
CrossEntropyLoss
ignore_index = 255
```

The model should retain the 18 Occ3D output classes for initial compatibility. Ground-truth-absent classes remain unsupervised rather than being remapped into invented labels.

For the first baseline, use:

```text
valid = semantics != 255
```

Also compute evaluation restricted to:

```text
mask_camera == True
```

Do not silently replace the main training validity rule with `mask_camera` without documenting the experiment.

---

# Step 10: Baseline fine-tuning

## Minimal experiment sequence

### Experiment 1: Single-frame ALOcc baseline

```text
Input: current images only
Output: current semantic occupancy
```

Purpose: verify the dataset and supervision independently of temporal fusion.

### Experiment 2: Target temporal model

```text
Input: history + current images
Motion input: history-to-current ego transform
Model: ALOcc + GDFusion
Output: current semantic occupancy
```

If the training framework requires an intermediate temporal baseline, add it only when needed to isolate an error.

## Required experiment records

Each run must save:

```text
resolved config
training seed
Step 6 contract SHA256
split contract SHA256
last checkpoint
best checkpoint
training log
validation metrics
runtime
peak GPU memory
```

The frozen Step 6 contract hash is:

```text
993f6b7cc5429aac3c849df2948755d84043421dd731e7c5fc26f8a2311b5a70
```

---

# Step 11: Evaluation

## Required metrics

Report per-class IoU for classes that have ground truth:

```text
others
bus
car
pedestrian
trailer
truck
driveable_surface
```

For absent dataset classes, report `N/A` rather than including them as artificial zero-IoU classes in the primary mIoU.

Also report:

```text
known-region mIoU: semantics != 255
camera-visible mIoU: mask_camera == True
occupied-actor IoU
single-frame versus temporal comparison
```

## Visual evaluation

Use fixed validation and test examples showing:

```text
four current images
occupancy GT BEV
occupancy prediction BEV
error map
per-class overlay
```

Check for axis flips, z-level errors, over-expanded roads, positional offsets, and improper inclusion of unknown voxels in metrics.

---

# Step 12: Minimal release freeze

## Required products

```text
dataset_adapter_contract.json
split_contract.json
resolved_training_config.py or .yaml
best_checkpoint
validation_metrics.json
test_metrics.json
evaluation_summary.json
dataset_and_model_release_contract.json
```

## Final frozen information

```text
dataset version
Step 6 contract hash
split hash
camera order
input resolution
temporal length
class table
ignore index
training seed
model configuration
best checkpoint hash
validation and test metrics
known limitations
```

---

## 9. Tasks explicitly out of scope

Do not implement these for the current minimal objective:

```text
Scene-Fact
structured CoC
Reasoning
LLM-generated supervision
natural-language explanations
navigation reasoning
planning targets
driving-action labels
trajectory prediction
occupancy flow
virtual LiDAR
USDZ redownload
static world mesh
building or vegetation pseudo-labels
observed-free labels
large hyperparameter searches
pre-generation of all rectified images unless online loading is proven too slow
```

---

## 10. Training-framework compatibility checklist

Before training, verify every item:

```text
[ ] Four-camera support, with no hard-coded six-camera assumption
[ ] Current and history frame ordering
[ ] Correct K_rect after image resize
[ ] Correct rig/current-ego to camera-optical transform direction
[ ] Correct history-to-current ego-motion transform
[ ] Grid range matches Step 6
[ ] Grid shape is [200, 200, 16]
[ ] Axis order is [x, y, z]
[ ] Semantic dtype is uint8 before tensor conversion
[ ] ignore_index is 255
[ ] Unknown voxels do not contribute to loss or metrics
[ ] mask_lidar being all False does not disable training unexpectedly
[ ] Framework interpretation of mask_camera is understood and documented
[ ] Eighteen output classes are configured
[ ] Dataset-absent classes do not corrupt primary mIoU
[ ] Map-level split prevents leakage
```

---

## 11. Current formal products to preserve

```text
outcome/manifests/occupancy_keyframes_v0.1.jsonl
outcome/manifests/occupancy_sources_v0.1.jsonl
outcome/manifests/occupancy_labels_v0.1.jsonl

outcome/occupancy/labels_v0.1/

outcome/reports/occupancy_label_summary_v0.1.json
outcome/reports/occupancy_camera_visibility_summary_v0.1.json
outcome/reports/occupancy_camera_visibility_audit_v0.1.jsonl
outcome/reports/step6_final_audit_v0.1.json
outcome/reports/step6_visibility_edge_cases_v0.1.jsonl
outcome/reports/step6_visibility_edge_diagnosis_v0.1.json
outcome/reports/step6_dataset_contract_v0.1.json
```

Do not delete or modify these without intentionally rebuilding and re-freezing Step 6.

Review or benchmark directories under `outcome/review/` are temporary and may be removed after confirming that they are not used by a formal contract.

---

## 12. Current test state

At Step 6 freeze:

```text
44 tests passed
2 third-party protobuf deprecation warnings
```

The warnings concern future Python 3.14 compatibility in `google._upb` and do not invalidate the current Python 3.12 pipeline.

Use unique pytest filenames across Step directories. A previous conflict occurred because both Step 5 and Step 6 used `test_finalize.py`; the Step 6 test was renamed to:

```text
tests/step6/test_step6_finalize.py
```

---

## 13. Instructions for the next AI development session

1. Read this README completely before proposing changes.
2. Treat Step 0 through Step 6 as frozen unless a training-adapter incompatibility proves that a specific change is necessary.
3. Do not restart Scene-Fact, CoC, or Reasoning work.
4. Begin with Step 7A by locating and auditing the local OccStudio, ALOcc, or GDFusion repository.
5. If the framework repository path cannot be discovered safely from the local project, ask only for that path.
6. Do not ask the user to repeat dataset history already recorded in this README.
7. Use the formal products and hashes listed here as the source of truth.
8. Implement files directly and provide downloadable patch or installer files when the execution environment cannot write to the user's machine.
9. Run the full project test suite after each coherent code batch using `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
10. Do not inspect, stage, or commit Git changes.
11. Continue automatically after successful validation when no user decision or additional input is required.
12. Keep the objective minimal: first obtain a working, reproducible ALOcc + GDFusion fine-tuning baseline.

---

## 14. One-paragraph handoff summary

The repository has completed and frozen a local-Clip-only data pipeline through Step 6. It discovers 908 AlpaSim Clips, reads all data through a unified API, builds 24,019 stable two-frame four-camera Keyframes, rectifies four cameras to a shared 960 x 540 pinhole contract, and generates 24,019 Occ3D-compatible partial semantic occupancy labels in the current ego frame. Each label has `semantics`, a conservative rectification-aware per-camera-z-buffer `mask_camera`, and an all-False `mask_lidar`. Unknown space is ID 255 and is not free space. No USDZ, remote asset, virtual LiDAR, observed-free label, Scene-Fact, CoC, or Reasoning is used. The immediate next task is to implement an OccStudio/ALOcc dataset adapter, then create a Map-ID-level 80/10/10 split, pass eight-sample and 64-sample training smoke tests, fine-tune a single-frame ALOcc baseline and an ALOcc + GDFusion temporal model, evaluate known-region and camera-visible occupancy metrics, and freeze the resulting dataset/model contracts.

## Full dataset production and Step 7 delivery status

Step 7 DataLoader development and delivery are complete.

- Implementation: `step7/`
- Automated tests: 18 passed
- Reference dataset: 908 Clips, 24,019 samples
- Reference raw root: `/home/lab/data_from_alpasim`
- Reference outcome root: `/home/lab/bev_alpasim_dataset_tools/outcome`
- Full dataset: 2,311 Clips
- Full raw root: `/home/lab/data_all_alpasim`
- Full outcome root: `/home/lab/bev_alpasim_dataset_tools/outcome_full`
- Time order: current, history
- Camera order: cross_left, front_wide, cross_right, front_tele
- Occupancy shape: `[200, 200, 16]`
- Free label: 17
- Ignore label: 255

The 2,311-Clip dataset uses the same frozen schemas and Step 1 through Step 7 code as the verified reference dataset. Only the raw-data and outcome roots change.

Production sequence:

1. Step 1: Clip manifest
2. Step 2: Shared reading and geometry API
3. Step 3: Occupancy Clip profile
4. Step 4: Keyframe manifest
5. Step 5: Calibration census and Rectification
6. Step 6A: Occupancy source manifest
7. Step 6B: Occupancy voxel labels
8. Step 6C: Camera-visibility supervision
9. Step 7: DataLoader and visualization audit

Do not overwrite the verified reference products under `outcome/`. The full build must write only to `outcome_full/`.

Do not copy sample counts, class totals, visibility totals, rejection counts, distribution statistics, or file hashes from the 908-Clip reference build. Recompute all scale-dependent values from the full build.
