# bev_dataset_tools

Offline tools for building the camera-only AlpaSim BEV/3D occupancy dataset.

## Frozen project paths

- Raw Clips, read-only: `/home/lab/data_from_alpasim`
- Generated artifacts: `/home/lab/bev_alpasim_dataset_tools/outcome`
- Source code: `/home/lab/bev_alpasim_dataset_tools/bev_dataset_tools`

Defaults can be overridden with `BEV_ALPASIM_DATA_ROOT` and `BEV_DATASET_OUTCOME_ROOT`.

## Test command

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

## Step 1 outputs

- `outcome/manifests/clips_v0.1.jsonl`
- `outcome/reports/clip_manifest_summary_v0.1.json`
- `outcome/schemas/clip_schema.md`

`manifest_usable` is a strict Raw Clip completeness indicator. It is not the BEV occupancy eligibility decision. Multi-camera synchronization, image-pose alignment, history pairing, calibration scaling, and geometry readiness are assessed later.
