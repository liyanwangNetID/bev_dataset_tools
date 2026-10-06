from __future__ import annotations
import argparse
from .manifests import ManifestIndex


def _scene_id(row):
    keyframe = row["keyframe"]
    return keyframe.get("scene_id") or keyframe.get("clip_id") or ""


def main():
    parser = argparse.ArgumentParser(description="List keyframe samples in an AlpaSim scene")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    index = ManifestIndex()
    rows = []
    for sample_id in index.sample_ids:
        joined = index.get(sample_id)
        if _scene_id(joined) == args.scene_id:
            keyframe = joined["keyframe"]
            rows.append((
                int(keyframe["current_timestamp_ns"]), sample_id,
                bool(keyframe.get("is_scene_start", False)),
                int(keyframe.get("history_delta_ns", 0)),
            ))
    rows.sort()
    if not rows:
        raise SystemExit(f"No samples found for scene: {args.scene_id}")
    print("index\tsample_id\tscene_start\thistory_delta_ms")
    for idx, (_, sample_id, start, delta_ns) in enumerate(rows[:args.limit]):
        print(f"{idx}\t{sample_id}\t{str(start).lower()}\t{delta_ns / 1e6:.3f}")
    print(f"scene_samples={len(rows)} shown={min(len(rows), args.limit)}")


if __name__ == "__main__":
    main()
