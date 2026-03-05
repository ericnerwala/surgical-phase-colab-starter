from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


PHASE_NAMES_DEFAULT = {
    0: "preparation",
    1: "calot_triangle_dissection",
    2: "clipping_and_cutting",
    3: "gallbladder_dissection",
    4: "gallbladder_packaging",
    5: "cleaning_and_coagulation",
    6: "gallbladder_retraction",
}


def parse_cholect50_challenge_labels(json_path: Path) -> pd.DataFrame:
    obj = json.loads(json_path.read_text())
    annotations: Dict[str, List[List[float]]] = obj["annotations"]
    rows = []
    for frame_id, frame_anns in annotations.items():
        if not frame_anns:
            continue
        # phase id appears in last position for each triplet entry in challenge labels
        phase = int(frame_anns[0][-1])
        rows.append({"frame_id": int(frame_id), "phase": phase})
    return pd.DataFrame(rows).sort_values("frame_id")


def build_manifest_from_challenge(
    data_root: str,
    out_csv: str,
    phase_map_out: str | None = None,
    val_video_ids: List[str] | None = None,
    test_video_ids: List[str] | None = None,
) -> pd.DataFrame:
    data_root = Path(data_root)
    labels_dir = data_root / "labels"
    videos_dir = data_root / "videos"

    val_video_ids = set(val_video_ids or [])
    test_video_ids = set(test_video_ids or [])

    rows = []
    for json_path in sorted(labels_dir.glob("VID*.json")):
        video_id = json_path.stem
        frame_df = parse_cholect50_challenge_labels(json_path)
        frame_dir = videos_dir / video_id

        for r in frame_df.itertuples(index=False):
            # handle duplicate-style naming in extracted challenge set
            candidates = sorted(frame_dir.glob(f"{r.frame_id:06d}*.png"))
            if not candidates:
                continue
            split = "train"
            if video_id in val_video_ids:
                split = "val"
            if video_id in test_video_ids:
                split = "test"
            rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "frame_id": int(r.frame_id),
                    "frame_path": str(candidates[0]),
                    "phase": int(r.phase),
                }
            )

    df = pd.DataFrame(rows).sort_values(["video_id", "frame_id"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if phase_map_out:
        Path(phase_map_out).write_text(json.dumps(PHASE_NAMES_DEFAULT, indent=2))

    return df
