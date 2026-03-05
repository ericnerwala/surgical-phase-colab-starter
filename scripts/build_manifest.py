#!/usr/bin/env python3
import sys
from pathlib import Path

# Allow running scripts without requiring `pip install -e .` first
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from surgical_phase.data.manifest import build_manifest_from_challenge


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path like .../cholect50-challenge-val")
    ap.add_argument("--out-csv", default="data/processed/manifest.csv")
    ap.add_argument("--phase-map-out", default="data/processed/phase_map.json")
    ap.add_argument("--val-videos", nargs="*", default=["VID73"])
    ap.add_argument("--test-videos", nargs="*", default=["VID75"])
    args = ap.parse_args()

    df = build_manifest_from_challenge(
        data_root=args.data_root,
        out_csv=args.out_csv,
        phase_map_out=args.phase_map_out,
        val_video_ids=args.val_videos,
        test_video_ids=args.test_videos,
    )
    print(df.head())
    print("Saved rows:", len(df))


if __name__ == "__main__":
    main()
