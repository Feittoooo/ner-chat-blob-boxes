"""`blob-boxes <mask.png|.npy> -o blob_bounding_boxes.csv`: a thin wrapper around `fit_boxes`."""

import argparse
import sys

from blob_boxes.fit import fit_boxes
from blob_boxes.io import MaskError, load_mask, write_csv

DEFAULT_OUTPUT = "blob_bounding_boxes.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blob-boxes",
        description="Find each blob in a binary mask and the rotated rectangle that maximizes IoU with it.",
    )
    parser.add_argument("mask", help="binary mask: a .npy array or an image (.png, ...); non-zero pixels are blobs")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help=f"output CSV (default {DEFAULT_OUTPUT})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mask = load_mask(args.mask)
    except MaskError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    results = fit_boxes(mask)
    write_csv([(r.blob_id, r.box) for r in results], args.output)

    # The summary goes to stderr so the CSV stays the only data output.
    if results:
        worst = min(results, key=lambda r: r.iou)
        mean = sum(r.iou for r in results) / len(results)
        summary = f"IoU mean {mean:.4f}, min {worst.iou:.4f} (blob {worst.blob_id})"
    else:
        summary = "no blobs found"
    print(f"{len(results)} blobs -> {args.output}; {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
