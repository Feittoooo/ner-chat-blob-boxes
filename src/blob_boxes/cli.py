"""`blob-boxes <mask.png|.npy> -o blob_bounding_boxes.csv`."""

import argparse
import sys

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
    build_parser().parse_args(argv)
    print("blob-boxes: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
