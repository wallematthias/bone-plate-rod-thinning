"""Command-line entry point for derivative batch workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from bone_imaging_derivatives import format_progress_event

from .batch import run_plate_rod_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plate-rod-thinning")
    commands = parser.add_subparsers(dest="command", required=True)
    run_batch = commands.add_parser("run-batch", help="run plate/rod morphometry over a derivative dataset")
    run_batch.add_argument("dataset_root", type=Path)
    run_batch.add_argument("--subject")
    run_batch.add_argument("--site")
    run_batch.add_argument("--output-root", type=Path)
    run_batch.add_argument("--dry-run", action="store_true")
    run_batch.add_argument("--force", action="store_true")
    run_batch.add_argument("--generate-missing", action="store_true")
    run_batch.add_argument("--no-common-region", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-batch":
        run_plate_rod_batch(
            args.dataset_root,
            subject_id=args.subject,
            site=args.site,
            output_root=args.output_root,
            dry_run=args.dry_run,
            force=args.force,
            generate_missing=args.generate_missing,
            use_common_region=not args.no_common_region,
            progress=lambda event: print(format_progress_event(event)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
