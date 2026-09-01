from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .delivery import (
    build_delivery,
    inspect_inputs,
    normalize_delivery_provenance,
    normalize_registration_profile,
    publish_web_delivery,
    refresh_auxiliary_hashes,
    validate_delivery,
    write_checksums,
)
from .new_capture import render_new_three
from .imu_mocap_comparison import (
    build_comparison_delivery,
    validate_comparison_delivery,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gt-calib-delivery",
        description="Build and verify the final nine-video GT-calibration delivery.",
    )
    parser.add_argument("--project-root", type=Path, default=project_root())
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect", help="Check all source inputs and system tools.")

    render_new = subparsers.add_parser(
        "render-new", help="Render only Take_007 marker/pose and no-glove videos."
    )
    render_new.add_argument(
        "--destination", type=Path
    )
    render_new.add_argument(
        "--manual-profile",
        type=Path,
        help=(
            "Apply a browser-exported manual XYZ profile. The final-nine schema "
            "uses the Take_007 entries for videos 07/08; the legacy Take_007-only "
            "schema remains supported."
        ),
    )

    build = subparsers.add_parser("build", help="Build the complete nine-video folder.")
    build.add_argument(
        "--destination", type=Path
    )

    comparison = subparsers.add_parser(
        "build-imu-comparison",
        help=(
            "Build four supplemental always-visible IMU-solved pose vs MOCAP "
            "comparison videos without changing the canonical nine-video contract."
        ),
    )
    comparison.add_argument(
        "--destination",
        type=Path,
    )
    comparison.add_argument(
        "--manual-profile",
        type=Path,
    )

    validate_comparison = subparsers.add_parser(
        "validate-imu-comparison",
        help="Validate the supplemental IMU-solved pose vs MOCAP delivery.",
    )
    validate_comparison.add_argument(
        "--destination",
        type=Path,
    )
    validate_comparison.add_argument("--full-decode", action="store_true")
    build.add_argument(
        "--manual-profile",
        type=Path,
        help=(
            "Apply a gt_calib.final_nine_manual_xyz.v1 browser export to videos "
            "01-08 in their respective MOCAP worlds (video 09 is excluded), or "
            "a legacy Take_007-only profile."
        ),
    )

    validate = subparsers.add_parser("validate", help="Validate the delivery manifest and media.")
    validate.add_argument(
        "--destination", type=Path
    )
    validate.add_argument("--full-decode", action="store_true")

    publish = subparsers.add_parser(
        "publish-web", help="Copy a validated delivery into web/public/downloads."
    )
    publish.add_argument(
        "--source", type=Path
    )
    publish.add_argument(
        "--destination",
        type=Path,
    )

    refresh = subparsers.add_parser(
        "refresh-aux-hashes",
        help="Refresh poster/metrics/frame-map hashes after approved visual QA repairs.",
    )
    refresh.add_argument(
        "--destination", type=Path
    )

    normalize = subparsers.add_parser(
        "normalize-provenance",
        help="Replace machine-local paths in delivery metrics with project:// URIs.",
    )
    normalize.add_argument(
        "--destination", type=Path
    )

    serve = subparsers.add_parser("serve", help="Serve the final review website with MP4 ranges.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8811)
    serve.add_argument("--directory", type=Path)
    return parser.parse_args(argv)


def _project_path(value: Path | None, default: Path) -> Path:
    """Resolve a CLI path relative to the selected project-root contract."""

    selected = default if value is None else value
    return selected.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.project_root.expanduser().resolve()
    if args.command == "inspect":
        result = inspect_inputs(root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "pass" else 2
    if args.command == "render-new":
        outputs = render_new_three(
            root,
            _project_path(args.destination, root / "outputs" / "new_capture_review"),
            manual_profile=(
                args.manual_profile.expanduser().resolve()
                if args.manual_profile is not None
                else None
            ),
        )
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
        return 0
    if args.command == "build":
        output = build_delivery(
            root,
            _project_path(args.destination, root / "final_9_video_delivery"),
            manual_profile=(
                args.manual_profile.expanduser().resolve()
                if args.manual_profile is not None
                else None
            ),
        )
        print(output)
        return 0
    if args.command == "build-imu-comparison":
        comparison_profile = _project_path(
            args.manual_profile,
            root
            / "calibration_profiles"
            / "operator_y_minus_44_all_hand_overlays.v1.json",
        )
        output = build_comparison_delivery(
            root,
            _project_path(args.destination, root / "imu_mocap_comparison_delivery"),
            manual_profile=comparison_profile,
        )
        print(output)
        return 0
    if args.command == "validate-imu-comparison":
        result = validate_comparison_delivery(
            _project_path(args.destination, root / "imu_mocap_comparison_delivery"),
            full_decode=args.full_decode,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "pass" else 2
    if args.command == "validate":
        destination = _project_path(args.destination, root / "final_9_video_delivery")
        result = validate_delivery(destination, full_decode=args.full_decode)
        validation_path = destination / "validation.json"
        validation_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "pass" else 2
    if args.command == "publish-web":
        output = publish_web_delivery(
            _project_path(args.source, root / "final_9_video_delivery"),
            _project_path(
                args.destination,
                root / "web" / "public" / "downloads" / "final-nine",
            ),
        )
        print(output)
        return 0
    if args.command == "refresh-aux-hashes":
        output = refresh_auxiliary_hashes(
            _project_path(args.destination, root / "final_9_video_delivery")
        )
        print(output)
        return 0
    if args.command == "normalize-provenance":
        destination = _project_path(args.destination, root / "final_9_video_delivery")
        changed = normalize_delivery_provenance(root, destination)
        if changed:
            # Keep the package valid at command completion.  This refreshes
            # only reviewed auxiliary artifacts; immutable video/profile
            # hashes remain protected by refresh_auxiliary_hashes().
            refresh_auxiliary_hashes(destination)
        changed.extend(normalize_registration_profile(root))
        print(json.dumps([str(path) for path in changed], ensure_ascii=False, indent=2))
        return 0
    if args.command == "serve":
        command = [
            sys.executable,
            str(root / "local_review_server.py"),
            "--host", args.host,
            "--port", str(args.port),
            "--directory", str(_project_path(args.directory, root / "web" / "public")),
        ]
        return subprocess.run(command).returncode
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
