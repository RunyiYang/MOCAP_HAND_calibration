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


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        prog="gt-calib-delivery",
        description="Build and verify the final nine-video GT-calibration delivery.",
    )
    parser.add_argument("--project-root", type=Path, default=root)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inspect", help="Check all source inputs and system tools.")

    render_new = subparsers.add_parser(
        "render-new", help="Render only Take_007 marker/pose and no-glove videos."
    )
    render_new.add_argument(
        "--destination", type=Path, default=root / "outputs" / "new_capture_review"
    )
    render_new.add_argument(
        "--manual-profile",
        type=Path,
        help="Apply a gt_calib.manual_xyz_profile.v1 exported by the browser workbench.",
    )

    build = subparsers.add_parser("build", help="Build the complete nine-video folder.")
    build.add_argument(
        "--destination", type=Path, default=root / "final_9_video_delivery"
    )
    build.add_argument(
        "--manual-profile",
        type=Path,
        help="Apply a browser-exported XYZ profile to Take_007 videos 07/08.",
    )

    validate = subparsers.add_parser("validate", help="Validate the delivery manifest and media.")
    validate.add_argument(
        "--destination", type=Path, default=root / "final_9_video_delivery"
    )
    validate.add_argument("--full-decode", action="store_true")

    publish = subparsers.add_parser(
        "publish-web", help="Copy a validated delivery into web/public/downloads."
    )
    publish.add_argument(
        "--source", type=Path, default=root / "final_9_video_delivery"
    )
    publish.add_argument(
        "--destination",
        type=Path,
        default=root / "web" / "public" / "downloads" / "final-nine",
    )

    refresh = subparsers.add_parser(
        "refresh-aux-hashes",
        help="Refresh poster/metrics/frame-map hashes after approved visual QA repairs.",
    )
    refresh.add_argument(
        "--destination", type=Path, default=root / "final_9_video_delivery"
    )

    normalize = subparsers.add_parser(
        "normalize-provenance",
        help="Replace machine-local paths in delivery metrics with project:// URIs.",
    )
    normalize.add_argument(
        "--destination", type=Path, default=root / "final_9_video_delivery"
    )

    serve = subparsers.add_parser("serve", help="Serve the final review website with MP4 ranges.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8811)
    serve.add_argument("--directory", type=Path, default=root / "web" / "public")
    return parser.parse_args(argv)


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
            args.destination.expanduser().resolve(),
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
            args.destination.expanduser().resolve(),
            manual_profile=(
                args.manual_profile.expanduser().resolve()
                if args.manual_profile is not None
                else None
            ),
        )
        print(output)
        return 0
    if args.command == "validate":
        destination = args.destination.expanduser().resolve()
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
            args.source.expanduser().resolve(),
            args.destination.expanduser().resolve(),
        )
        print(output)
        return 0
    if args.command == "refresh-aux-hashes":
        output = refresh_auxiliary_hashes(args.destination.expanduser().resolve())
        print(output)
        return 0
    if args.command == "normalize-provenance":
        changed = normalize_delivery_provenance(
            root, args.destination.expanduser().resolve()
        )
        changed.extend(normalize_registration_profile(root))
        print(json.dumps([str(path) for path in changed], ensure_ascii=False, indent=2))
        return 0
    if args.command == "serve":
        command = [
            sys.executable,
            str(root / "local_review_server.py"),
            "--host", args.host,
            "--port", str(args.port),
            "--directory", str(args.directory.expanduser().resolve()),
        ]
        return subprocess.run(command).returncode
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
