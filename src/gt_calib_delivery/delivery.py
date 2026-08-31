from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any

import numpy as np

from .manual_profiles import FinalNineManualProfile, load_final_nine_profile
from .new_capture import CALIBRATION_MEMBER, render_new_three, sha256_file


@dataclass(frozen=True)
class OldTake:
    number: int
    segment_key: str
    segment_name: str
    source_first: int
    frame_count: int
    solved_suffix: str


OLD_TAKES = (
    OldTake(
        # BVH ordinal zero is a vendor seed/dummy pose.  Take_000's first
        # otherwise-synchronized RGB row touches that seed, so the formal
        # BVH-backed pair begins one frame later than the former Human.cma pair.
        1, "01", "01_210814_Take_000", 61, 1747,
        "calibration_fit",
    ),
    OldTake(
        2, "02", "02_210955_Take_001", 0, 1805,
        "holdout_from_Take_000",
    ),
    OldTake(
        # Same seed exclusion as Take_000.
        3, "03", "03_211139_Take_002", 4, 1799,
        "holdout_from_Take_000",
    ),
)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _remux_faststart(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:v:0", "-c", "copy",
            "-movflags", "+faststart", str(destination),
        ]
    )


def _transcode_h264(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination),
        ]
    )


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _portable_json_value(value: Any, project_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _portable_json_value(child, project_root)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_portable_json_value(child, project_root) for child in value]
    if isinstance(value, str):
        prefix = str(project_root) + "/"
        if value.startswith(prefix):
            return "project://" + Path(value).relative_to(project_root).as_posix()
    return value


def normalize_delivery_provenance(project_root: Path, destination: Path) -> list[Path]:
    """Replace machine-local project paths in delivery JSON with project URIs.

    The copied manual profile is deliberately excluded: it is an immutable
    operator input whose exact SHA-256 is part of the delivery contract.
    """

    project_root = Path(project_root).resolve()
    destination = Path(destination).resolve()
    changed: list[Path] = []
    json_paths = list((destination / "metrics").glob("*.json"))
    json_paths.extend(
        path
        for path in (destination / "calibration-workbench").glob("*.json")
        if path.name != "applied_manual_profile.json"
    )
    for path in sorted(json_paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        portable = _portable_json_value(payload, project_root)
        if portable == payload:
            continue
        path.write_text(
            json.dumps(portable, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        changed.append(path)
    return changed


def normalize_registration_profile(project_root: Path) -> list[Path]:
    """Normalize the frozen profile that is versioned as a build input."""

    project_root = Path(project_root).resolve()
    path = (
        project_root
        / "outputs"
        / "mocap_root_fusion_review"
        / "01_210814_Take_000_mocap_root_fusion_registration_profile.json"
    )
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    portable = _portable_json_value(payload, project_root)
    if portable == payload:
        return []
    path.write_text(
        json.dumps(portable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [path]


def _require_system_tools() -> dict[str, str]:
    versions = {}
    for executable in ("ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        if path is None:
            raise FileNotFoundError(f"Required system executable is missing: {executable}")
        result = subprocess.run(
            [path, "-version"], check=True, capture_output=True, text=True
        )
        versions[executable] = result.stdout.splitlines()[0]
    return versions


def inspect_inputs(project_root: Path) -> dict[str, Any]:
    root = Path(project_root)
    missing: list[str] = []
    for take in OLD_TAKES:
        take_number = take.segment_name.split("Take_", 1)[1]
        for relative in (
            f"同步整理_20260829_三段/{take.segment_name}/视频/RGB.mp4",
            f"同步整理_20260829_三段/{take.segment_name}/原始BAG与内参/camera_1_rgb_depth.bag",
            f"同步整理_20260829_三段/{take.segment_name}/同步校验/camera_cmavatar_alignment.csv",
            f"同步整理_20260829_三段/{take.segment_name}/同步校验/common_interval_sync_report.json",
            f"同步整理_20260829_三段/{take.segment_name}/动捕/Take_{take_number}/Take_{take_number}_Human.cma",
            f"同步整理_20260829_三段/{take.segment_name}/动捕/Take_{take_number}/Take_{take_number}_Skeleton_0.bvh",
            f"同步整理_20260829_三段/{take.segment_name}/动捕/Take_{take_number}/Take_{take_number}_Skeleton_1.bvh",
            f"同步整理_20260829_三段/{take.segment_name}/手套解算/solved/primary/left_hand_keypoints.csv",
            f"同步整理_20260829_三段/{take.segment_name}/手套解算/solved/primary/right_hand_keypoints.csv",
        ):
            if not (root / relative).is_file():
                missing.append(relative)
    required = (
        "gt_calib_viz.py",
        "mocap_video_overlay.py",
        "bvh_web_export.py",
        "同步整理_20260829_三段/movementcap_worldcalib_pointcloud_package_20260830/movementcap_ruler_worldcalib/results/manual_final/camera_to_world.json",
        "outputs/mocap_root_fusion_review/01_210814_Take_000_mocap_root_fusion_registration_profile.json",
        "movementcap_20260831_worldcalib_tabletop_final.tar.gz",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/rgbd_unpack/RGB.mp4",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/rgbd_unpack/Depth.mp4",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/rgbd_unpack/camera_1_intrinsics.json",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/mocap/Take_007/Take_007.cmm",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/mocap/Take_007/alignment/camera_cmavatar_alignment.csv",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/glove_processing/aligned/primary/aligned_frame_summary.csv",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/glove_processing/solved/primary/left_hand_keypoints.csv",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_161912/glove_processing/solved/primary/right_hand_keypoints.csv",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_155410/rgbd_unpack/RGB.mp4",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_155410/rgbd_unpack/Depth.mp4",
        "thor_new4_20260831_processed/camera_glove_recording_20260831_155410/rgbd_unpack/camera_1_intrinsics.json",
    )
    for relative in required:
        if not (root / relative).is_file():
            missing.append(relative)
    return {
        "schema": "gt_calib.delivery_preflight.v1",
        "status": "pass" if not missing else "fail",
        "expected_video_count": 9,
        "old_take_count": 3,
        "new_action_take": "camera_glove_recording_20260831_161912 / Take_007",
        "no_glove_recording": "camera_glove_recording_20260831_155410",
        "missing": missing,
        "system_tools": _require_system_tools(),
    }


def _render_old_mocap(
    project_root: Path,
    take: OldTake,
    staging: Path,
    translation_xyz_mm: np.ndarray,
) -> tuple[Path, Path, Path, Path]:
    scratch = staging / "_render_scratch" / take.segment_key / "mocap"
    scratch.mkdir(parents=True, exist_ok=True)
    x_mm, y_mm, z_mm = (float(value) for value in translation_xyz_mm)
    command = [
        sys.executable,
        str(project_root / "mocap_video_overlay.py"),
        "--dataset-root", str(project_root / "同步整理_20260829_三段"),
        "--segment", take.segment_key,
        "--output-dir", str(scratch),
        "--output-width", "960",
        "--snapshot-count", "6",
        "--frame-content-samples", "0",
        "--mocap-position-source", "skeleton-bvh",
        "--mocap-world-x-offset-mm", str(x_mm),
        "--mocap-world-y-offset-mm", str(y_mm),
        "--mocap-world-z-offset-mm", str(z_mm),
    ]
    _run(command)
    stem = f"{take.segment_name}_mocap_video_aligned_strict25_skeleton_bvh"
    video = scratch / f"{stem}_h264.mp4"
    poster = scratch / f"{stem}.contact.jpg"
    metrics = scratch / f"{stem}.metrics.json"
    frame_map = scratch / f"{stem}.alignment.csv"
    if not all(path.is_file() for path in (video, poster, metrics, frame_map)):
        raise FileNotFoundError(
            f"Skeleton BVH MOCAP render is incomplete for {take.segment_name}"
        )
    return video, poster, metrics, frame_map


def _render_old_solved(
    project_root: Path,
    take: OldTake,
    staging: Path,
    translation_xyz_mm: np.ndarray,
) -> tuple[Path, Path, Path]:
    scratch = staging / "_render_scratch" / take.segment_key
    scratch.mkdir(parents=True, exist_ok=True)
    profile = (
        project_root / "outputs" / "mocap_root_fusion_review"
        / "01_210814_Take_000_mocap_root_fusion_registration_profile.json"
    )
    command = [
        sys.executable,
        str(project_root / "gt_calib_viz.py"),
        "--dataset-root", str(project_root / "同步整理_20260829_三段"),
        "--segment", take.segment_key,
        "--pose-mode", "mocap-root-fusion",
        "--render-layer", "solved-pose-only",
        "--registration-profile", str(profile),
        "--output-dir", str(scratch),
        "--output-width", "960",
        "--start-frame", str(take.source_first),
        "--max-frames", str(take.frame_count),
        "--snapshot-count", "6",
        "--operator-world-x-mm", str(float(translation_xyz_mm[0])),
        "--operator-world-y-mm", str(float(translation_xyz_mm[1])),
        "--operator-world-z-mm", str(float(translation_xyz_mm[2])),
    ]
    _run(command)
    stem = f"{take.segment_name}_solved_hand_pose_{take.solved_suffix}"
    video = scratch / f"{stem}.mp4"
    poster = scratch / f"{stem}.contact.jpg"
    metrics = scratch / f"{stem}.metrics.json"
    if not all(path.is_file() for path in (video, poster, metrics)):
        raise FileNotFoundError(f"Old solved-pose render is incomplete for {take.segment_name}")
    return video, poster, metrics


def _extract_calibration(project_root: Path, destination: Path) -> None:
    archive_path = project_root / "movementcap_20260831_worldcalib_tabletop_final.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(CALIBRATION_MEMBER)
        if not member.isfile():
            raise ValueError("Calibration archive member is not a regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError(CALIBRATION_MEMBER)
        payload = handle.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration:format=duration,size",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    return {
        "codec": stream["codec_name"],
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": numerator / denominator,
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(stream.get("duration") or payload["format"]["duration"]),
        "bytes": int(payload["format"]["size"]),
    }


def _faststart(path: Path) -> bool:
    data = path.read_bytes()
    moov = data.find(b"moov")
    mdat = data.find(b"mdat")
    return moov >= 0 and mdat >= 0 and moov < mdat


def _delivery_video_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    number = 1
    for take in OLD_TAKES:
        base = f"take{take.number:02d}"
        specs.append(
            {
                "order": number,
                "id": f"{base}-mocap",
                "group": base,
                "label": f"Take {take.number:02d} · MOCAP",
                "variant": "skeleton_0_1_bvh_fk_21_joint",
                "filename": f"{number:02d}_take{take.number:02d}_mocap.mp4",
                "expected_frames": take.frame_count,
                "source_rgb_first": take.source_first,
                "source_rgb_last": take.source_first + take.frame_count - 1,
                "poster": f"posters/{number:02d}_take{take.number:02d}_mocap.jpg",
                "metrics": f"metrics/{number:02d}_take{take.number:02d}_mocap.json",
                "frame_map": f"frame_maps/take{take.number:02d}_rgb_to_mocap.csv",
                "semantics": "latest requested Skeleton_0/1 BVH forward kinematics, 21 joints per hand; Human.cma is used only for its validated 120 Hz timestamp axis",
            }
        )
        number += 1
        specs.append(
            {
                "order": number,
                "id": f"{base}-solved",
                "group": base,
                "label": f"Take {take.number:02d} · Solved pose",
                "variant": "glove_solved_20_joint_only",
                "filename": f"{number:02d}_take{take.number:02d}_solved_pose.mp4",
                "expected_frames": take.frame_count,
                "source_rgb_first": take.source_first,
                "source_rgb_last": take.source_first + take.frame_count - 1,
                "poster": f"posters/{number:02d}_take{take.number:02d}_solved_pose.jpg",
                "metrics": f"metrics/{number:02d}_take{take.number:02d}_solved_pose.json",
                "frame_map": f"frame_maps/take{take.number:02d}_rgb_to_mocap.csv",
                "semantics": "glove local articulation conditioned on synchronized MOCAP wrist SE(3); not independent glove 6DoF",
            }
        )
        number += 1
    specs.extend(
        (
            {
                "order": 7,
                "id": "take007-mocap-markers",
                "group": "take007",
                "label": "Take_007 · Labeled CMM markers",
                "variant": "twenty_surface_markers_plus_virtual_wrist",
                "filename": "07_take007_labeled_mocap_markers.mp4",
                "expected_frames": 1981,
                "source_rgb_first": 0,
                "source_rgb_last": 1980,
                "poster": "posters/07_take007_labeled_mocap_markers.jpg",
                "metrics": "metrics/07_take007_labeled_mocap_markers.json",
                "frame_map": "frame_maps/take007_rgb_to_cmm.csv",
                "semantics": "20 photographed CMM surface markers (10/hand); marker #11 is hidden and conditions a VIZ-only virtual wrist 20 mm proximal",
            },
            {
                "order": 8,
                "id": "take007-solved",
                "group": "take007",
                "label": "Take_007 · Aligned solved pose",
                "variant": "cmm_palm_conditioned_glove_articulation",
                "filename": "08_take007_aligned_hand_pose.mp4",
                "expected_frames": 1981,
                "source_rgb_first": 0,
                "source_rgb_last": 1980,
                "poster": "posters/08_take007_aligned_hand_pose.jpg",
                "metrics": "metrics/08_take007_aligned_hand_pose.json",
                "frame_map": "frame_maps/take007_rgb_to_cmm.csv",
                "semantics": "glove 20-joint/hand articulation with virtual wrist and an anchored similarity fit to all ten photographed CMM surface markers",
            },
            {
                "order": 9,
                "id": "no-glove-calibration",
                "group": "calibration",
                "label": "No glove · World calibration",
                "variant": "cs400_world_axes_calibration_evidence",
                "filename": "09_no_glove_world_calibration.mp4",
                "expected_frames": 98,
                "source_rgb_first": 0,
                "source_rgb_last": 97,
                "poster": "posters/09_no_glove_world_calibration.jpg",
                "metrics": "metrics/09_no_glove_world_calibration.json",
                "frame_map": None,
                "semantics": "CS-400 world axes and marker plane; calibration evidence, not hand GT",
            },
        )
    )
    return specs


def _write_readme(destination: Path, profile: FinalNineManualProfile) -> None:
    if profile.source_path is None:
        profile_note = (
            "- No manual profile was supplied; all operator XYZ translations are zero. "
            "Video 9 remains excluded because it has no hand MOCAP."
        )
    else:
        applied = [
            video_id
            for video_id, annotation in profile.annotations.items()
            if annotation.apply_translation
        ]
        excluded = [
            video_id
            for video_id, annotation in profile.annotations.items()
            if not annotation.apply_translation
        ]
        profile_note = (
            f"- Applied manual profile `{profile.source_path.name}` with global XYZ "
            f"{profile.global_world_xyz_mm.tolist()} mm. Applied video IDs: "
            f"{', '.join(applied)}. Excluded video IDs: {', '.join(excluded)}. "
            "Every value is interpreted in that video's own MOCAP world; this is "
            "an operator-selected display correction, not one shared cross-session "
            "extrinsic and not independent GT evidence."
        )

    text = """# Final nine-video delivery

This folder is the reviewed, browser-ready delivery generated by the UV project
in the parent directory.

## Video inventory

1. Take 01 MOCAP from Skeleton_0/1 BVH forward kinematics
2. Take 01 solved hand pose
3. Take 02 MOCAP from Skeleton_0/1 BVH forward kinematics
4. Take 02 solved hand pose
5. Take 03 MOCAP from Skeleton_0/1 BVH forward kinematics
6. Take 03 solved hand pose
7. Take_007 labeled CMM markers: 10 measured surface points per hand plus a virtual wrist
8. Take_007 solved hand pose aligned from the corrected CMM palm/root mapping
9. Matching no-glove 155410 CS-400 world/camera calibration evidence

All media are H.264/yuv420p, 960x540, 30 fps, fast-start MP4.  See
`manifest.json` for exact frame counts, provenance, hashes, and semantics.

Important boundaries:

- Videos 1/3/5 use the requested detailed Skeleton_0/1 BVH joint positions.
  Human.cma supplies only the validated frame-counter/timestamp axis for those
  renders; its joint-position columns are not used.
{profile_note}
- Old solved-pose videos are MOCAP-wrist-SE(3)-conditioned visualizations, not
  independent glove wrist 6DoF.
- The placement photograph maps CMM marker #1..#10 to five fingertip/base pairs.
  Marker #11 is not shown as a joint; it only conditions a virtual wrist located
  another 20 mm toward the forearm.
- Take_007 video 8 fixes the root at the virtual wrist and estimates a frozen
  scale plus per-frame rotation from all ten photographed surface markers.
  This same-take fit is visualization alignment, not independent accuracy GT.
- Take_007 RGB frame 1981 is excluded because the final timecode anchor closes
  on frame 1980.  No frame is held or fabricated.
- Video 9 uses the matching no-glove 155410 RGB, its own intrinsics, and the
  2026-08-31 CS-400 calibration.  It is not dynamic hand GT.
- `calibration-workbench/` contains clean RGB and binary 3D trajectories for the
  browser XYZ tool.  The browser can select all nine videos, stores a per-video
  XYZ residual, and exposes side-specific residuals only for Take_007 videos
  7/8.  Its `gt_calib.final_nine_manual_xyz.v1` export can be passed back to the
  UV build.  Exported manual offsets are operator calibration, not GT.
""".format(profile_note=profile_note)
    (destination / "README.md").write_text(text, encoding="utf-8")


def write_checksums(destination: Path) -> Path:
    """Write deterministic hashes for every regular delivery artifact.

    The checksum file itself is excluded.  Calling this again after validation
    also adds ``validation.json`` without changing any reviewed media.
    """

    destination = Path(destination).resolve()
    lines = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "SHA256SUMS.txt":
            continue
        relative = path.relative_to(destination).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    output = destination / "SHA256SUMS.txt"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _delivery_artifact(
    destination: Path,
    relative: Any,
    *,
    label: str,
) -> Path:
    """Resolve one manifest artifact without allowing package escape/symlinks."""

    destination = Path(destination).resolve()
    if (
        not isinstance(relative, str)
        or not relative
        or ":" in relative
        or "\\" in relative
    ):
        raise ValueError(f"{label} path must be a non-empty relative string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} path escapes the delivery: {relative!r}")
    candidate = destination / relative_path
    current = destination
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {relative!r}")
    try:
        artifact = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {relative}") from exc
    if not artifact.is_relative_to(destination):
        raise ValueError(f"{label} path escapes the delivery: {relative!r}")
    if not artifact.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {relative}")
    return artifact


def build_delivery(
    project_root: Path,
    destination: Path,
    *,
    manual_profile: Path | None = None,
) -> Path:
    project_root = Path(project_root).resolve()
    destination = Path(destination).resolve()
    preflight = inspect_inputs(project_root)
    if preflight["status"] != "pass":
        raise FileNotFoundError("Preflight failed: " + ", ".join(preflight["missing"]))
    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing delivery or staging directory: {destination}"
        )
    for name in ("videos", "posters", "metrics", "frame_maps", "calibration"):
        (staging / name).mkdir(parents=True, exist_ok=True)

    profile = load_final_nine_profile(manual_profile)

    def translation(video_id: str) -> np.ndarray:
        value = profile.effective_world_xyz_mm(video_id)
        return (
            np.zeros(3, dtype=np.float64)
            if value is None
            else np.asarray(value, dtype=np.float64)
        )

    for take in OLD_TAKES:
        mocap_order = (take.number - 1) * 2 + 1
        solved_order = mocap_order + 1
        mocap_id = f"take{take.number:02d}-mocap"
        solved_id = f"take{take.number:02d}-solved"
        mocap_render, mocap_poster, mocap_metrics, frame_map = _render_old_mocap(
            project_root, take, staging, translation(mocap_id)
        )
        _remux_faststart(
            mocap_render,
            staging / "videos" / f"{mocap_order:02d}_take{take.number:02d}_mocap.mp4",
        )
        _copy_file(
            mocap_poster,
            staging / "posters" / f"{mocap_order:02d}_take{take.number:02d}_mocap.jpg",
        )
        _copy_file(
            mocap_metrics,
            staging / "metrics" / f"{mocap_order:02d}_take{take.number:02d}_mocap.json",
        )
        _copy_file(
            frame_map,
            staging / "frame_maps" / f"take{take.number:02d}_rgb_to_mocap.csv",
        )
        rendered, poster, metrics = _render_old_solved(
            project_root, take, staging, translation(solved_id)
        )
        _transcode_h264(
            rendered,
            staging / "videos" / f"{solved_order:02d}_take{take.number:02d}_solved_pose.mp4",
        )
        _copy_file(
            poster,
            staging / "posters" / f"{solved_order:02d}_take{take.number:02d}_solved_pose.jpg",
        )
        _copy_file(
            metrics,
            staging / "metrics" / f"{solved_order:02d}_take{take.number:02d}_solved_pose.json",
        )

    # Rendering uses high-bitrate MP4V intermediates.  They are build scratch,
    # not part of the downloadable delivery, and can more than double its size.
    shutil.rmtree(staging / "_render_scratch")

    render_new_three(project_root, staging, manual_profile=manual_profile)
    normalize_delivery_provenance(project_root, staging)
    _extract_calibration(
        project_root, staging / "calibration" / "camera_to_world.json"
    )
    _write_readme(staging, profile)

    specs = _delivery_video_specs()
    videos = []
    for spec in specs:
        path = staging / "videos" / spec["filename"]
        probe = _probe_video(path)
        if probe["frame_count"] != spec["expected_frames"]:
            raise ValueError(
                f"Frame count mismatch for {path.name}: {probe['frame_count']} != {spec['expected_frames']}"
            )
        item = dict(spec)
        item.update(probe)
        item["sha256"] = sha256_file(path)
        item["poster_sha256"] = sha256_file(staging / item["poster"])
        item["metrics_sha256"] = sha256_file(staging / item["metrics"])
        item["frame_map_sha256"] = (
            sha256_file(staging / item["frame_map"])
            if item["frame_map"] is not None
            else None
        )
        item["faststart"] = _faststart(path)
        videos.append(item)
    applied_profile = staging / "calibration-workbench" / "applied_manual_profile.json"
    reproducibility_command = (
        "uv run gt-calib-delivery build "
        "--destination final_9_video_delivery.rebuilt"
    )
    if applied_profile.is_file():
        reproducibility_command += (
            " --manual-profile "
            "final_9_video_delivery/calibration-workbench/applied_manual_profile.json"
        )
    manifest = {
        "schema": "gt_calib.final_nine_video_delivery.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "video_count": len(videos),
        "groups": 5,
        "videos": videos,
        "calibration": {
            "path": "calibration/camera_to_world.json",
            "sha256": sha256_file(staging / "calibration" / "camera_to_world.json"),
            "source_archive": "movementcap_20260831_worldcalib_tabletop_final.tar.gz",
            "source_archive_sha256": sha256_file(
                project_root / "movementcap_20260831_worldcalib_tabletop_final.tar.gz"
            ),
        },
        "calibration_workbench": {
            "path": "calibration-workbench/take007_alignment.json",
            "sha256": sha256_file(
                staging / "calibration-workbench" / "take007_alignment.json"
            ),
            "clean_rgb": {
                "path": "calibration-workbench/take007_clean_rgb.mp4",
                "sha256": sha256_file(
                    staging / "calibration-workbench" / "take007_clean_rgb.mp4"
                ),
            },
            "mocap_trajectory": {
                "path": "calibration-workbench/take007_mocap.f32",
                "sha256": sha256_file(
                    staging / "calibration-workbench" / "take007_mocap.f32"
                ),
            },
            "solved_trajectory": {
                "path": "calibration-workbench/take007_solved.f32",
                "sha256": sha256_file(
                    staging / "calibration-workbench" / "take007_solved.f32"
                ),
            },
            "applied_manual_profile": (
                profile.provenance()
                if applied_profile.is_file()
                else None
            ),
        },
        "reproducibility": {
            "command": reproducibility_command,
            "system_tools": preflight["system_tools"],
        },
        "claim_boundary": (
            "Videos demonstrate synchronization, projection, and conditioned composition. "
            "They do not establish independent dynamic hand-pose ground-truth accuracy."
        ),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_checksums(staging)
    staging.rename(destination)
    return destination


def validate_delivery(destination: Path, *, full_decode: bool = False) -> dict[str, Any]:
    destination = Path(destination).resolve()
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    def checked_artifact(relative: Any, label: str) -> Path | None:
        try:
            return _delivery_artifact(destination, relative, label=label)
        except (FileNotFoundError, ValueError) as exc:
            failures.append(str(exc))
            return None

    manifest_videos = manifest.get("videos")
    if not isinstance(manifest_videos, list):
        failures.append("manifest videos must be a list")
        manifest_videos = []
    if manifest.get("schema") != "gt_calib.final_nine_video_delivery.v1":
        failures.append("manifest schema mismatch")
    if manifest.get("video_count") != 9 or len(manifest_videos) != 9:
        failures.append("manifest must declare and contain exactly 9 videos")
    videos = sorted((destination / "videos").glob("*.mp4"))
    if len(videos) != 9:
        failures.append(f"expected exactly 9 MP4 files, found {len(videos)}")
    expected_names = {
        item.get("filename")
        for item in manifest_videos
        if isinstance(item, dict) and isinstance(item.get("filename"), str)
    }
    actual_names = {path.name for path in videos}
    if expected_names != actual_names:
        failures.append("manifest/video filename set mismatch")
    probes: dict[str, dict[str, Any]] = {}
    for item in manifest_videos:
        if not isinstance(item, dict):
            failures.append("manifest video entry must be an object")
            continue
        filename = item.get("filename")
        if not isinstance(filename, str):
            failures.append("manifest video filename must be a string")
            continue
        path = checked_artifact(f"videos/{filename}", f"video {filename}")
        if path is None:
            continue
        probe = _probe_video(path)
        probes[path.name] = probe
        expected = {
            "codec": "h264", "pixel_format": "yuv420p", "width": 960,
            "height": 540, "frame_count": item["expected_frames"],
        }
        for key, value in expected.items():
            if probe.get(key) != value:
                failures.append(f"{path.name}: {key}={probe.get(key)!r}, expected {value!r}")
        if not abs(probe["fps"] - 30.0) < 1e-6:
            failures.append(f"{path.name}: fps={probe['fps']}")
        if sha256_file(path) != item["sha256"]:
            failures.append(f"{path.name}: SHA256 mismatch")
        if not _faststart(path):
            failures.append(f"{path.name}: moov atom is not before mdat")
        for path_key, hash_key in (
            ("poster", "poster_sha256"),
            ("metrics", "metrics_sha256"),
            ("frame_map", "frame_map_sha256"),
        ):
            relative = item.get(path_key)
            expected_hash = item.get(hash_key)
            if relative is None:
                if expected_hash is not None:
                    failures.append(
                        f"{path.name}: {hash_key} must be null when {path_key} is null"
                    )
                continue
            artifact = checked_artifact(relative, f"{path.name} {path_key}")
            if not isinstance(expected_hash, str):
                failures.append(f"missing {hash_key}: {relative}")
            elif artifact is not None and sha256_file(artifact) != expected_hash:
                failures.append(f"{relative}: SHA256 mismatch")
        if full_decode:
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", str(path),
                    "-map", "0:v:0", "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or result.stderr.strip():
                failures.append(f"{path.name}: full decode failed: {result.stderr.strip()}")
    for first, second in ((1, 2), (3, 4), (5, 6), (7, 8)):
        a = next(
            (
                item
                for item in manifest_videos
                if isinstance(item, dict) and item.get("order") == first
            ),
            None,
        )
        b = next(
            (
                item
                for item in manifest_videos
                if isinstance(item, dict) and item.get("order") == second
            ),
            None,
        )
        if a and b and probes.get(a.get("filename"), {}).get("frame_count") != probes.get(b.get("filename"), {}).get("frame_count"):
            failures.append(f"pair {first}/{second} frame counts differ")
    workbench = manifest.get("calibration_workbench", {})
    if not isinstance(workbench, dict):
        failures.append("calibration_workbench must be an object")
        workbench = {}
    workbench_artifacts: dict[str, Path] = {}
    for label, entry in (
        ("metadata", workbench),
        ("clean_rgb", workbench.get("clean_rgb", {})),
        ("mocap_trajectory", workbench.get("mocap_trajectory", {})),
        ("solved_trajectory", workbench.get("solved_trajectory", {})),
    ):
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            failures.append(f"calibration workbench {label} manifest entry is incomplete")
            continue
        artifact = checked_artifact(relative, f"calibration workbench {label}")
        if artifact is None:
            continue
        workbench_artifacts[label] = artifact
        if sha256_file(artifact) != expected_hash:
            failures.append(f"calibration workbench {label} SHA256 mismatch")

    metadata_path = workbench_artifacts.get("metadata")
    if metadata_path is not None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"calibration workbench metadata is invalid JSON: {exc}")
            metadata = None
        if isinstance(metadata, dict):
            metadata_parent = Path(str(workbench.get("path"))).parent
            metadata_entries = (
                ("clean_rgb", metadata.get("video")),
                (
                    "mocap_trajectory",
                    metadata.get("layers", {}).get("mocap")
                    if isinstance(metadata.get("layers"), dict)
                    else None,
                ),
                (
                    "solved_trajectory",
                    metadata.get("layers", {}).get("solved")
                    if isinstance(metadata.get("layers"), dict)
                    else None,
                ),
            )
            for label, inner in metadata_entries:
                if not isinstance(inner, dict):
                    failures.append(f"calibration workbench metadata {label} entry is incomplete")
                    continue
                inner_path = inner.get("path")
                inner_hash = inner.get("sha256")
                if not isinstance(inner_path, str) or not isinstance(inner_hash, str):
                    failures.append(f"calibration workbench metadata {label} entry is incomplete")
                    continue
                artifact = checked_artifact(
                    (metadata_parent / inner_path).as_posix(),
                    f"calibration workbench metadata {label}",
                )
                outer_artifact = workbench_artifacts.get(label)
                if artifact is not None and outer_artifact is not None and artifact != outer_artifact:
                    failures.append(f"calibration workbench metadata {label} path disagrees with manifest")
                if artifact is not None and sha256_file(artifact) != inner_hash:
                    failures.append(f"calibration workbench metadata {label} SHA256 mismatch")
    profile = workbench.get("applied_manual_profile")
    if profile is not None:
        if not isinstance(profile, dict):
            failures.append("calibration workbench applied_manual_profile must be an object or null")
        else:
            relative = profile.get("path")
            expected_hash = profile.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                failures.append("calibration workbench applied_manual_profile entry is incomplete")
            else:
                artifact = checked_artifact(
                    relative, "calibration workbench applied_manual_profile"
                )
                if artifact is not None and sha256_file(artifact) != expected_hash:
                    failures.append("calibration workbench applied_manual_profile SHA256 mismatch")
    return {
        "schema": "gt_calib.delivery_validation.v1",
        "status": "pass" if not failures else "fail",
        "full_decode": full_decode,
        "video_count": len(videos),
        "failures": failures,
        "probes": probes,
    }


def refresh_auxiliary_hashes(destination: Path) -> Path:
    """Refresh reviewed non-video artifact hashes after an approved repair.

    Video and applied-profile hashes are intentionally immutable here.  A
    changed profile requires a rebuild because it is a render input.  Binary
    workbench hashes are refreshed in both the manifest and its metadata JSON
    so browsers cannot retain stale cache keys.
    """

    destination = Path(destination).resolve()
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workbench = manifest.get("calibration_workbench")
    if workbench is not None and not isinstance(workbench, dict):
        raise ValueError("Invalid calibration-workbench manifest entry")

    # Fail before writing anything if the immutable render profile changed.
    if isinstance(workbench, dict):
        profile = workbench.get("applied_manual_profile")
        if profile is not None:
            if not isinstance(profile, dict):
                raise ValueError("Invalid applied_manual_profile manifest entry")
            profile_path = _delivery_artifact(
                destination,
                profile.get("path"),
                label="calibration workbench applied_manual_profile",
            )
            expected_profile_hash = profile.get("sha256")
            if not isinstance(expected_profile_hash, str):
                raise ValueError("Applied manual profile SHA256 is missing")
            if sha256_file(profile_path) != expected_profile_hash:
                raise ValueError(
                    "Applied manual profile changed; rebuild all rendered videos "
                    "instead of refreshing auxiliary hashes"
                )

    for item in manifest.get("videos", []):
        for path_key, hash_key in (
            ("poster", "poster_sha256"),
            ("metrics", "metrics_sha256"),
            ("frame_map", "frame_map_sha256"),
        ):
            relative = item.get(path_key)
            if relative is None:
                item[hash_key] = None
                continue
            artifact = _delivery_artifact(
                destination,
                relative,
                label=f"video auxiliary artifact {path_key}",
            )
            item[hash_key] = sha256_file(artifact)

    if isinstance(workbench, dict):
        metadata_path = _delivery_artifact(
            destination,
            workbench.get("path"),
            label="calibration workbench metadata",
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Calibration-workbench metadata must be an object")
        layers = metadata.get("layers")
        if not isinstance(layers, dict):
            raise ValueError("Calibration-workbench metadata layers are missing")
        metadata_parent = Path(str(workbench.get("path"))).parent
        mapped_entries = (
            ("clean_rgb", metadata.get("video")),
            ("mocap_trajectory", layers.get("mocap")),
            ("solved_trajectory", layers.get("solved")),
        )
        for label, inner in mapped_entries:
            outer = workbench.get(label)
            if not isinstance(outer, dict) or not isinstance(inner, dict):
                raise ValueError(f"Invalid calibration-workbench {label} entry")
            outer_path = _delivery_artifact(
                destination,
                outer.get("path"),
                label=f"calibration workbench {label}",
            )
            inner_relative = inner.get("path")
            inner_path = _delivery_artifact(
                destination,
                (metadata_parent / str(inner_relative)).as_posix()
                if isinstance(inner_relative, str)
                else inner_relative,
                label=f"calibration workbench metadata {label}",
            )
            if inner_path != outer_path:
                raise ValueError(
                    f"Calibration-workbench metadata {label} path disagrees with manifest"
                )
            digest = sha256_file(outer_path)
            outer["sha256"] = digest
            inner["sha256"] = digest
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        workbench["sha256"] = sha256_file(metadata_path)
    manifest["auxiliary_hashes_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_checksums(destination)
    return manifest_path


def publish_web_delivery(source: Path, destination: Path) -> Path:
    """Copy one validated delivery into the static website without symlinks.

    The operation is deliberately atomic and refuses to overwrite an existing
    publication.  This keeps the downloadable files identical to the reviewed
    local folder and prevents a partially copied website from being served.
    """

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    validation = validate_delivery(source, full_decode=False)
    if validation["status"] != "pass":
        raise ValueError(
            "Refusing to publish an invalid delivery: "
            + "; ".join(validation["failures"])
        )
    if (source / "_render_scratch").exists():
        raise ValueError("Refusing to publish build scratch")

    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() or staging.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing web delivery: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging, symlinks=False)
    if any(path.is_symlink() for path in staging.rglob("*")):
        shutil.rmtree(staging)
        raise ValueError("Symlinks are forbidden in the web delivery")
    staging.rename(destination)
    return destination
