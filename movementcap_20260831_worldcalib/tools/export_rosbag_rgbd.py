#!/usr/bin/env python3
"""Export RGB and metric depth from the ROS1 bags written by the Orbbec recorder.

The recorder used for the 2026-08-29 movement-capture delivery writes a
ROS1 bag whose ``sensor_msgs/Image`` definition does not match the actual
wire order.  In particular, the four uint64 fields declared around the image
payload are serialized after it.  This reader deliberately parses that wire
layout instead of relying on an Orbbec SDK version.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import struct

import cv2
import numpy as np
from rosbags.rosbag1 import Reader


RGB_TOPIC = "/cam/sensor_2/frameType_2"
DEPTH_TOPIC = "/cam/sensor_3/frameType_3"
COLOR_PROFILE_TOPIC = "/cam/streamProfileType_2"
DEPTH_PROFILE_TOPIC = "/cam/streamProfileType_3"


@dataclass(frozen=True)
class ImageMessage:
    sequence: int
    stamp_sec: int
    stamp_nsec: int
    frame_id: str
    height: int
    width: int
    encoding: str
    is_bigendian: bool
    step: int
    metadata: bytes
    image_data: bytes
    frame_number: int
    timestamp_usec: int
    timestamp_system_usec: int
    timestamp_global_usec: int


@dataclass(frozen=True)
class VideoProfile:
    stream_type: int
    pixel_format: int
    rotation_matrix: tuple[float, ...]
    translation_mm: tuple[float, ...]
    width: int
    height: int
    fps: int
    intrinsics: tuple[float, ...]
    distortion: tuple[float, ...]
    distortion_model: int


class BufferReader:
    def __init__(self, payload: bytes | memoryview) -> None:
        self.payload = memoryview(payload)
        self.offset = 0

    def unpack(self, fmt: str) -> tuple[int | float, ...]:
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, self.payload, self.offset)
        self.offset += size
        return values

    def read(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.payload):
            raise ValueError(f"Invalid field size {size} at offset {self.offset}")
        result = self.payload[self.offset : self.offset + size].tobytes()
        self.offset += size
        return result

    def ros_string(self) -> str:
        (size,) = self.unpack("<I")
        return self.read(size).decode("utf-8")


def parse_orbbec_image(payload: bytes | memoryview) -> ImageMessage:
    reader = BufferReader(payload)
    sequence, stamp_sec, stamp_nsec = reader.unpack("<III")
    frame_id = reader.ros_string()
    height, width = reader.unpack("<II")
    encoding = reader.ros_string()
    (is_bigendian,) = reader.unpack("<B")
    (step,) = reader.unpack("<I")
    (metadata_size,) = reader.unpack("<I")
    (data_size,) = reader.unpack("<I")
    packed_data = reader.read(data_size)
    if metadata_size > len(packed_data):
        raise ValueError(
            f"Metadata size {metadata_size} exceeds packed data size {len(packed_data)}"
        )
    metadata = packed_data[:metadata_size]
    image_data = packed_data[metadata_size:]
    frame_number, timestamp_usec, timestamp_system_usec, timestamp_global_usec = reader.unpack(
        "<QQQQ"
    )
    if reader.offset != len(reader.payload):
        raise ValueError(
            f"Unexpected trailing bytes: parsed {reader.offset}, payload {len(reader.payload)}"
        )
    return ImageMessage(
        sequence=sequence,
        stamp_sec=stamp_sec,
        stamp_nsec=stamp_nsec,
        frame_id=frame_id,
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=bool(is_bigendian),
        step=step,
        metadata=metadata,
        image_data=image_data,
        frame_number=frame_number,
        timestamp_usec=timestamp_usec,
        timestamp_system_usec=timestamp_system_usec,
        timestamp_global_usec=timestamp_global_usec,
    )


def parse_video_profile(payload: bytes | memoryview) -> VideoProfile:
    reader = BufferReader(payload)
    reader.unpack("<III")
    reader.ros_string()
    stream_type, pixel_format = reader.unpack("<BB")
    rotation_matrix = reader.unpack("<9f")
    translation_mm = reader.unpack("<3f")
    width, height, fps = reader.unpack("<3H")
    intrinsics = reader.unpack("<4f")
    distortion = reader.unpack("<8f")
    (distortion_model,) = reader.unpack("<B")
    if reader.offset != len(reader.payload):
        raise ValueError(
            f"Unexpected profile trailing bytes: parsed {reader.offset}, payload {len(reader.payload)}"
        )
    return VideoProfile(
        stream_type=int(stream_type),
        pixel_format=int(pixel_format),
        rotation_matrix=tuple(float(value) for value in rotation_matrix),
        translation_mm=tuple(float(value) for value in translation_mm),
        width=int(width),
        height=int(height),
        fps=int(fps),
        intrinsics=tuple(float(value) for value in intrinsics),
        distortion=tuple(float(value) for value in distortion),
        distortion_model=int(distortion_model),
    )


def profile_dict(profile: VideoProfile) -> dict[str, object]:
    return {
        "stream_type": profile.stream_type,
        "pixel_format": profile.pixel_format,
        "rotation_matrix_row_major": list(profile.rotation_matrix),
        "translation_mm": list(profile.translation_mm),
        "width": profile.width,
        "height": profile.height,
        "fps": profile.fps,
        "intrinsics_fx_fy_cx_cy": list(profile.intrinsics),
        "distortion_k1_k2_k3_k4_k5_k6_p1_p2": list(profile.distortion),
        "distortion_model": profile.distortion_model,
    }


def decode_rgb(message: ImageMessage) -> np.ndarray:
    encoding = message.encoding.lower()
    if encoding in {"mjpg", "jpeg", "jpg"}:
        image = cv2.imdecode(np.frombuffer(message.image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV failed to decode MJPG frame")
        return image
    if encoding in {"rgb8", "bgr8"}:
        image = np.frombuffer(message.image_data, dtype=np.uint8).reshape(
            message.height, message.width, 3
        )
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image.copy()
    raise ValueError(f"Unsupported RGB encoding: {message.encoding}")


def decode_depth(message: ImageMessage) -> np.ndarray:
    if message.encoding.lower() not in {"mono16", "16uc1"}:
        raise ValueError(f"Unsupported depth encoding: {message.encoding}")
    dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
    expected = message.height * message.step
    if len(message.image_data) != expected:
        raise ValueError(
            f"Depth byte count {len(message.image_data)} does not match height*step {expected}"
        )
    row_values = message.step // dtype.itemsize
    depth = np.frombuffer(message.image_data, dtype=dtype).reshape(message.height, row_values)
    return depth[:, : message.width].astype(np.uint16, copy=False)


def frame_row(index: int, bag_timestamp_ns: int, message: ImageMessage, path: Path) -> list[object]:
    return [
        index,
        path.name,
        bag_timestamp_ns,
        message.sequence,
        message.timestamp_usec,
        message.timestamp_system_usec,
        message.timestamp_global_usec,
        message.frame_number,
        message.width,
        message.height,
        message.encoding,
        message.step,
        len(message.metadata),
        message.metadata.hex(),
    ]


def export_bag(bag_path: Path, output_dir: Path, max_frames: int | None) -> None:
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth_raw_mm"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[list[object]]] = {"rgb": [], "depth": []}
    counts = {"rgb": 0, "depth": 0}
    profiles: dict[str, VideoProfile] = {}
    with Reader(bag_path) as reader:
        wanted = [
            connection
            for connection in reader.connections
            if connection.topic
            in {RGB_TOPIC, DEPTH_TOPIC, COLOR_PROFILE_TOPIC, DEPTH_PROFILE_TOPIC}
        ]
        topics = {connection.topic for connection in wanted}
        expected_topics = {
            RGB_TOPIC,
            DEPTH_TOPIC,
            COLOR_PROFILE_TOPIC,
            DEPTH_PROFILE_TOPIC,
        }
        if topics != expected_topics:
            raise RuntimeError(f"Missing RGB-D topics; found {sorted(topics)}")

        for connection, timestamp_ns, payload in reader.messages(connections=wanted):
            if connection.topic in {COLOR_PROFILE_TOPIC, DEPTH_PROFILE_TOPIC}:
                name = "color" if connection.topic == COLOR_PROFILE_TOPIC else "depth"
                profiles[name] = parse_video_profile(payload)
                continue
            stream = "rgb" if connection.topic == RGB_TOPIC else "depth"
            index = counts[stream]
            if max_frames is not None and index >= max_frames:
                if all(counts[name] >= max_frames for name in counts):
                    break
                continue

            message = parse_orbbec_image(payload)
            path = (rgb_dir if stream == "rgb" else depth_dir) / f"{index:06d}.png"
            image = decode_rgb(message) if stream == "rgb" else decode_depth(message)
            parameters = [] if stream == "rgb" else [cv2.IMWRITE_PNG_COMPRESSION, 0]
            if not cv2.imwrite(str(path), image, parameters):
                raise IOError(f"Failed to write {path}")
            rows[stream].append(frame_row(index, timestamp_ns, message, path))
            counts[stream] += 1

    header = [
        "index",
        "file_name",
        "bag_timestamp_ns",
        "sequence",
        "timestamp_usec",
        "timestamp_system_usec",
        "timestamp_global_usec",
        "frame_number",
        "width",
        "height",
        "encoding",
        "step",
        "metadata_size",
        "metadata_hex",
    ]
    for stream in ("rgb", "depth"):
        csv_path = output_dir / f"{stream}_frames.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows[stream])

    if set(profiles) != {"color", "depth"}:
        raise RuntimeError(f"Missing camera profiles; found {sorted(profiles)}")
    profile_path = output_dir / "camera_profiles.json"
    profile_path.write_text(
        json.dumps(
            {name: profile_dict(profile) for name, profile in profiles.items()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if counts["rgb"] != counts["depth"]:
        raise RuntimeError(f"RGB/depth count mismatch: {counts}")
    print(f"Exported {counts['rgb']} RGB-D frames to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_bag(args.bag, args.output_dir, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
