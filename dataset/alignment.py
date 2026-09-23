"""Causal event selection and explicit old-BVH label mapping."""
from __future__ import annotations
import heapq
import numpy as np


def asof_indices(acquisition: np.ndarray, arrival: np.ndarray, query: np.ndarray,
                 max_age: float = .05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the latest acquisition that has ARRIVED, never a future neighbour.

    Out-of-order arrivals are supported. `fresh` is true only once for each
    selected packet; it prevents treating a held observation as new evidence.
    Returned -1 means unavailable. Empty input streams are supported.
    """
    acquisition, arrival, query = [np.asarray(v, np.float64) for v in (acquisition, arrival, query)]
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in (acquisition, arrival, query)):
        raise ValueError('Clocks must be finite vectors in a common reference clock')
    if len(acquisition) != len(arrival) or (arrival < acquisition).any():
        raise ValueError('Invalid arrival timestamps; verify the clock transform')
    if len(acquisition) > 1 and (np.diff(acquisition) <= 0).any():
        raise ValueError('Acquisition times must increase; split clock resets')
    if (np.diff(query) <= 0).any() or max_age < 0:
        raise ValueError('Queries must increase and max_age must be nonnegative')
    order = np.argsort(arrival, kind='stable')
    indices = np.full(len(query), -1, np.int64)
    ages = np.zeros(len(query), np.float32)
    fresh = np.zeros(len(query), bool)
    cursor, previous = 0, -1
    heap = []
    for k, now in enumerate(query):
        while cursor < len(order) and arrival[order[cursor]] <= now:
            i = int(order[cursor])
            heapq.heappush(heap, (-float(acquisition[i]), i))
            cursor += 1
        if heap:
            i = heap[0][1]
            age = now - acquisition[i]
            if 0 <= age <= max_age:
                indices[k], ages[k] = i, age
                fresh[k] = i != previous
                previous = i
    return indices, ages, fresh


def resample_observation(values: np.ndarray, acquisition: np.ndarray, arrival: np.ndarray,
                         query: np.ndarray, max_age: float = .05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if len(values) != len(acquisition):
        raise ValueError('Values/timestamps length mismatch')
    index, age, fresh = asof_indices(acquisition, arrival, query, max_age)
    out = np.zeros((len(query), *values.shape[1:]), values.dtype)
    valid = index >= 0
    out[valid] = values[index[valid]]
    return out, age, fresh


def old_bvh21_targets(positions: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GT-only map: BVH wrist + joints 5:21 -> native20 joints 4:20.

    Source positions must come from Skeleton_0/1 BVH-FK, not superseded CMA XYZ.
    Root subtraction here constructs LABELS only. Thumb is deliberately masked.
    """
    p, v = np.asarray(positions), np.asarray(valid, bool)
    if p.ndim != 3 or p.shape[1:] != (21, 3) or v.shape != p.shape[:2]:
        raise ValueError('Expected [T,21,3] BVH positions and [T,21] mask')
    out = np.zeros((len(p), 20, 3), np.float32)
    mask = np.zeros((len(p), 20), bool)
    out[:, 4:20] = p[:, 5:21] - p[:, :1]
    mask[:, 4:20] = v[:, 5:21] & v[:, :1]
    out[~mask] = 0
    return out, mask
