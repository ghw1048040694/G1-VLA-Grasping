"""Bounded, read-only source-video probes for external dataset audits.

Byte-range digests attest only to the bytes fetched.  They are deliberately
not presented as a full-file checksum or evidence of physical sensor sync.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import math
import re
import struct
import urllib.parse
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from collections import OrderedDict
from typing import Any


class VideoAuditError(ValueError):
    pass


class RangeReadBudget:
    """One instance must be shared by all video probes in an audit run."""

    def __init__(self, max_read_bytes: int):
        if isinstance(max_read_bytes, bool) or not isinstance(max_read_bytes, int) or max_read_bytes <= 0:
            raise VideoAuditError("max_read_bytes must be a positive integer")
        self.max_read_bytes = max_read_bytes
        self.used_bytes = 0

    @property
    def remaining_bytes(self) -> int:
        return self.max_read_bytes - self.used_bytes

    def consume(self, count: int) -> None:
        self.used_bytes += count
        if self.used_bytes > self.max_read_bytes:
            raise VideoAuditError("shared video HTTP read budget exhausted")


class HTTPRangeReader(io.RawIOBase):
    """Seekable AVIO input with mandatory HTTP 206 and a bounded byte budget."""

    def __init__(self, url: str, budget: RangeReadBudget, *, read_chunk_bytes: int = 262144,
                 timeout_s: float = 30.0, opener=None):
        super().__init__()
        if not 0 < read_chunk_bytes <= 2 * 1024 * 1024:
            raise VideoAuditError("read_chunk_bytes must be positive and at most 2 MiB")
        self.url = url
        self.budget = budget
        self.read_chunk_bytes = read_chunk_bytes
        self.timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen
        self._position = 0
        self._size = None
        self._cache_start = 0
        self._cache = b""
        self._completed_cache = OrderedDict()
        self._completed_cache_bytes = 0
        self._cache_reuses = {}
        self._etag = None
        self.receipts: list[dict[str, Any]] = []
        self.transport_errors: list[dict[str, Any]] = []

    def _store_completed_range(self, start: int, data: bytes):
        # Only a fully received, protocol-validated 206 body reaches here.
        key = (start, len(data))
        if key in self._completed_cache:
            old, _ = self._completed_cache.pop(key)
            self._completed_cache_bytes -= len(old)
        self._completed_cache[key] = (data, hashlib.sha256(data).hexdigest())
        self._completed_cache_bytes += len(data)
        while len(self._completed_cache) > 8 or self._completed_cache_bytes > 2 * 1024 * 1024:
            _, (evicted, _) = self._completed_cache.popitem(last=False)
            self._completed_cache_bytes -= len(evicted)
        self._cache_start, self._cache = start, data

    def _activate_completed_range(self, position: int):
        for key in reversed(self._completed_cache):
            start, count = key
            if start <= position < start + count:
                data, digest = self._completed_cache[key]
                self._completed_cache.move_to_end(key)
                self._cache_start, self._cache = start, data
                return key, digest
        return None

    def cache_evidence(self):
        return {"max_completed_ranges": 8, "max_cached_bytes": 2 * 1024 * 1024,
                "current_completed_ranges": len(self._completed_cache),
                "current_cached_bytes": self._completed_cache_bytes,
                "reused_ranges": list(self._cache_reuses.values())}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    @property
    def size_bytes(self):
        if self._size is None:
            self._fetch(0)
        return self._size

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._position + offset
        elif whence == io.SEEK_END:
            target = self.size_bytes + offset
        else:
            raise VideoAuditError("invalid seek mode")
        if target < 0:
            raise VideoAuditError("negative byte seek")
        self._position = int(target)
        return self._position

    def _fetch(self, start: int):
        # Three transport attempts at most. Protocol failures (including a
        # server ignoring Range) are never retried as full-file downloads.
        for attempt in range(1, 4):
            before = self.budget.used_bytes
            try:
                self._fetch_attempt(start, attempt)
                return
            except urllib.error.HTTPError:
                raise
            except (urllib.error.URLError, OSError, http.client.IncompleteRead) as error:
                self.transport_errors.append({"attempt": attempt, "start_byte": start,
                                              "downloaded_bytes": self.budget.used_bytes - before,
                                              "error": f"{type(error).__name__}: {error}"})
                if attempt == 3:
                    raise

    def _fetch_attempt(self, start: int, attempt: int):
        if self.budget.remaining_bytes <= 0:
            raise VideoAuditError("shared video HTTP read budget exhausted")
        count = min(self.read_chunk_bytes, self.budget.remaining_bytes)
        if self._size is not None:
            count = min(count, self._size - start)
        if count <= 0:
            return
        end = start + count - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                                          "Accept-Encoding": "identity"})
        response = self._opener(request, timeout=self.timeout_s)
        receipt = None
        pieces = []
        complete = False
        try:
            status = getattr(response, "status", None) or response.getcode()
            # A 200 response could be a multi-GB video. Never consume its body.
            if status != 206:
                raise VideoAuditError(f"HTTP Range required: received {status}, no full-file fallback")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range.strip())
            if match is None:
                raise VideoAuditError("missing or invalid Content-Range")
            actual_start, actual_end, total = map(int, match.groups())
            if (actual_start != start or actual_end < actual_start or actual_end > end or
                    total <= actual_end or actual_end != min(end, total - 1)):
                raise VideoAuditError("Content-Range does not match requested bounded range")
            if self._size is not None and total != self._size:
                raise VideoAuditError("source video size changed between byte requests")
            etag = response.headers.get("ETag")
            if self._etag is not None and etag is not None and self._etag != etag:
                raise VideoAuditError("source video ETag changed between byte requests")
            expected_count = actual_end - actual_start + 1
            self._size = total
            self._etag = etag
            receipt = {"start_byte": actual_start,
                       "requested_end_byte": end, "response_content_range": content_range,
                       "source_size_bytes": total, "http_status": status,
                       "etag": etag, "attempt": attempt, "full_file_checksum": False}
            received_count = 0
            while received_count < expected_count:
                try:
                    part = response.read(min(65536, expected_count - received_count))
                except http.client.IncompleteRead as error:
                    if error.partial:
                        self.budget.consume(len(error.partial))
                        pieces.append(error.partial)
                    raise
                if not part:
                    break
                # Charge every returned payload chunk immediately. A retry
                # cannot erase bytes consumed before a later transport error.
                self.budget.consume(len(part))
                pieces.append(part)
                received_count += len(part)
            data = b"".join(pieces)
            if len(data) != expected_count:
                raise VideoAuditError("truncated HTTP Range body")
            complete = True
            self._store_completed_range(start, data)
        finally:
            if receipt is not None:
                fetched = b"".join(pieces)
                receipt.update(end_byte=start + len(fetched) - 1, downloaded_bytes=len(fetched),
                               sha256=hashlib.sha256(fetched).hexdigest(), complete=complete)
                self.receipts.append(receipt)
            response.close()

    def read(self, size=-1):
        self._checkClosed()
        if size == 0:
            return b""
        if size is None or size < 0:
            remaining = self.size_bytes - self._position
            if remaining > self.budget.remaining_bytes:
                raise VideoAuditError("unbounded video read exceeds shared budget")
            size = remaining
        pieces = []
        while size > 0:
            if self._size is not None and self._position >= self._size:
                break
            cached = self._activate_completed_range(self._position)
            if cached is None:
                self._fetch(self._position)
            cache_index = self._position - self._cache_start
            available = self._cache[cache_index:cache_index + size]
            if not available:
                break
            if cached is not None:
                key, digest = cached
                reuse = self._cache_reuses.setdefault(key, {"start_byte": key[0],
                    "end_byte": key[0] + key[1] - 1, "reference_download_sha256": digest,
                    "reuse_count": 0, "reused_read_bytes": 0, "additional_download_bytes": 0})
                reuse["reuse_count"] += 1
                reuse["reused_read_bytes"] += len(available)
            pieces.append(available)
            self._position += len(available)
            size -= len(available)
        return b"".join(pieces)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def _png_bytes(rgb, width: int, height: int) -> bytes:
    pixels = rgb.tobytes()
    if len(pixels) != width * height * 3:
        raise VideoAuditError("RGB frame byte count does not match source dimensions")

    def chunk(kind, content):
        return (struct.pack(">I", len(content)) + kind + content +
                struct.pack(">I", zlib.crc32(kind + content) & 0xffffffff))

    stride = width * 3
    filtered = b"".join(b"\0" + pixels[start:start + stride]
                         for start in range(0, len(pixels), stride))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(filtered)) + chunk(b"IEND", b""))


def _pinned_video_url(repo_id: str, revision: str, video_path: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise VideoAuditError("invalid dataset repo_id")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise VideoAuditError("video revision must be a pinned 40-character commit")
    if (video_path.startswith("/") or ".." in video_path.split("/") or
            not video_path.startswith("videos/") or not video_path.endswith(".mp4")):
        raise VideoAuditError("invalid dataset video path")
    return ("https://huggingface.co/datasets/" + repo_id + "/resolve/" + revision + "/" +
            urllib.parse.quote(video_path, safe="/"))


def _codec_evidence(av_module, codec_context, expected_codec: str) -> dict:
    # Decoder implementation names (e.g. libdav1d) are not the encoded
    # protocol. Verify libavcodec's exact codec ID, rather than a name allowlist.
    expected_id = int(av_module.Codec(expected_codec, "r").id)
    actual_id = int(codec_context.codec.id)
    matches = actual_id == expected_id
    return {"decoder_name": codec_context.name, "codec_id": actual_id,
            "expected_codec_id": expected_id, "expected_encoded_codec": expected_codec,
            "encoded_codec": expected_codec if matches else None,
            "encoded_codec_verified_by_id": matches}


def probe_video_samples(config: dict, repo_id: str, revision: str, video_path: str,
                        expected_source_pts_s: list[float], *, budget: RangeReadBudget,
                        output_dir: Path | None = None, sample_ids: list[str] | None = None,
                        expected_width: int = 640, expected_height: int = 480,
                        expected_fps: float = 30.0, expected_codec: str = "av1") -> dict:
    """Decode selected absolute source PTS, retaining the unmodified RGB frames.

    The caller derives PTS from episode video from_timestamp plus the parquet
    row timestamp.  This probes format conversion, not original sensor clocks.
    """
    result = {"schema_version": 1, "video_path": video_path, "repo_id": repo_id,
              "revision": revision, "passed": False, "status": "blocked", "errors": [],
              "samples": [], "http_range_receipts": [], "full_video_sha256": None,
              "sensor_clock_synchronization_proven": False, "time_resampled": False,
              "read_bytes": 0, "budget_used_bytes": budget.used_bytes}
    reader = None
    start_used = budget.used_bytes
    try:
        url = _pinned_video_url(repo_id, revision, video_path)
        pts_targets = [float(x) for x in expected_source_pts_s]
        if not pts_targets or any(not math.isfinite(x) or x < 0 for x in pts_targets):
            raise VideoAuditError("source PTS must be a nonempty list of finite nonnegative seconds")
        if not math.isfinite(expected_fps) or expected_fps <= 0:
            raise VideoAuditError("expected source fps must be positive")
        ids = sample_ids or [f"sample-{index:03d}" for index in range(len(pts_targets))]
        if len(ids) != len(pts_targets) or len(ids) != len(set(ids)):
            raise VideoAuditError("sample IDs must be unique and match the PTS count")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", item) for item in ids):
            raise VideoAuditError("sample IDs must be safe filename components")
        import av

        video_config = config.get("external_data", {}).get("video", {})
        chunk_bytes = int(video_config.get("read_chunk_bytes", 262144))
        timeout_s = float(video_config.get("timeout_s", 30.0))
        max_frames = int(video_config.get("max_decode_frames_per_sample", 360))
        if max_frames <= 0:
            raise VideoAuditError("max_decode_frames_per_sample must be positive")
        reader = HTTPRangeReader(url, budget, read_chunk_bytes=chunk_bytes, timeout_s=timeout_s)
        with av.open(reader, mode="r") as container:
            if not container.streams.video:
                raise VideoAuditError("source file has no video stream")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            codec = _codec_evidence(av, stream.codec_context, expected_codec)
            result["stream"] = {"index": stream.index, "width": stream.width, "height": stream.height,
                                "average_fps": fps, **codec, "time_base_s": float(stream.time_base),
                                "duration_ticks": stream.duration, "reported_frames": stream.frames}
            if (stream.width != expected_width or stream.height != expected_height or fps is None or
                    abs(fps - expected_fps) > 1e-6 or not codec["encoded_codec_verified_by_id"]):
                raise VideoAuditError("source video stream disagrees with pinned feature metadata")
            tolerance_s = 0.5 / expected_fps + 1e-6
            for sample_id, target in zip(ids, pts_targets):
                container.seek(int(target / float(stream.time_base)), stream=stream, backward=True,
                               any_frame=False)
                closest = None
                closest_error = float("inf")
                decoded_count = 0
                for frame in container.decode(stream):
                    decoded_count += 1
                    if frame.pts is None:
                        raise VideoAuditError("decoded source frame lacks PTS")
                    actual = float(frame.pts * frame.time_base)
                    error = abs(actual - target)
                    if error < closest_error:
                        closest, closest_error = frame, error
                    if actual >= target or decoded_count >= max_frames:
                        break
                if closest is None or closest_error > tolerance_s:
                    raise VideoAuditError(f"{sample_id}: no frame within half source frame of requested PTS")
                if closest.width != expected_width or closest.height != expected_height:
                    raise VideoAuditError("decoded source frame changed dimensions")
                rgb = closest.to_ndarray(format="rgb24")
                rgb_bytes = rgb.tobytes()
                sample = {"sample_id": sample_id, "requested_source_pts_s": target,
                          "decoded_source_pts_s": float(closest.pts * closest.time_base),
                          "decoded_pts_ticks": int(closest.pts), "pts_error_s": closest_error,
                          "allowed_pts_error_s": tolerance_s, "decoded_frames": decoded_count,
                          "width": closest.width, "height": closest.height, "pixel_format": "rgb24",
                          "rgb_sha256": hashlib.sha256(rgb_bytes).hexdigest(), "png_path": None}
                if output_dir is not None:
                    directory = Path(output_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    camera_prefix = hashlib.sha256(video_path.encode()).hexdigest()[:12]
                    png_path = directory / f"{camera_prefix}-{sample_id}.png"
                    png_data = _png_bytes(rgb, closest.width, closest.height)
                    with png_path.open("xb") as handle:
                        handle.write(png_data)
                    sample["png_path"] = str(png_path)
                    sample["png_sha256"] = hashlib.sha256(png_data).hexdigest()
                result["samples"].append(sample)
            result["passed"] = True
            result["status"] = "sampled_format_and_pts_passed"
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if reader is not None:
            result["http_range_receipts"] = reader.receipts
            result["http_transport_errors"] = reader.transport_errors
            result["http_range_cache"] = reader.cache_evidence()
            result["source_size_bytes"] = reader._size
            reader.close()
        result["read_bytes"] = budget.used_bytes - start_used
        result["budget_used_bytes"] = budget.used_bytes
        result["budget_max_read_bytes"] = budget.max_read_bytes
    return result
