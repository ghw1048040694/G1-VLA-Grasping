import io
import ssl
import types
import unittest
import urllib.error

from g1_vla.video_audit import (HTTPRangeReader, RangeReadBudget, VideoAuditError,
                                _codec_evidence, _pinned_video_url, probe_video_samples)


class Response:
    def __init__(self, body, status=206, headers=None):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}
        self.read_calls = 0
        self.closed = False

    def read(self, count):
        self.read_calls += 1
        return self.body.read(count)

    def close(self):
        self.closed = True


class RangeServer:
    def __init__(self, data=b"abcdefghij"):
        self.data = data
        self.responses = []
        self.requested_ranges = []

    def __call__(self, request, timeout):
        field = request.headers["Range"]
        self.requested_ranges.append(field)
        start, end = map(int, field.removeprefix("bytes=").split("-"))
        end = min(end, len(self.data) - 1)
        response = Response(self.data[start:end + 1], headers={
            "Content-Range": f"bytes {start}-{end}/{len(self.data)}", "ETag": '"pinned"'})
        self.responses.append(response)
        return response


class VideoRangeTests(unittest.TestCase):
    def test_seek_and_read_fetch_only_requested_ranges_and_digest_them(self):
        server = RangeServer()
        budget = RangeReadBudget(50)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=4, opener=server)
        self.assertEqual(reader.read(3), b"abc")
        self.assertEqual(reader.seek(-2, io.SEEK_END), 8)
        self.assertEqual(reader.read(2), b"ij")
        self.assertEqual(budget.used_bytes, 6)
        self.assertEqual(server.requested_ranges, ["bytes=0-3", "bytes=8-9"])
        self.assertEqual(reader.receipts[1]["downloaded_bytes"], 2)
        self.assertFalse(reader.receipts[1]["full_file_checksum"])
        self.assertTrue(all(item.closed for item in server.responses))

    def test_cached_bytes_do_not_consume_budget_twice(self):
        budget = RangeReadBudget(4)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=4, opener=RangeServer())
        self.assertEqual(reader.read(4), b"abcd")
        reader.seek(1)
        self.assertEqual(reader.read(2), b"bc")
        self.assertEqual(budget.used_bytes, 4)
        reader.seek(4)
        with self.assertRaisesRegex(VideoAuditError, "budget exhausted"):
            reader.read(1)

    def test_seek_back_to_completed_tail_range_reuses_after_other_ranges(self):
        # Reproduces the real MP4 failure: AVIO read the checked tail, sought
        # elsewhere, then revisited the tail while the transport was failing.
        server = RangeServer(bytes(range(100)))
        calls = []

        def fail_if_tail_requested_twice(request, timeout):
            field = request.headers["Range"]
            calls.append(field)
            if field == "bytes=92-99" and calls.count(field) > 1:
                raise urllib.error.URLError(ssl.SSLEOFError("tail TLS EOF"))
            return server(request, timeout)

        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=16,
                                 opener=fail_if_tail_requested_twice)
        self.assertEqual(reader.read(16), bytes(range(16)))  # Discover source length.
        reader.seek(92)
        self.assertEqual(reader.read(8), bytes(range(92, 100)))
        reader.seek(0)
        self.assertEqual(reader.read(16), bytes(range(16)))
        reader.seek(48)
        self.assertEqual(reader.read(16), bytes(range(48, 64)))
        reader.seek(95)
        self.assertEqual(reader.read(5), bytes(range(95, 100)))
        self.assertEqual(budget.used_bytes, 40)
        self.assertEqual(calls.count("bytes=92-99"), 1)
        self.assertEqual(len(reader.receipts), 3)
        self.assertEqual(reader.cache_evidence()["reused_ranges"][0]["additional_download_bytes"], 0)

    def test_lru_cache_evicts_oldest_and_remains_eight_ranges(self):
        server = RangeServer(bytes(range(100)))
        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=4, opener=server)
        for start in range(0, 32, 4):
            reader.seek(start)
            reader.read(4)
        reader.seek(0)
        reader.read(4)  # Refresh this range, so range 4 becomes least recent.
        reader.seek(32)
        reader.read(4)
        reader.seek(0)
        reader.read(4)
        self.assertEqual(server.requested_ranges.count("bytes=0-3"), 1)
        reader.seek(4)
        reader.read(4)
        self.assertEqual(server.requested_ranges.count("bytes=4-7"), 2)
        self.assertEqual(reader.cache_evidence()["current_completed_ranges"], 8)
        self.assertLessEqual(reader.cache_evidence()["current_cached_bytes"], 2 * 1024 * 1024)

    def test_truncated_body_is_never_cached_for_later_seek(self):
        first = Response(b"ab", headers={"Content-Range": "bytes 0-3/10"})
        server = RangeServer()
        calls = []

        def first_truncated(request, timeout):
            calls.append(1)
            return first if len(calls) == 1 else server(request, timeout)

        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=4, opener=first_truncated)
        with self.assertRaisesRegex(VideoAuditError, "truncated"):
            reader.read(2)
        self.assertEqual(reader.cache_evidence()["current_completed_ranges"], 0)
        reader.seek(0)
        self.assertEqual(reader.read(4), b"abcd")
        self.assertEqual(len(calls), 2)
        self.assertEqual(budget.used_bytes, 6)

    def test_shared_budget_applies_across_files(self):
        budget = RangeReadBudget(5)
        first = HTTPRangeReader("https://one.invalid", budget, read_chunk_bytes=4, opener=RangeServer())
        second = HTTPRangeReader("https://two.invalid", budget, read_chunk_bytes=4, opener=RangeServer())
        first.read(4)
        self.assertEqual(second.read(1), b"a")
        with self.assertRaisesRegex(VideoAuditError, "budget exhausted"):
            second.read(1)
        self.assertEqual(budget.used_bytes, 5)

    def test_full_file_fallback_is_rejected_before_reading_response_body(self):
        response = Response(b"very large body", status=200)
        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, opener=lambda *a, **k: response)
        with self.assertRaisesRegex(VideoAuditError, "no full-file fallback"):
            reader.read(1)
        self.assertEqual(response.read_calls, 0)
        self.assertEqual(budget.used_bytes, 0)
        self.assertTrue(response.closed)

    def test_wrong_range_is_rejected_before_body_read(self):
        response = Response(b"abc", headers={"Content-Range": "bytes 1-3/10"})
        reader = HTTPRangeReader("https://source.invalid", RangeReadBudget(100),
                                 read_chunk_bytes=3, opener=lambda *a, **k: response)
        with self.assertRaisesRegex(VideoAuditError, "does not match"):
            reader.read(1)
        self.assertEqual(response.read_calls, 0)

    def test_truncated_body_is_counted_and_reported(self):
        response = Response(b"ab", headers={"Content-Range": "bytes 0-3/10"})
        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=4,
                                 opener=lambda *a, **k: response)
        with self.assertRaisesRegex(VideoAuditError, "truncated"):
            reader.read(1)
        self.assertEqual(budget.used_bytes, 2)
        self.assertEqual(reader.receipts[0]["downloaded_bytes"], 2)

    def test_unbounded_read_refuses_large_source(self):
        budget = RangeReadBudget(5)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=1, opener=RangeServer())
        with self.assertRaisesRegex(VideoAuditError, "unbounded"):
            reader.read()
        self.assertEqual(budget.used_bytes, 1)

    def test_source_identity_must_be_pinned_and_path_safe(self):
        good = _pinned_video_url("unitree/data", "a" * 40, "videos/cam/file.mp4")
        self.assertIn("/resolve/" + "a" * 40, good)
        for revision, path in [("main", "videos/cam/file.mp4"), ("a" * 40, "videos/../file.mp4")]:
            with self.assertRaises(VideoAuditError):
                _pinned_video_url("unitree/data", revision, path)

    def test_invalid_pts_is_diagnostic_failure_without_http_read(self):
        budget = RangeReadBudget(100)
        report = probe_video_samples({}, "unitree/data", "a" * 40, "videos/cam/file.mp4",
                                     [float("nan")], budget=budget)
        self.assertFalse(report["passed"])
        self.assertEqual(report["read_bytes"], 0)
        self.assertFalse(report["sensor_clock_synchronization_proven"])

    def test_transport_request_retries_are_limited_to_three_and_consume_no_unread_bytes(self):
        calls = []

        def offline(*args, **kwargs):
            calls.append(1)
            raise urllib.error.URLError(ssl.SSLEOFError("TLS EOF"))

        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, opener=offline)
        with self.assertRaises(urllib.error.URLError):
            reader.read(1)
        self.assertEqual(len(calls), 3)
        self.assertEqual(budget.used_bytes, 0)
        self.assertEqual(len(reader.transport_errors), 3)

    def test_recovered_request_keeps_range_protocol(self):
        calls = []
        server = RangeServer()

        def transient(request, timeout):
            calls.append(request.headers["Range"])
            if len(calls) < 3:
                raise urllib.error.URLError(ssl.SSLEOFError("TLS EOF"))
            return server(request, timeout)

        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=4, opener=transient)
        self.assertEqual(reader.read(4), b"abcd")
        self.assertEqual(calls, ["bytes=0-3"] * 3)
        self.assertEqual(budget.used_bytes, 4)
        self.assertEqual(reader.receipts[0]["attempt"], 3)

    def test_partial_payload_before_ssl_error_still_consumes_budget_on_retry(self):
        class InterruptedResponse(Response):
            def read(self, count):
                self.read_calls += 1
                if self.read_calls == 1:
                    return b"a"
                raise ssl.SSLEOFError("TLS EOF after payload")

        response = InterruptedResponse(b"", headers={"Content-Range": "bytes 0-3/10", "ETag": '"pinned"'})
        server = RangeServer()
        calls = []

        def transient(request, timeout):
            calls.append(1)
            return response if len(calls) == 1 else server(request, timeout)

        budget = RangeReadBudget(100)
        reader = HTTPRangeReader("https://source.invalid", budget, read_chunk_bytes=4, opener=transient)
        self.assertEqual(reader.read(4), b"abcd")
        self.assertEqual(budget.used_bytes, 5)
        self.assertEqual([r["downloaded_bytes"] for r in reader.receipts], [1, 4])
        self.assertEqual([r["complete"] for r in reader.receipts], [False, True])

    def test_codec_id_accepts_av1_decoder_alias_but_not_unrelated_codec(self):
        av_stub = types.SimpleNamespace(Codec=lambda name, mode: types.SimpleNamespace(id=226))
        context = types.SimpleNamespace(name="libdav1d", codec=types.SimpleNamespace(id=226))
        report = _codec_evidence(av_stub, context, "av1")
        self.assertEqual(report["decoder_name"], "libdav1d")
        self.assertEqual(report["encoded_codec"], "av1")
        self.assertTrue(report["encoded_codec_verified_by_id"])
        context.codec.id = 27
        self.assertFalse(_codec_evidence(av_stub, context, "av1")["encoded_codec_verified_by_id"])
        self.assertIsNone(_codec_evidence(av_stub, context, "av1")["encoded_codec"])


if __name__ == "__main__":
    unittest.main()
