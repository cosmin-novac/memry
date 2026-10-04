"""The optional offsite copy: uploaded to a temporary key, checked, copied
into place; a failure leaves the object already in the bucket as it was."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import unquote

import httpx
import pytest

from memry import offsite
from memry.config import Config, SnapshotConfig
from memry.offsite import EMPTY_SHA256, S3Client, sign_v4, upload_snapshot
from memry.providers.embeddings import HashEmbedder
from memry.providers.llm import NoneLLM
from memry.snapshot import take_snapshot
from memry.store import MemoryStore


def test_signature_v4_matches_the_aws_example():
    """The GET Object example of the AWS Signature Version 4 documentation."""
    headers = sign_v4(
        "GET", "examplebucket.s3.amazonaws.com", "/test.txt", {"Range": "bytes=0-9"},
        payload_hash=EMPTY_SHA256, key_id="AKIAIOSFODNN7EXAMPLE",
        secret="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", region="us-east-1",
        now=datetime(2013, 5, 24, tzinfo=timezone.utc),
    )
    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )


class FakeS3:
    """Just enough of S3 for the offsite copy: PUT, PUT with a copy source,
    HEAD and DELETE, path-style, with knobs to make one step fail."""

    def __init__(self, bucket: str = "memry-backups") -> None:
        self.bucket = bucket
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.log: list[tuple[str, str]] = []
        self.truncate_uploads = False
        self.fail_copy = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=key-id/")
        bucket, _, key = unquote(request.url.path).lstrip("/").partition("/")
        assert bucket == self.bucket
        method = request.method
        if method == "PUT" and "x-amz-copy-source" in request.headers:
            self.log.append(("COPY", key))
            if self.fail_copy:
                return httpx.Response(500, text="<Error><Code>InternalError</Code></Error>")
            source = unquote(request.headers["x-amz-copy-source"]).split("/", 2)[2]
            self.objects[key] = self.objects[source]
            return httpx.Response(200, text="<CopyObjectResult/>")
        if method == "PUT":
            body = request.read()
            assert request.headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
            assert int(request.headers["content-length"]) == len(body)
            meta = {k[len("x-amz-meta-"):]: v for k, v in request.headers.items()
                    if k.startswith("x-amz-meta-")}
            self.log.append(("PUT", key))
            self.objects[key] = (body[:-1] if self.truncate_uploads else body, meta)
            return httpx.Response(200)
        if method == "HEAD":
            self.log.append(("HEAD", key))
            if key not in self.objects:
                return httpx.Response(404)
            body, meta = self.objects[key]
            headers = {"content-length": str(len(body))}
            headers.update({f"x-amz-meta-{k}": v for k, v in meta.items()})
            return httpx.Response(200, headers=headers)
        if method == "DELETE":
            self.log.append(("DELETE", key))
            self.objects.pop(key, None)
            return httpx.Response(204)
        return httpx.Response(405)


def offsite_config(db_path, target) -> Config:
    return Config(db_path=str(db_path), snapshot=SnapshotConfig(
        dir=str(target), offsite_url="https://acct.r2.cloudflarestorage.com",
        offsite_bucket="memry-backups", offsite_key_id="key-id", offsite_secret="secret",
    ))


@pytest.fixture
def snap(tmp_path):
    config = offsite_config(tmp_path / "data" / "memry.db", tmp_path / "backups")
    store = MemoryStore(config, llm=NoneLLM(), embedder=HashEmbedder(64))
    store.add("Ada likes tea.", user_id="ada", infer=False)
    result = take_snapshot(config, offsite=False)
    assert result["ok"]
    yield config, store, tmp_path / "backups", result
    store.close()


def client_for(fake: FakeS3) -> S3Client:
    return S3Client("https://acct.r2.cloudflarestorage.com", fake.bucket, "key-id", "secret",
                    transport=httpx.MockTransport(fake.handler))


def test_off_unless_a_bucket_is_configured():
    assert not offsite.configured(Config())
    assert not offsite.configured(Config(snapshot=SnapshotConfig(offsite_url="https://x")))


def test_upload_goes_to_a_temporary_key_is_checked_then_copied_into_place(snap):
    config, _, target, manifest = snap
    fake = FakeS3()
    result = upload_snapshot(config, target, manifest, client=client_for(fake))

    assert result["ok"], result
    assert sorted(fake.objects) == ["memry/memry.db.gz", "memry/snapshot.json"]
    body, meta = fake.objects["memry/memry.db.gz"]
    assert gzip.decompress(body) == (target / "memry.db").read_bytes()
    assert meta["sha256"] == hashlib.sha256(body).hexdigest()
    assert meta["source-sha256"] == manifest["files"]["memry.db"]["sha256"]
    uploaded = json.loads(fake.objects["memry/snapshot.json"][0])
    assert uploaded["files"] == manifest["files"]
    steps = [(op, key) for op, key in fake.log if "memry.db" in key]
    temp = steps[0][1]
    assert temp.startswith("memry/memry.db.gz.") and temp.endswith(".tmp")
    assert steps == [("PUT", temp), ("HEAD", temp), ("COPY", "memry/memry.db.gz"),
                     ("HEAD", "memry/memry.db.gz"), ("DELETE", temp)]
    assert not list(target.glob(".*.tmp"))  # the local gzip is gone too


def test_a_damaged_upload_keeps_the_object_already_there(snap):
    config, _, target, manifest = snap
    fake = FakeS3()
    fake.objects["memry/memry.db.gz"] = (b"yesterday", {"sha256": "old"})
    fake.truncate_uploads = True
    result = upload_snapshot(config, target, manifest, client=client_for(fake))

    assert result["ok"] is False and "did not arrive whole" in result["error"]
    assert fake.objects == {"memry/memry.db.gz": (b"yesterday", {"sha256": "old"})}
    assert ("COPY", "memry/memry.db.gz") not in fake.log


def test_a_failed_copy_keeps_the_object_already_there(snap):
    config, _, target, manifest = snap
    fake = FakeS3()
    fake.objects["memry/memry.db.gz"] = (b"yesterday", {"sha256": "old"})
    fake.fail_copy = True
    result = upload_snapshot(config, target, manifest, client=client_for(fake))

    assert result["ok"] is False and "copy" in result["error"]
    assert fake.objects == {"memry/memry.db.gz": (b"yesterday", {"sha256": "old"})}


def test_a_snapshot_records_the_offsite_result_and_survives_its_failure(tmp_path):
    config = offsite_config(tmp_path / "data" / "memry.db", tmp_path / "backups")
    store = MemoryStore(config, llm=NoneLLM(), embedder=HashEmbedder(64))
    try:
        store.add("Ada likes tea.", user_id="ada", infer=False)
        fake = FakeS3()
        done = take_snapshot(config, offsite_client=client_for(fake))
        assert done["ok"] and done["offsite"]["ok"]
        recorded = json.loads((tmp_path / "backups" / "snapshot.json").read_text("utf-8"))
        assert recorded["offsite"]["ok"] and recorded["offsite"]["bucket"] == "memry-backups"

        fake.fail_copy = True
        again = take_snapshot(config, offsite_client=client_for(fake))
        assert again["ok"] and again["offsite"]["ok"] is False  # the local copy stands
    finally:
        store.close()


def test_missing_credentials_fail_the_upload_not_the_snapshot(snap):
    config, _, target, manifest = snap
    bare = config.model_copy(update={"snapshot": config.snapshot.model_copy(
        update={"offsite_secret": None})})
    result = upload_snapshot(bare, target, manifest)
    assert result["ok"] is False and "OFFSITE_SECRET" in result["error"]
