"""The optional offsite copy of the nightly snapshot, in an S3-compatible bucket.

Off unless ``MEMRY_SNAPSHOT_OFFSITE_URL`` and ``MEMRY_SNAPSHOT_OFFSITE_BUCKET``
are set. Cloudflare R2 is the suggested store (10 GB free, no egress fees);
Backblaze B2 and Contabo Object Storage speak the same protocol. memry.tech's
own server runs local-only for now: its one disk holds both the database and
the local copy, so that copy survives a damaged file but not a lost disk, and
this module is what closes that gap when it is turned on.

After a good local snapshot each file is gzipped and uploaded so that the
object already there is replaced only by one that arrived whole:

1. upload to a temporary key, with the gzip's sha256 as object metadata;
2. read the temporary object's size and sha256 back (HEAD) and compare;
3. copy it to the final key (a server-side copy, atomic for readers), check
   that too, and delete the temporary key.

A failure at any step leaves the final object as it was. ``snapshot.json``
goes up last, the same way, so the manifest in the bucket describes the files
beside it. Requests are signed with AWS Signature Version 4 over httpx, so no
SDK is needed.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote, urlsplit

import httpx

from .config import Config

log = logging.getLogger("memry")

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CHUNK = 1024 * 1024


class OffsiteError(RuntimeError):
    pass


def configured(config: Config) -> bool:
    snap = config.snapshot
    return bool(snap.offsite_url and snap.offsite_bucket)


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def sign_v4(
    method: str,
    host: str,
    path: str,
    headers: dict[str, str],
    *,
    payload_hash: str,
    key_id: str,
    secret: str,
    region: str,
    now: datetime,
    query: str = "",
    service: str = "s3",
) -> dict[str, str]:
    """The headers to send: ``headers`` plus host, x-amz-date,
    x-amz-content-sha256 and the Authorization of AWS Signature Version 4.
    ``path`` is already URI-encoded; every header given is signed."""
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    day = amz_date[:8]
    signed = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    signed["host"] = host
    signed["x-amz-date"] = amz_date
    signed["x-amz-content-sha256"] = payload_hash
    names = sorted(signed)
    canonical = "\n".join([
        method,
        path,
        query,
        "".join(f"{name}:{signed[name]}\n" for name in names),
        ";".join(names),
        payload_hash,
    ])
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        scope,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    ])
    key = _hmac(_hmac(_hmac(_hmac(f"AWS4{secret}".encode("utf-8"), day), region), service),
                "aws4_request")
    signature = hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    out = {**headers, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
    out["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={key_id}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    return out


class S3Client:
    """The four calls the offsite copy needs, path-style (``/bucket/key``),
    which R2, B2 and Contabo all accept."""

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        key_id: str,
        secret: str,
        region: str = "auto",
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        parts = urlsplit(endpoint.rstrip("/"))
        self.base = f"{parts.scheme}://{parts.netloc}"
        self.host = parts.netloc
        self.root = parts.path.rstrip("/")
        self.bucket = bucket
        self.key_id = key_id
        self.secret = secret
        self.region = region or "auto"
        self.http = httpx.Client(transport=transport, timeout=timeout)

    def close(self) -> None:
        self.http.close()

    def _path(self, key: str) -> str:
        return f"{self.root}/{quote(self.bucket, safe='')}/{quote(key, safe='/-_.~')}"

    def _send(
        self,
        method: str,
        key: str,
        headers: dict[str, str] | None = None,
        *,
        content: Any = None,
        payload_hash: str = EMPTY_SHA256,
        extra: dict[str, str] | None = None,
    ) -> httpx.Response:
        path = self._path(key)
        signed = sign_v4(
            method, self.host, path, headers or {},
            payload_hash=payload_hash, key_id=self.key_id, secret=self.secret,
            region=self.region, now=datetime.now(timezone.utc),
        )
        signed.update(extra or {})  # sent, not signed (Content-Length)
        return self.http.request(method, self.base + path, headers=signed, content=content)

    def put_file(self, key: str, path: Path, *, sha256: str, metadata: dict[str, str]) -> None:
        size = path.stat().st_size

        def body() -> Iterator[bytes]:
            with open(path, "rb") as fh:
                yield from iter(lambda: fh.read(CHUNK), b"")

        headers = {f"x-amz-meta-{k}": v for k, v in metadata.items()}
        headers["content-type"] = "application/octet-stream"
        response = self._send("PUT", key, headers, content=body(), payload_hash=sha256,
                              extra={"Content-Length": str(size)})
        if response.status_code >= 300:
            raise OffsiteError(f"upload of {key} failed: HTTP {response.status_code} {response.text[:200]}")

    def head(self, key: str) -> dict[str, Any] | None:
        response = self._send("HEAD", key)
        if response.status_code == 404:
            return None
        if response.status_code >= 300:
            raise OffsiteError(f"HEAD {key} failed: HTTP {response.status_code}")
        return {
            "size": int(response.headers.get("content-length", "-1")),
            "sha256": response.headers.get("x-amz-meta-sha256"),
        }

    def copy(self, source_key: str, dest_key: str) -> None:
        source = f"/{self.bucket}/{quote(source_key, safe='/-_.~')}"
        response = self._send("PUT", dest_key, {"x-amz-copy-source": source})
        # S3 can answer a failed copy with 200 and an <Error> body.
        if response.status_code >= 300 or "<Error>" in response.text:
            raise OffsiteError(f"copy to {dest_key} failed: HTTP {response.status_code} {response.text[:200]}")

    def delete(self, key: str) -> None:
        response = self._send("DELETE", key)
        if response.status_code >= 300 and response.status_code != 404:
            raise OffsiteError(f"delete of {key} failed: HTTP {response.status_code}")


def client_from_config(config: Config, transport: httpx.BaseTransport | None = None) -> S3Client:
    snap = config.snapshot
    if not (snap.offsite_key_id and snap.offsite_secret):
        raise OffsiteError("set MEMRY_SNAPSHOT_OFFSITE_KEY_ID and MEMRY_SNAPSHOT_OFFSITE_SECRET")
    return S3Client(
        snap.offsite_url or "", snap.offsite_bucket or "", snap.offsite_key_id,
        snap.offsite_secret, snap.offsite_region, transport=transport,
    )


def _gzip(source: Path, dest: Path) -> tuple[str, int]:
    """Gzip ``source`` to ``dest`` (no name or time in the header, so the same
    file gives the same bytes); the gzip's sha256 and size."""
    digest = hashlib.sha256()
    with open(source, "rb") as src, open(dest, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            for chunk in iter(lambda: src.read(CHUNK), b""):
                gz.write(chunk)
    with open(dest, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest(), dest.stat().st_size


def replace_object(client: S3Client, key: str, path: Path, sha256: str, size: int,
                   stamp: str, metadata: dict[str, str] | None = None) -> None:
    """Put ``path`` at ``key``, replacing what is there only once the upload
    is complete and checked; on failure the object at ``key`` is unchanged."""
    temp_key = f"{key}.{stamp}.tmp"
    try:
        client.put_file(temp_key, path, sha256=sha256, metadata={"sha256": sha256, **(metadata or {})})
        uploaded = client.head(temp_key)
        if not uploaded or uploaded["size"] != size or uploaded["sha256"] != sha256:
            raise OffsiteError(f"the upload of {key} did not arrive whole: {uploaded}")
        client.copy(temp_key, key)
        final = client.head(key)
        if not final or final["size"] != size or final["sha256"] != sha256:
            raise OffsiteError(f"the copy to {key} does not match the upload: {final}")
    finally:
        try:
            client.delete(temp_key)
        except Exception as exc:  # a stray temporary object costs storage, not data
            log.warning("offsite: could not delete %s: %s", temp_key, exc)


def upload_snapshot(
    config: Config,
    directory: Path,
    manifest: dict[str, Any],
    *,
    client: S3Client | None = None,
) -> dict[str, Any]:
    """Upload the snapshot in ``directory``. Never raises: the answer says
    whether it worked, and is recorded in the local ``snapshot.json``."""
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    prefix = config.snapshot.offsite_prefix or ""
    own_client = client is None
    keys: list[str] = []
    temps: list[Path] = []
    try:
        client = client or client_from_config(config)
        for name, info in manifest.get("files", {}).items():
            gz = directory / f".{name}.gz.{os.getpid()}.tmp"
            temps.append(gz)
            sha, size = _gzip(directory / name, gz)
            key = f"{prefix}{name}.gz"
            replace_object(client, key, gz, sha, size, stamp,
                           {"source-sha256": str(info.get("sha256", ""))})
            keys.append(key)
        body = directory / f".manifest.{os.getpid()}.tmp"
        temps.append(body)
        body.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        key = f"{prefix}snapshot.json"
        replace_object(client, key, body, hashlib.sha256(body.read_bytes()).hexdigest(),
                       body.stat().st_size, stamp)
        keys.append(key)
        result = {"ok": True, "at": started.isoformat(timespec="seconds"),
                  "bucket": config.snapshot.offsite_bucket, "keys": keys}
        log.info("offsite: snapshot uploaded to %s", config.snapshot.offsite_bucket)
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.error("offsite: upload failed, the copy in the bucket is kept: %s", error)
        return {"ok": False, "at": started.isoformat(timespec="seconds"),
                "bucket": config.snapshot.offsite_bucket, "error": error, "keys": keys}
    finally:
        for path in temps:
            try:
                path.unlink()
            except OSError:
                pass
        if own_client and client is not None:
            client.close()
