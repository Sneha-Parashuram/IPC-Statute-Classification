"""Resumable download helper for the ILSI dataset from Zenodo record 6053791.

Features:
- Byte-range resumption (HTTP Range header) for interrupted streams.
- Preserves partially downloaded files between retries.
- Automatic retries with backoff on network drop or IncompleteRead.
- MD5 checksum verification against Zenodo official checksum.
- Downloads only the 4 required splits (~511 MB), skipping graph embeddings.
"""
import argparse
import hashlib
import time
from pathlib import Path

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, RequestException
from urllib3.exceptions import IncompleteRead, ProtocolError

RECORD = "https://zenodo.org/api/records/6053791"
WANTED = ["train.jsonl", "dev.jsonl", "test.jsonl", "secs.jsonl"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, dest, total_size, expected_md5, max_retries=20, timeout=120):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == total_size:
        got = md5(dest)
        if got == expected_md5:
            print(f"{dest.name}: already complete, MD5 checksum OK", flush=True)
            return
        else:
            print(f"{dest.name}: local size matches but MD5 mismatch, resetting...", flush=True)
            dest.unlink()

    for attempt in range(1, max_retries + 1):
        downloaded = dest.stat().st_size if dest.exists() else 0
        if downloaded > total_size:
            print(f"{dest.name}: local size ({downloaded}) > total ({total_size}), resetting...", flush=True)
            dest.unlink()
            downloaded = 0

        headers = {}
        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"
            print(f"{dest.name}: resuming from {downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB (attempt {attempt}/{max_retries}) ...", flush=True)
        else:
            print(f"{dest.name}: downloading {total_size / 1e6:.1f} MB (attempt {attempt}/{max_retries}) ...", flush=True)

        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if downloaded > 0 and r.status_code == 416:
                    print(f"{dest.name}: range 416 (not satisfiable), resetting...", flush=True)
                    dest.unlink()
                    continue
                if downloaded > 0 and r.status_code not in (200, 206):
                    r.raise_for_status()
                elif downloaded == 0 and r.status_code != 200:
                    r.raise_for_status()

                mode = "ab" if (downloaded > 0 and r.status_code == 206) else "wb"
                if mode == "wb" and downloaded > 0:
                    print(f"{dest.name}: server returned 200 instead of 206, resetting from 0...", flush=True)

                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)

            if dest.stat().st_size == total_size:
                got = md5(dest)
                if got == expected_md5:
                    print(f"{dest.name}: download complete, MD5 checksum OK", flush=True)
                    return
                else:
                    print(f"{dest.name}: MD5 mismatch (got {got}, expected {expected_md5})", flush=True)
                    dest.unlink()
                    if attempt == max_retries:
                        raise SystemExit(f"{dest.name}: MD5 verification failed after {max_retries} attempts")
            else:
                curr = dest.stat().st_size if dest.exists() else 0
                print(f"{dest.name}: stream ended at {curr / 1e6:.1f} / {total_size / 1e6:.1f} MB. Retrying in 3s...", flush=True)
                time.sleep(3)

        except (RequestException, ProtocolError, ConnectionError, IncompleteRead, ChunkedEncodingError) as e:
            curr = dest.stat().st_size if dest.exists() else 0
            print(f"{dest.name}: connection interrupted at {curr / 1e6:.1f} MB ({type(e).__name__}). Retrying in 5s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(5)

    raise SystemExit(f"{dest.name}: failed to complete download after {max_retries} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Querying Zenodo record {RECORD} ...", flush=True)
    resp = requests.get(RECORD, timeout=60).json()
    files = {f["key"]: f for f in resp["files"]}

    for name in WANTED:
        meta = files[name]
        expected = meta["checksum"].split(":", 1)[1]
        dest = out / name
        download_file(meta["links"]["self"], dest, meta["size"], expected)

    print("\nAll 4 ILSI dataset files successfully downloaded and verified!", flush=True)


if __name__ == "__main__":
    main()
