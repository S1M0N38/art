#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx",
#   "python-dotenv",
# ]
# ///

"""
Upload painting images to BunnyCDN Storage Zone.

Reads credentials from environment variables or a .env file in the project root.
Required env vars: BUNNY_STORAGE_ZONE, BUNNY_STORAGE_PASSWORD, BUNNY_STORAGE_ENDPOINT.

Usage:
    uv run scripts/utils/upload_cdn.py                                  # upload all from images/processed/
    uv run scripts/utils/upload_cdn.py images/processed/abc123.jpg      # upload a single file
    uv run scripts/utils/upload_cdn.py --list                           # list remote files
    uv run scripts/utils/upload_cdn.py --skip-existing                  # skip files already on CDN
    uv run scripts/utils/upload_cdn.py --dry-run                        # show what would be uploaded
"""

import argparse
import hashlib
import mimetypes
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROCESSED_DIR = Path("images/processed")
ENV_FILE = Path(".env")


def get_config() -> dict[str, str]:
    """Load BunnyCDN credentials from environment / .env file."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

    import os

    zone = os.environ.get("BUNNY_STORAGE_ZONE", "")
    password = os.environ.get("BUNNY_STORAGE_PASSWORD", "")
    endpoint = os.environ.get("BUNNY_STORAGE_ENDPOINT", "https://storage.bunnycdn.com")

    if not zone or not password:
        print("Error: BUNNY_STORAGE_ZONE and BUNNY_STORAGE_PASSWORD must be set.")
        print("Create a .env file from .env.example or export them in your shell.")
        sys.exit(1)

    return {"zone": zone, "password": password, "endpoint": endpoint}


def sha256_checksum(data: bytes) -> str:
    """Compute uppercase SHA256 hex digest for BunnyCDN checksum verification."""
    return hashlib.sha256(data).hexdigest().upper()


def list_remote_files(config: dict[str, str]) -> list[dict]:
    """List all files in the storage zone root."""
    url = f"{config['endpoint']}/{config['zone']}/"
    headers = {"AccessKey": config["password"]}
    resp = httpx.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def upload_file(
    src: Path,
    config: dict[str, str],
    *,
    dry_run: bool = False,
) -> bool:
    """Upload a single file to BunnyCDN. Returns True on success."""
    data = src.read_bytes()
    checksum = sha256_checksum(data)
    content_type = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
    remote_name = src.name
    url = f"{config['endpoint']}/{config['zone']}/{remote_name}"

    if dry_run:
        size_kb = len(data) / 1024
        print(f"  {src.name}: would upload ({size_kb:.0f} KB)")
        return True

    headers = {
        "AccessKey": config["password"],
        "Content-Type": content_type,
        "Checksum": checksum,
    }

    resp = httpx.put(url, content=data, headers=headers, timeout=120)

    if resp.status_code == 201:
        size_kb = len(data) / 1024
        print(f"  {src.name}: uploaded ({size_kb:.0f} KB)")
        return True
    else:
        print(f"  {src.name}: FAILED (HTTP {resp.status_code})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload painting images to BunnyCDN Storage Zone"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to upload (default: all in images/processed/)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_files",
        help="List files currently on CDN and exit",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist on CDN",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    config = get_config()

    # -- List mode --
    if args.list_files:
        print(f"Files in storage zone '{config['zone']}':")
        remote = list_remote_files(config)
        files = [f for f in remote if not f.get("IsDirectory")]
        if not files:
            print("  (empty)")
        for f in sorted(files, key=lambda x: x["ObjectName"]):
            size_kb = f["Length"] / 1024
            print(f"  {f['ObjectName']:<50} {size_kb:>8.0f} KB")
        print(f"\nTotal: {len(files)} file(s)")
        return

    # -- Upload mode --
    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = (
            sorted(PROCESSED_DIR.glob("*.jpg"))
            + sorted(PROCESSED_DIR.glob("*.jpeg"))
            + sorted(PROCESSED_DIR.glob("*.png"))
        )

    if not files:
        print(f"No images found in {PROCESSED_DIR}/")
        sys.exit(1)

    # Fetch remote listing if skip-existing
    remote_names: set[str] = set()
    if args.skip_existing:
        print("Checking existing files on CDN...")
        remote = list_remote_files(config)
        remote_names = {f["ObjectName"] for f in remote if not f.get("IsDirectory")}
        print(f"  {len(remote_names)} file(s) already on CDN")

    # Filter out existing
    if args.skip_existing:
        to_upload = [f for f in files if f.name not in remote_names]
        skipped = len(files) - len(to_upload)
        if skipped:
            print(f"  Skipping {skipped} already-uploaded file(s)")
        files = to_upload

    if not files:
        print("Nothing to upload — all files already on CDN.")
        return

    label = "Would upload" if args.dry_run else "Uploading"
    print(f"{label} {len(files)} file(s)...")

    success = 0
    failed = 0
    for f in files:
        if not f.exists():
            print(f"  {f}: not found, skipping")
            failed += 1
            continue
        if upload_file(f, config, dry_run=args.dry_run):
            success += 1
        else:
            failed += 1

    print(f"\nDone. {success} uploaded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
