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
Upload optimized painting images to BunnyCDN Storage Zone.

Uploads pre-generated image variants from images/optimized/ to the CDN,
preserving the directory structure (originals/, thumbs/, placeholders/).

Reads credentials from environment variables or a .env file in the project root.
Required env vars: BUNNY_STORAGE_ZONE, BUNNY_STORAGE_PASSWORD, BUNNY_STORAGE_ENDPOINT.

Usage:
    uv run scripts/utils/upload_cdn.py                          # upload all variants
    uv run scripts/utils/upload_cdn.py --variant thumbs         # upload only thumbnails
    uv run scripts/utils/upload_cdn.py --list                   # list all remote files
    uv run scripts/utils/upload_cdn.py --skip-existing          # skip already uploaded
    uv run scripts/utils/upload_cdn.py --dry-run                # preview what would upload
"""

import argparse
import hashlib
import mimetypes
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

OPTIMIZED_DIR = Path("images/optimized")
VARIANT_DIRS = ["originals", "thumbs", "placeholders"]
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


def list_remote_files(config: dict[str, str], subdir: str = "") -> list[dict]:
    """List files in the storage zone, optionally within a subdirectory."""
    path = f"{config['zone']}/{subdir}/" if subdir else f"{config['zone']}/"
    url = f"{config['endpoint']}/{path}"
    headers = {"AccessKey": config["password"]}
    resp = httpx.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def upload_file(
    src: Path,
    config: dict[str, str],
    *,
    remote_dir: str = "",
    dry_run: bool = False,
) -> bool:
    """Upload a single file to BunnyCDN. Returns True on success."""
    data = src.read_bytes()
    checksum = sha256_checksum(data)
    content_type = mimetypes.guess_type(src.name)[0] or "application/octet-stream"
    remote_path = f"{remote_dir}/{src.name}" if remote_dir else src.name
    url = f"{config['endpoint']}/{config['zone']}/{remote_path}"
    label = f"[{remote_dir}] {src.name}" if remote_dir else src.name

    if dry_run:
        size_kb = len(data) / 1024
        print(f"  {label}: would upload ({size_kb:.0f} KB)")
        return True

    headers = {
        "AccessKey": config["password"],
        "Content-Type": content_type,
        "Checksum": checksum,
    }

    resp = httpx.put(url, content=data, headers=headers, timeout=120)

    if resp.status_code == 201:
        size_kb = len(data) / 1024
        print(f"  {label}: uploaded ({size_kb:.0f} KB)")
        return True
    else:
        print(f"  {label}: FAILED (HTTP {resp.status_code})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload optimized painting images to BunnyCDN Storage Zone"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to upload (default: all in images/optimized/)",
    )
    parser.add_argument(
        "--variant",
        choices=["all", "originals", "thumbs", "placeholders"],
        default="all",
        help="Which variant(s) to upload (default: all)",
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

    # Determine which variant dirs to operate on
    variants = VARIANT_DIRS if args.variant == "all" else [args.variant]

    # -- List mode --
    if args.list_files:
        print(f"Files in storage zone '{config['zone']}':")
        total = 0
        for variant in VARIANT_DIRS:
            remote = list_remote_files(config, subdir=variant)
            files = [f for f in remote if not f.get("IsDirectory")]
            for f in sorted(files, key=lambda x: x["ObjectName"]):
                size_kb = f["Length"] / 1024
                display = f"{variant}/{f['ObjectName']}"
                print(f"  {display:<50} {size_kb:>8.0f} KB")
            total += len(files)
        if not total:
            print("  (empty)")
        print(f"\nTotal: {total} file(s)")
        return

    # -- Upload mode: collect files --
    # Each entry is (local_path, remote_dir)
    upload_list: list[tuple[Path, str]] = []

    if args.files:
        # Positional files: infer remote_dir from parent dir name
        for f in args.files:
            p = Path(f)
            parent = p.parent.name
            remote_dir = parent if parent in VARIANT_DIRS else ""
            upload_list.append((p, remote_dir))
    else:
        # Auto-collect from optimized variant dirs
        for variant in variants:
            variant_path = OPTIMIZED_DIR / variant
            if not variant_path.is_dir():
                continue
            found = sorted(variant_path.glob("*.jpg")) + sorted(
                variant_path.glob("*.webp")
            )
            upload_list.extend((f, variant) for f in found)

    if not upload_list:
        print(f"No images found in {OPTIMIZED_DIR}/")
        sys.exit(1)

    # Fetch remote listing per variant if skip-existing
    remote_by_dir: dict[str, set[str]] = {}
    if args.skip_existing:
        print("Checking existing files on CDN...")
        dirs_to_check = {remote_dir for _, remote_dir in upload_list}
        for d in sorted(dirs_to_check):
            remote = list_remote_files(config, subdir=d)
            names = {f["ObjectName"] for f in remote if not f.get("IsDirectory")}
            remote_by_dir[d] = names
            print(f"  [{d or 'root'}] {len(names)} file(s) already on CDN")

    # Filter out existing
    if args.skip_existing:
        before = len(upload_list)
        upload_list = [
            (p, d)
            for p, d in upload_list
            if p.name not in remote_by_dir.get(d, set())
        ]
        skipped = before - len(upload_list)
        if skipped:
            print(f"  Skipping {skipped} already-uploaded file(s)")

    if not upload_list:
        print("Nothing to upload — all files already on CDN.")
        return

    # Count variants represented
    variant_count = len({d for _, d in upload_list if d})
    label = "Would upload" if args.dry_run else "Uploading"
    if variant_count > 1:
        print(f"{label} {len(upload_list)} file(s) across {variant_count} variant(s)...")
    else:
        print(f"{label} {len(upload_list)} file(s)...")

    success = 0
    failed = 0
    for f, remote_dir in upload_list:
        if not f.exists():
            tag = f"[{remote_dir}] " if remote_dir else ""
            print(f"  {tag}{f}: not found, skipping")
            failed += 1
            continue
        if upload_file(f, config, remote_dir=remote_dir, dry_run=args.dry_run):
            success += 1
        else:
            failed += 1

    print(f"\nDone. {success} uploaded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
