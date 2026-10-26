#!/usr/bin/env python3
"""Pack/unpack persistent Zied Quant state into GitHub-safe chunks."""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

CHUNK_SIZE = 45 * 1024 * 1024
MANIFEST = "manifest.json"


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pack(source_dir, store_dir):
    source = Path(source_dir).resolve()
    store = Path(store_dir).resolve()
    if not source.is_dir():
        raise RuntimeError(f"Source directory not found: {source}")
    if source == store or source in store.parents or store in source.parents:
        raise RuntimeError("Source and store directories must be separate.")
    if store.exists():
        shutil.rmtree(store)
    (store / "parts").mkdir(parents=True)
    entries = []
    part_no = 0
    for path in sorted(p for p in source.rglob("*") if p.is_file()):
        rel = path.relative_to(source).as_posix()
        chunks = []
        size = 0
        h = hashlib.sha256()
        with path.open("rb") as src:
            while True:
                block = src.read(CHUNK_SIZE)
                if not block:
                    break
                name = f"{part_no:08d}.part"
                target = store / "parts" / name
                with target.open("wb") as dst:
                    dst.write(block)
                chunks.append(f"parts/{name}")
                part_no += 1
                size += len(block)
                h.update(block)
        entries.append({"path": rel, "size": size, "sha256": h.hexdigest(), "chunks": chunks})
        print(f"PACKED {rel}: {size} bytes -> {len(chunks)} part(s)")
    manifest = {"format": 1, "chunk_size": CHUNK_SIZE, "files": entries}
    temp = store / (MANIFEST + ".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(store / MANIFEST)
    print(f"PACK_OK files={len(entries)} parts={part_no}")


def unpack(store_dir, destination_dir):
    store = Path(store_dir).resolve()
    dest = Path(destination_dir).resolve()
    manifest_path = store / MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing chunk manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != 1 or not isinstance(manifest.get("files"), list):
        raise RuntimeError("Unsupported or invalid chunk manifest.")
    if dest == store or dest in store.parents or store in dest.parents:
        raise RuntimeError("Store and destination directories must be separate.")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    count = 0
    for entry in manifest["files"]:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"Unsafe path in manifest: {rel}")
        output = dest / rel
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(output.name + ".restore-tmp")
        h = hashlib.sha256()
        size = 0
        with tmp.open("wb") as dst:
            for chunk_rel in entry["chunks"]:
                chunk_path = (store / chunk_rel).resolve()
                if store not in chunk_path.parents or not chunk_path.is_file():
                    raise RuntimeError(f"Missing/unsafe chunk: {chunk_rel}")
                with chunk_path.open("rb") as src:
                    for block in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(block)
                        h.update(block)
                        size += len(block)
        if size != entry["size"] or h.hexdigest() != entry["sha256"]:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Integrity check failed for {rel}")
        os.replace(tmp, output)
        count += 1
        print(f"RESTORED {rel}: {size} bytes (SHA-256 OK)")
    print(f"UNPACK_OK files={count}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_pack = sub.add_parser("pack")
    p_pack.add_argument("source")
    p_pack.add_argument("store")
    p_unpack = sub.add_parser("unpack")
    p_unpack.add_argument("store")
    p_unpack.add_argument("destination")
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.source, args.store)
    else:
        unpack(args.store, args.destination)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"STATE_CHUNKS_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
