from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from packaging.version import InvalidVersion, Version

from .data import (
    DATASET_REPO_ID,
    DatasetLayoutError,
    detect_dataset_paths,
    download_archives,
    download_collection_and_indexes,
    extract_archives,
)
from .server import DEFAULT_CACHE_SIZE, DEFAULT_CHECKPOINT, create_app, create_searcher

PACKAGE_NAME = "colbert-server"


def _resolve_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "0.0.0"


VERSION = _resolve_version()
CACHE_TTL_SECONDS = 24 * 60 * 60


def _cache_path() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "colbert-server" / "update.json"


def _read_cached_latest() -> tuple[str | None, float | None]:
    try:
        data = json.loads(_cache_path().read_text())
        latest = data.get("latest")
        checked_at = float(data.get("checked_at", 0))
        return latest, checked_at
    except (OSError, ValueError, TypeError):
        return None, None


def _write_cache(latest: str) -> None:
    cache_file = _cache_path()
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"latest": latest, "checked_at": time.time()}))
    except OSError:
        pass


def _fetch_latest_version(timeout: float = 2.0) -> str | None:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{PACKAGE_NAME}/json",
        headers={"User-Agent": f"{PACKAGE_NAME}/{VERSION}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        latest = payload.get("info", {}).get("version")
        if isinstance(latest, str) and latest:
            _write_cache(latest)
            return latest
    except (urllib.error.URLError, ValueError, OSError):
        return None
    return None


def maybe_warn_on_update() -> None:
    if os.environ.get("COLBERT_SERVER_DISABLE_UPDATE_CHECK") == "1":
        return

    cached_version, checked_at = _read_cached_latest()
    now = time.time()
    if cached_version and checked_at and now - checked_at < CACHE_TTL_SECONDS:
        latest_version = cached_version
    else:
        latest_version = _fetch_latest_version()
        if latest_version is None:
            latest_version = cached_version

    if not latest_version:
        return

    try:
        current_version = Version(VERSION)
        remote_version = Version(latest_version)
    except InvalidVersion:
        return

    if remote_version > current_version:
        print(
            (
                f"A newer version of {PACKAGE_NAME} is available "
                f"({remote_version} > {current_version}). "
                "Update with `uv tool upgrade colbert-server`."
            ),
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colbert-server",
        description="Run the ColBERT Wikipedia search server or manage its dataset assets.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="Show the installed version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the ColBERT search API server.")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    serve_parser.add_argument("--port", type=int, default=8893, help="Port to listen on.")
    serve_parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"ColBERT checkpoint to load (default: {DEFAULT_CHECKPOINT}).",
    )
    serve_parser.add_argument(
        "--cache-size",
        type=int,
        default=DEFAULT_CACHE_SIZE,
        help=f"Maximum number of cached search queries (default: {DEFAULT_CACHE_SIZE}).",
    )
    source_group = serve_parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--from-cache",
        action="store_true",
        help="Download the collection and indexes into the Hugging Face cache.",
    )
    source_group.add_argument(
        "--download-archives",
        type=Path,
        metavar="DIR",
        help="Download dataset archives into DIR before serving.",
    )
    serve_parser.add_argument(
        "--extract-to",
        type=Path,
        metavar="DIR",
        help="Extraction directory used with --download-archives. Defaults to the download DIR.",
    )
    serve_parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded archives when using --download-archives.",
    )
    serve_parser.add_argument(
        "--index-root",
        type=Path,
        help="Path to the local ColBERT index root folder.",
    )
    serve_parser.add_argument(
        "--index-name",
        help="Name of the ColBERT index to load (folder name within the index root).",
    )
    serve_parser.add_argument(
        "--collection-path",
        type=Path,
        help="Path to the document collection file (optional).",
    )
    serve_parser.add_argument(
        "--repo-id",
        default=DATASET_REPO_ID,
        help=f"Hugging Face dataset repository (default: {DATASET_REPO_ID}).",
    )
    serve_parser.add_argument(
        "--revision",
        help="Optional dataset revision (branch, tag, or commit hash).",
    )
    serve_parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face access token for private datasets.",
    )
    serve_parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory for --from-cache downloads.",
    )
    serve_parser.set_defaults(func=handle_serve)

    archives_parser = subparsers.add_parser(
        "download-archives",
        help="Download (and optionally extract) the ColBERT dataset archives.",
    )
    archives_parser.add_argument(
        "destination",
        type=Path,
        help="Directory that will receive the downloaded archives.",
    )
    archives_parser.add_argument(
        "--extract-to",
        type=Path,
        metavar="DIR",
        help="Extract the archives into DIR after downloading.",
    )
    archives_parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archives in-place (or into --extract-to if provided).",
    )
    archives_parser.add_argument(
        "--repo-id",
        default=DATASET_REPO_ID,
        help=f"Hugging Face dataset repository (default: {DATASET_REPO_ID}).",
    )
    archives_parser.add_argument(
        "--revision",
        help="Optional dataset revision (branch, tag, or commit hash).",
    )
    archives_parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face access token for private datasets.",
    )
    archives_parser.set_defaults(func=handle_download_archives)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect environment and dataset prerequisites without downloading large assets.",
    )
    doctor_parser.set_defaults(func=handle_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    maybe_warn_on_update()

    try:
        return args.func(args)
    except DatasetLayoutError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


def handle_serve(args: argparse.Namespace) -> int:
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    if args.from_cache:
        snapshot_path = download_collection_and_indexes(
            repo_id=args.repo_id,
            revision=args.revision,
            token=hf_token,
            cache_dir=args.cache_dir,
        )
        index_root, index_name, inferred_collection = detect_dataset_paths(
            snapshot_path, preferred_index_name=args.index_name
        )
        collection_path = (
            Path(args.collection_path) if args.collection_path else inferred_collection
        )
        print(f"Downloaded dataset snapshot to {snapshot_path}")
    elif args.download_archives:
        snapshot_path = download_archives(
            args.download_archives,
            repo_id=args.repo_id,
            revision=args.revision,
            token=hf_token,
        )
        print(f"Archives downloaded to {snapshot_path / 'archives'}")

        if not args.extract and args.extract_to is None:
            print(
                "Archives downloaded. Extract them manually and relaunch the server "
                "with --index-root/--index-name.",
            )
            return 0

        extraction_dir = args.extract_to or args.download_archives
        extracted_root = extract_archives(snapshot_path, extraction_dir)
        print(f"Archives extracted to {extracted_root}")
        index_root, index_name, inferred_collection = detect_dataset_paths(
            extracted_root, preferred_index_name=args.index_name
        )
        collection_path = (
            Path(args.collection_path) if args.collection_path else inferred_collection
        )
    else:
        if not args.index_root or not args.index_name:
            raise DatasetLayoutError(
                "--index-root and --index-name are required when not downloading from Hugging Face."
            )
        index_root = Path(args.index_root)
        index_name = args.index_name
        collection_path = Path(args.collection_path) if args.collection_path else None

    if collection_path is None:
        print(
            "Warning: collection path could not be inferred. "
            "ColBERT will run without document text.",
            file=sys.stderr,
        )

    searcher = create_searcher(
        index_root=str(index_root),
        index_name=index_name,
        collection_path=str(collection_path) if collection_path else None,
        checkpoint=args.checkpoint,
    )
    app = create_app(searcher, cache_size=args.cache_size)

    print(f"Serving index '{index_name}' from root {index_root}")
    if collection_path:
        print(f"Using collection file {collection_path}")

    app.run(host=args.host, port=args.port)
    return 0


def handle_download_archives(args: argparse.Namespace) -> int:
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    snapshot_path = download_archives(
        args.destination,
        repo_id=args.repo_id,
        revision=args.revision,
        token=hf_token,
    )
    print(f"Archives downloaded to {snapshot_path / 'archives'}")

    if args.extract or args.extract_to:
        extraction_dir = args.extract_to or args.destination
        extracted_root = extract_archives(snapshot_path, extraction_dir)
        print(f"Archives extracted to {extracted_root}")

    return 0


def _check_package(name: str, friendly: str | None = None) -> tuple[bool, str]:
    label = friendly or name
    try:
        importlib.import_module(name)
        return True, f"{label}: OK"
    except ModuleNotFoundError as exc:
        return False, f"{label}: missing ({exc})"
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"{label}: error ({exc})"


def _check_torch_cpu() -> tuple[bool, str]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        return (
            False,
            f"torch import failed ({exc}). Install via `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return True, f"torch {torch.__version__} detected ({device})"


def _check_faiss() -> tuple[bool, str]:
    libs = ["faiss", "faiss_cpu"]
    messages = []
    for lib in libs:
        ok, msg = _check_package(lib)
        if ok:
            return True, msg
        messages.append(msg)
    return False, "; ".join(messages)


def _describe_cache() -> str:
    cache_dir = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface" / "hub"
    )
    if cache_dir.exists():
        try:
            size = sum(p.stat().st_size for p in cache_dir.rglob("*"))
            size_mb = size / (1024 * 1024)
            return f"Hugging Face cache: {cache_dir} (~{size_mb:.1f} MB)"
        except OSError:
            return f"Hugging Face cache: {cache_dir} (size unknown)"
    return f"Hugging Face cache: {cache_dir} (directory missing, will be created)"


def handle_doctor(args: argparse.Namespace) -> int:  # noqa: ARG001
    print(f"colbert-server {VERSION}")
    print(_describe_cache())

    checks = [
        _check_torch_cpu(),
        _check_faiss(),
        _check_package("huggingface_hub", "huggingface-hub"),
        _check_package("flask", "flask"),
    ]

    errors = False
    for ok, message in checks:
        status = "OK" if ok else "WARN"
        print(f"[{status}] {message}")
        if not ok:
            errors = True

    if errors:
        print("Some checks failed. Review warnings above before running `serve`.")
        return 1

    print(
        "Environment looks ready. Run `colbert-server serve --from-cache` "
        "to download indices (~13 GB)."
    )
    return 0
