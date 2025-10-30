from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import snapshot_download

DATASET_REPO_ID = "nielsgl/colbert-wiki2017"
ARCHIVES_DIRNAME = "archives"
COLLECTION_DIRNAME = "collection"
INDEXES_DIRNAME = "indexes"
SUPPORTED_ARCHIVE_SUFFIXES = {
    suffix for _, suffixes, _ in shutil.get_unpack_formats() for suffix in suffixes
}
SUPPORTED_ARCHIVE_SUFFIXES_LOWER = tuple(
    suffix.lower() for suffix in SUPPORTED_ARCHIVE_SUFFIXES
)


class DatasetLayoutError(RuntimeError):
    """Raised when the downloaded dataset structure is not as expected."""


def download_archives(
    destination: Path,
    *,
    repo_id: str = DATASET_REPO_ID,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """
    Download the compressed archives from the Hugging Face dataset into ``destination``.

    Returns the path to the local snapshot that contains the ``archives`` folder.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        allow_patterns=[f"{ARCHIVES_DIRNAME}/*"],
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(snapshot_path)


def download_collection_and_indexes(
    *,
    repo_id: str = DATASET_REPO_ID,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """
    Download the ``collection`` and ``indexes`` folders into the Hugging Face cache.

    Returns the path to the snapshot containing the folders.
    """
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=[f"{COLLECTION_DIRNAME}/*", f"{INDEXES_DIRNAME}/*"],
        ignore_patterns=[f"{ARCHIVES_DIRNAME}/*"],
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(snapshot_path)


def extract_archives(snapshot_path: Path, extract_to: Path) -> Path:
    """
    Extract every archive found in ``snapshot_path/archives`` into ``extract_to``.

    Returns the extraction directory.
    """
    snapshot_path = Path(snapshot_path)
    extract_to = Path(extract_to)
    archives_root = snapshot_path / ARCHIVES_DIRNAME
    if not archives_root.exists():
        raise DatasetLayoutError(
            f"Expected an '{ARCHIVES_DIRNAME}' folder below {snapshot_path}, "
            "but none was found. Did you download the archives?"
        )

    extract_to.mkdir(parents=True, exist_ok=True)
    extracted_any = False
    for archive in sorted(archives_root.glob("*")):
        if not archive.is_file():
            continue
        if SUPPORTED_ARCHIVE_SUFFIXES_LOWER and not archive.name.lower().endswith(
            SUPPORTED_ARCHIVE_SUFFIXES_LOWER
        ):
            # Skip files that do not look like archives.
            continue
        try:
            shutil.unpack_archive(str(archive), str(extract_to))
            extracted_any = True
        except (shutil.ReadError, ValueError) as err:
            raise DatasetLayoutError(
                f"Failed to extract archive {archive.name}: {err}"
            ) from err

    if not extracted_any:
        raise DatasetLayoutError(
            f"No archives were extracted from {archives_root}. Is the folder empty?"
        )

    return extract_to


def locate_dataset_root(root: Path) -> Path:
    """
    Locate the directory that contains the dataset's ``indexes`` folder.

    This handles cases where the archives were extracted into a nested directory.
    """
    root = Path(root)
    queue = [root]
    visited = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if (current / INDEXES_DIRNAME).exists():
            return current

        for child in sorted(current.iterdir()):
            if child.is_dir():
                queue.append(child)

    raise DatasetLayoutError(
        f"Could not find an '{INDEXES_DIRNAME}' folder beneath {root}. "
        "Please verify your download or supply the paths explicitly."
    )


def detect_dataset_paths(
    base_path: Path, *, preferred_index_name: Optional[str] = None
) -> Tuple[Path, str, Optional[Path]]:
    """
    Inspect ``base_path`` to determine the index root, index name, and collection path.

    Returns a tuple ``(index_root, index_name, collection_path)`` where ``collection_path``
    may be ``None`` if it could not be inferred automatically.
    """
    dataset_root = locate_dataset_root(base_path)
    indexes_root = dataset_root / INDEXES_DIRNAME
    if not indexes_root.exists():
        raise DatasetLayoutError(
            f"The resolved dataset root {dataset_root} does not contain "
            f"an '{INDEXES_DIRNAME}' directory."
        )

    candidate_indexes = [p for p in indexes_root.iterdir() if p.is_dir()]
    if preferred_index_name:
        target = indexes_root / preferred_index_name
        if not target.exists():
            raise DatasetLayoutError(
                f"Index '{preferred_index_name}' was not found under {indexes_root}."
            )
        index_name = preferred_index_name
    else:
        if not candidate_indexes:
            raise DatasetLayoutError(
                f"No index directories were found under {indexes_root}."
            )
        if len(candidate_indexes) > 1:
            options = ", ".join(sorted(p.name for p in candidate_indexes))
            raise DatasetLayoutError(
                "Multiple index directories detected. Please supply --index-name. "
                f"Available options: {options}"
            )
        index_name = candidate_indexes[0].name

    collection_path = infer_collection_path(dataset_root)
    return indexes_root, index_name, collection_path


def infer_collection_path(dataset_root: Path) -> Optional[Path]:
    """Attempt to infer the collection file path from the dataset root."""
    dataset_root = Path(dataset_root)

    collection_dir = dataset_root / COLLECTION_DIRNAME
    if collection_dir.is_file():
        return collection_dir

    if collection_dir.is_dir():
        tsv_files = sorted(collection_dir.glob("*.tsv"))
        if len(tsv_files) == 1:
            return tsv_files[0]
        if len(tsv_files) > 1:
            raise DatasetLayoutError(
                f"Multiple TSV files found in {collection_dir}; please specify "
                "the collection path explicitly."
            )
        # Fall back to the directory itself if formats differ
        return collection_dir

    # Fallback: look for a collection file at the root
    tsv_candidates = sorted(dataset_root.glob("collection*.tsv"))
    if len(tsv_candidates) == 1:
        return tsv_candidates[0]

    return None
