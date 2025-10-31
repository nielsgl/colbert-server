from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import colbert_server.__init__ as cli


def test_doctor_success(monkeypatch: mock.MagicMock) -> None:
    monkeypatch.setenv("COLBERT_SERVER_DISABLE_UPDATE_CHECK", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(Path.cwd() / "tmp-cache"))
    with mock.patch("importlib.import_module", return_value=mock.Mock(__version__="1.0")):
        assert cli.handle_doctor(argparse.Namespace()) == 0


def test_serve_from_cache_mock(monkeypatch: mock.MagicMock, tmp_path: Path) -> None:
    monkeypatch.setenv("COLBERT_SERVER_DISABLE_UPDATE_CHECK", "1")

    snapshot_dir = tmp_path / "snapshot"
    collection_dir = snapshot_dir / "collection"
    collection_dir.mkdir(parents=True)
    (collection_dir / "collection.tsv").write_text("0\tmock text")
    indexes_dir = snapshot_dir / "indexes" / "mock-index"
    indexes_dir.mkdir(parents=True)
    (indexes_dir / "0.metadata.json").write_text("{}")

    with (
        mock.patch(
            "colbert_server.data.download_collection_and_indexes",
            return_value=snapshot_dir,
        ),
        mock.patch(
            "colbert_server.data.detect_dataset_paths",
            return_value=(indexes_dir.parent, "mock-index", collection_dir / "collection.tsv"),
        ),
        mock.patch("colbert_server.__init__.create_searcher"),
        mock.patch("colbert_server.__init__.create_app") as mock_app,
    ):
        cli.handle_serve(
            argparse.Namespace(
                from_cache=True,
                download_archives=None,
                extract=False,
                extract_to=None,
                index_root=None,
                index_name=None,
                collection_path=None,
                repo_id=cli.DATASET_REPO_ID,
                revision=None,
                hf_token=None,
                cache_dir=None,
                checkpoint=cli.DEFAULT_CHECKPOINT,
                cache_size=cli.DEFAULT_CACHE_SIZE,
                host="127.0.0.1",
                port=8000,
            )
        )
        mock_app.return_value.run.assert_called_once()
