from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import sys
from unittest import mock

import pytest

import colbert_server.__init__ as cli


def test_doctor_success(monkeypatch: mock.MagicMock) -> None:
    monkeypatch.setenv("COLBERT_SERVER_DISABLE_UPDATE_CHECK", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(Path.cwd() / "tmp-cache"))
    with mock.patch("importlib.import_module", return_value=mock.Mock(__version__="1.0")):
        assert cli.handle_doctor(argparse.Namespace()) == 0


def test_version_warns_when_remote_newer(monkeypatch: mock.MagicMock, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("COLBERT_SERVER_DISABLE_UPDATE_CHECK", raising=False)

    with (
        mock.patch.object(cli, "VERSION", "0.1.0"),
        mock.patch("colbert_server.__init__._fetch_latest_version", return_value="0.2.0"),
        mock.patch("colbert_server.__init__._write_cache"),
    ):
        parser = cli.build_parser()
        buf_err, buf_out = StringIO(), StringIO()
        with mock.patch.object(sys, "stderr", buf_err), mock.patch.object(sys, "stdout", buf_out):
            with pytest.raises(SystemExit):
                parser.parse_args(["--version"])
        assert "A newer version" in buf_err.getvalue()


def test_version_uses_cache(monkeypatch: mock.MagicMock, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("COLBERT_SERVER_DISABLE_UPDATE_CHECK", raising=False)
    cache_file = cli._cache_path()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"latest": "0.9.0", "checked_at": 2000000000}')

    with (
        mock.patch.object(cli, "VERSION", "0.8.0"),
        mock.patch("colbert_server.__init__._fetch_latest_version", return_value=None),
        mock.patch("colbert_server.__init__._write_cache"),
    ):
        parser = cli.build_parser()
        buf_err, buf_out = StringIO(), StringIO()
        with mock.patch.object(sys, "stderr", buf_err), mock.patch.object(sys, "stdout", buf_out):
            with pytest.raises(SystemExit):
                parser.parse_args(["--version"])
        assert "0.9.0" in buf_err.getvalue()


def test_version_skips_when_no_newer(monkeypatch: mock.MagicMock, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("COLBERT_SERVER_DISABLE_UPDATE_CHECK", raising=False)
    with (
        mock.patch.object(cli, "VERSION", "0.3.0"),
        mock.patch("colbert_server.__init__._fetch_latest_version", return_value="0.3.0"),
        mock.patch("colbert_server.__init__._write_cache"),
    ):
        parser = cli.build_parser()
        buf_err, buf_out = StringIO(), StringIO()
        with mock.patch.object(sys, "stderr", buf_err), mock.patch.object(sys, "stdout", buf_out):
            with pytest.raises(SystemExit):
                parser.parse_args(["--version"])
        assert "A newer version" not in buf_err.getvalue()


def test_version_disabled_by_env(monkeypatch: mock.MagicMock, tmp_path: Path) -> None:
    monkeypatch.setenv("COLBERT_SERVER_DISABLE_UPDATE_CHECK", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with (
        mock.patch.object(cli, "VERSION", "0.5.0"),
        mock.patch("colbert_server.__init__._fetch_latest_version") as fetch_mock,
    ):
        parser = cli.build_parser()
        with (
            mock.patch.object(sys, "stderr", StringIO()),
            mock.patch.object(sys, "stdout", StringIO()),
        ):
            with pytest.raises(SystemExit):
                parser.parse_args(["--version"])
        fetch_mock.assert_not_called()


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
            "colbert_server.__init__.download_collection_and_indexes",
            return_value=snapshot_dir,
        ),
        mock.patch(
            "colbert_server.__init__.detect_dataset_paths",
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
