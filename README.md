# ColBERT Wikipedia Server

[![uv tool](https://img.shields.io/badge/uv-tool-3b82f6?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![python 3.13+](https://img.shields.io/badge/python-3.13+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![license MIT](https://img.shields.io/badge/license-MIT-7c3aed.svg)](LICENSE)
[![dataset](https://img.shields.io/badge/dataset-huggingface-ff9a00?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/nielsgl/colbert-wiki2017)
[![API Flask](https://img.shields.io/badge/api-flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

CLI tooling to fetch the ColBERT Wikipedia 2017 dataset and run a lightweight Flask API on top of the ColBERT v2 searcher.

```text
$> I wrote this because the ColBERT server is down and I couldn't try one of the tutorial from DSPy.
$> I only tested this on my macbook, please open an issue if you have problems or feature requests.
```

## Features

- One-command install and execution via `uv tool`.
- Automatically downloads either ready-to-serve indexes/collection or the original archives.
- Optional archive extraction flow for offline usage.
- Caches ColBERT queries for fast, repeated lookups.
- Exposes a simple `/api/search` endpoint for programmatic access.

## Installation

```bash
uv tool install colbert-server
```

This registers a `colbert-server` executable in your `uv` toolchain.

Or if you just want to run it:

```bash
uvx run colbert-server --help
```

## Running the server

### Use data from the Hugging Face cache (recommended quick start)

```bash
colbert-server serve --from-cache
```

This downloads only the `collection/` and `indexes/` folders from
[`nielsgl/colbert-wiki2017`](https://huggingface.co/datasets/nielsgl/colbert-wiki2017),
resolves the on-disk paths from the Hugging Face cache, and starts the server.

### Provide existing local assets

```bash
colbert-server serve \
  --index-root /path/to/indexes \
  --index-name wiki17.nbits.local \
  --collection-path /path/to/collection/wiki.abstracts.2017/collection.tsv
```

Use this mode when you already have ColBERT indexes and a collection TSV locally.

### Download archives first, then serve

```bash
colbert-server serve \
  --download-archives /tmp/wiki-assets \
  --extract \
  --port 8894
```

This fetches the archive files into `/tmp/wiki-assets/archives`, extracts them into
`/tmp/wiki-assets`, auto-detects the resulting layout (e.g. `wiki17.nbits.local`),
and starts the Flask server on port `8894`.

## API usage

Once running, the server listens on the host/port provided (defaults to `0.0.0.0:8893`)
and serves ColBERT search results via:

```
GET /api/search?query=<text>&k=<top-k>
```

Example request:

```
http://127.0.0.1:8893/api/search?query=halloween+movie&k=3
```

The JSON response includes the ranked passages, their scores, and normalized probabilities.

## Managing dataset archives only

If you just want the raw archive bundles in a local directory:

```bash
colbert-server download-archives ./downloads --extract
```

Add `--extract-to /desired/path` to unpack into a different directory. You can later reuse
the extracted paths with the `serve` command’s `--index-root` and `--collection-path` flags.

## Alternative / Manual Method

In case you don't want to use the script / `uv` tool you can set it up as follows:

1. Add the dependencies to your project: `uv add colbert-ai flask faiss-cpu torch`
2. Download the files (both the index and the collection) from the `archives` directory from the HuggingFace dataset and unzip them.
3. Copy the `standalone.py` script from this repository and edit the `INDEX_ROOT` and `COLLECTION_PATH` variables.
4. Run the server with `uv run standalone.py` and <tada.wav>

## Development tips

- Requires Python 3.13+ (or adjust the `pyproject.toml` requirement to match your interpreter).
- Run `colbert-server --help` or `colbert-server serve --help` to inspect available options.
- The dataset helpers live under `colbert_server/data.py`; server configuration sits in `colbert_server/server.py`.
- GitHub Actions runs lint/tests on every push; see `.github/workflows/ci.yml` for details.
- Publishing uses the `.github/workflows/publish.yml` workflow. Before releasing, add `PYPI_API_TOKEN` (and optionally `TEST_PYPI_API_TOKEN`) to the repository secrets, bump the version in `pyproject.toml`, create a `vX.Y.Z` tag, and push it to trigger the publish job.

Happy searching! 🧠📚
