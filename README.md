# colbert-wiki

CLI tooling to download the ColBERT Wikipedia dataset and run a lightweight
Flask server that exposes `/api/search`.

## Installation

```bash
uv tool install .
```

## Usage

### Serve local assets

```bash
colbert-wiki serve \
  --index-root /path/to/indexes \
  --index-name wiki2017-index \
  --collection-path /path/to/collection.tsv
```

### Serve using the Hugging Face cache

```bash
colbert-wiki serve --from-cache
```

The command downloads the `collection/` and `indexes/` folders from the
`nielsgl/colbert-wiki2017` dataset into your Hugging Face cache, infers the
paths, and boots the server.

### Download archives for offline use

```bash
colbert-wiki download-archives ./downloads --extract
```

This downloads the ZIP bundles under `archives/` into `./downloads` and extracts
them in-place. To extract into a different directory, add `--extract-to`.

You can also combine download + extraction with serving in one step:

```bash
colbert-wiki serve --download-archives ./downloads --extract
```

After extraction, the server will auto-detect the dataset paths and start.
