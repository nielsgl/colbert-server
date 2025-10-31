from __future__ import annotations

from functools import lru_cache
import math
from typing import TYPE_CHECKING

from flask import Flask, request

if TYPE_CHECKING:  # pragma: no cover - imported lazily at runtime
    from colbert import Searcher

DEFAULT_CHECKPOINT = "colbert-ir/colbertv2.0"
DEFAULT_CACHE_SIZE = 1_000_000


def create_searcher(
    index_root: str,
    index_name: str,
    collection_path: str | None,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> Searcher:
    """Instantiate a ColBERT Searcher with the given configuration."""
    from colbert import Searcher  # Import lazily to avoid eager torch/faiss loading

    return Searcher(
        index=index_name,
        checkpoint=checkpoint,
        collection=collection_path,
        index_root=index_root,
    )


def create_app(searcher: Searcher, cache_size: int = DEFAULT_CACHE_SIZE) -> Flask:
    """Build a Flask app that serves ColBERT search results."""
    app = Flask(__name__)
    counter = {"api": 0}

    @lru_cache(maxsize=cache_size)
    def api_search_query(query: str, k: int | None):
        print(f"Query={query}")
        if query is None:
            return {"query": "", "topk": []}

        try:
            requested_k = 10 if k is None else max(1, min(int(k), 100))
        except (TypeError, ValueError):
            requested_k = 10
        pids, ranks, scores = searcher.search(query, k=100)
        pids, ranks, scores = pids[:requested_k], ranks[:requested_k], scores[:requested_k]

        exp_scores = [math.exp(score) for score in scores]
        total = sum(exp_scores)
        probs = [score / total for score in exp_scores] if total else [0.0 for _ in scores]

        topk = []
        for pid, rank, score, prob in zip(pids, ranks, scores, probs):
            text = searcher.collection[pid] if searcher.collection is not None else None
            topk.append({"text": text, "pid": pid, "rank": rank, "score": score, "prob": prob})

        topk.sort(key=lambda item: (-item["score"], item["pid"]))
        return {"query": query, "topk": topk}

    @app.route("/api/search", methods=["GET"])
    def api_search():
        if request.method == "GET":
            counter["api"] += 1
            print("API request count:", counter["api"])
            return api_search_query(request.args.get("query"), request.args.get("k"))
        return ("", 405)

    return app
