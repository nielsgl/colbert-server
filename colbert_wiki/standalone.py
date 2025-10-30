from functools import lru_cache
import math

from colbert import Searcher
from flask import Flask, request

INDEX_NAME = "wiki17.nbits.local"
INDEX_ROOT = "<path/to/root/of/INDEX_NAME>"
# INDEX_ROOT = "/Users/niels.van.Galen.last/code/hola-dspy/experiments/notebook/indexes"
COLLECTION_PATH = "/path/to/wiki.abstracts.2017/collection.tsv"
# COLLECTION_PATH = (
#     "/Users/niels.van.Galen.last/code/hola-dspy/data/wiki/wiki.abstracts.2017/collection.tsv"
# )

app = Flask(__name__)


searcher = Searcher(
    index=INDEX_NAME,
    checkpoint="colbert-ir/colbertv2.0",
    collection=COLLECTION_PATH,
    index_root=INDEX_ROOT,
)
counter = {"api": 0}


@lru_cache(maxsize=1000000)
def api_search_query(query, k: int = 10):
    print(f"Query={query}")
    if k <= 0:
        k = 10

    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        d = {"text": text, "pid": pid, "rank": rank, "score": score, "prob": prob}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p["score"], p["pid"])))

    return {"query": query, "topk": topk}


@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method != "GET":
        return ("", 405)

    counter["api"] += 1
    print("API request count:", counter["api"])

    try:
        k = int(request.args.get("k", 10))
    except ValueError:
        k = 10

    return api_search_query(request.args.get("query"), k)


if __name__ == "__main__":
    app.run("0.0.0.0", 8893)
