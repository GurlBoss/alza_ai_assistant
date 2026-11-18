"""Nearest-neighbour search over chunk embeddings.

VectorStore is the interface; VertexVectorSearch uses a Vertex AI Vector Search
streaming index. Another backend (e.g. Firestore vector search) only needs to
implement the same three methods.
"""

import math
from typing import Protocol

import google.auth
from google.auth.transport.requests import AuthorizedSession

TIMEOUT_SECONDS = 10


class VectorStore(Protocol):
    def upsert(self, items: list[tuple[str, list[float]]]) -> None:
        """Adds or replaces (ID, vector) pairs."""

    def remove(self, ids: list[str]) -> None: ...

    def query(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        """(ID, similarity) pairs, most similar first; higher is more similar."""


class VertexVectorSearch:
    """Index with DOT_PRODUCT_DISTANCE on L2-normalized vectors: the returned
    distance is the cosine similarity."""

    def __init__(
        self,
        project: str,
        region: str,
        index_id: str,
        endpoint_id: str,
        deployed_index_id: str,
    ) -> None:
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._session = AuthorizedSession(credentials)
        api = f"https://{region}-aiplatform.googleapis.com/v1"
        location = f"projects/{project}/locations/{region}"
        self._index_url = f"{api}/{location}/indexes/{index_id}"
        self._endpoint_name = f"{location}/indexEndpoints/{endpoint_id}"
        self._endpoint_url = f"{api}/{self._endpoint_name}"
        self._deployed_index_id = deployed_index_id
        self._query_url: str | None = None

    def upsert(self, items: list[tuple[str, list[float]]]) -> None:
        body = {
            "datapoints": [
                {"datapointId": item_id, "featureVector": vector}
                for item_id, vector in items
            ]
        }
        self._post(f"{self._index_url}:upsertDatapoints", body)

    def remove(self, ids: list[str]) -> None:
        if ids:
            self._post(f"{self._index_url}:removeDatapoints", {"datapointIds": ids})

    def query(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        body = {
            "deployedIndexId": self._deployed_index_id,
            "queries": [
                {"datapoint": {"featureVector": vector}, "neighborCount": top_k}
            ],
        }
        response = self._post(self._find_neighbors_url(), body)
        neighbors = (response.get("nearestNeighbors") or [{}])[0].get("neighbors", [])
        return [
            (item["datapoint"]["datapointId"], float(item.get("distance", 0.0)))
            for item in neighbors
        ]

    def _find_neighbors_url(self) -> str:
        """Queries go to the endpoint's own public domain, looked up once."""
        if self._query_url is None:
            endpoint = self._session.get(self._endpoint_url, timeout=TIMEOUT_SECONDS)
            endpoint.raise_for_status()
            domain = endpoint.json()["publicEndpointDomainName"]
            self._query_url = f"https://{domain}/v1/{self._endpoint_name}:findNeighbors"
        return self._query_url

    def _post(self, url: str, body: dict) -> dict:
        response = self._session.post(url, json=body, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()


class InMemoryVectorStore:
    """Exact search in a dict, for tests."""

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def upsert(self, items: list[tuple[str, list[float]]]) -> None:
        self.vectors.update(items)

    def remove(self, ids: list[str]) -> None:
        for item_id in ids:
            self.vectors.pop(item_id, None)

    def query(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        scored = [
            (item_id, _cosine(vector, stored))
            for item_id, stored in self.vectors.items()
        ]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0
