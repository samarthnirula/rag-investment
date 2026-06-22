from __future__ import annotations

from unittest.mock import Mock, patch

from insightlens.embeddings.embedder import Embedder


def test_voyage_embeddings_use_lightweight_http_api(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2]},
        ]
    }

    with patch("insightlens.embeddings.embedder.requests.post", return_value=response) as post:
        results = Embedder("unused-local-model").embed_texts(["first", "second"])

    assert [result.vector for result in results] == [[0.1, 0.2], [0.3, 0.4]]
    post.assert_called_once()
    assert post.call_args.kwargs["json"]["input_type"] == "document"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
