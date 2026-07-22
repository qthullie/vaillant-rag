from unittest.mock import MagicMock, patch

from vaillant_rag.rerank import Reranker
from vaillant_rag.retrieval import RetrievedChunk

CANDIDATES = [
    RetrievedChunk("d0#chunk-0", "Cats are mammals.", 0.9),
    RetrievedChunk("d1#chunk-0", "Paris is in France.", 0.8),
    RetrievedChunk("d2#chunk-0", "Water boils at 100C.", 0.7),
]


def test_rerank_reorders_by_cross_encoder_score():
    fake_model = MagicMock()
    fake_model.predict.return_value = [0.1, 0.9, 0.5]  # favors Paris chunk
    with patch("vaillant_rag.rerank._load_cross_encoder", return_value=fake_model):
        reranker = Reranker("fake-cross-encoder")
        result = reranker.rerank("Where is Paris?", CANDIDATES, top_n=2)

    assert [c.chunk_id for c in result] == ["d1#chunk-0", "d2#chunk-0"]
    assert result[0].score == 0.9
    # Pairs sent to the model are (query, chunk_text).
    pairs = fake_model.predict.call_args.args[0]
    assert pairs[0] == ("Where is Paris?", "Cats are mammals.")


def test_rerank_empty_candidates():
    reranker = Reranker("fake-cross-encoder")
    assert reranker.rerank("anything", [], top_n=3) == []
