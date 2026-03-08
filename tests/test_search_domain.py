from openbrain.domain.search import build_candidates, synthesize_answer
from openbrain.domain.text import tokenize


class RawMatch:
    def __init__(self, content: str):
        self.id = 1
        self.content = content
        self.created_at = None
        self.content_len = len(content)
        self.source = "capture_thought"


def test_tokenize_preserves_quran_style_citation():
    tokens = tokenize("What does 28:88 say")
    assert "28:88" in tokens


def test_build_candidates_prefers_sentence_containing_citation():
    text = (
        "We have made you into nations and tribes so that you may know one another. "
        "Everything is perishing except His Face. (28:88) Diversity is not to be worshipped."
    )
    candidates = build_candidates(
        "What does 28:88 say",
        vector_results=[{"content": text, "score": 0.01, "metadata": {"origin": "capture_thought"}}],
        relations=[],
        raw_matches=[RawMatch(text)],
    )
    assert any("28:88" in candidate.text for candidate in candidates)
    assert not any(candidate.text.startswith("...") for candidate in candidates if candidate.source == "raw")


def test_synthesize_answer_is_query_aware_for_quote_style_queries():
    evidence = build_candidates(
        "What does 28:88 say",
        vector_results=[
            {
                "content": (
                    "We have made you into nations and tribes so that you may know one another. "
                    "Everything is perishing except His Face. (28:88) Diversity is not to be worshipped."
                ),
                "score": 0.01,
                "metadata": {"origin": "capture_thought"},
            }
        ],
        relations=[],
        raw_matches=[],
    )
    answer = synthesize_answer("What does 28:88 say", evidence)
    assert "28:88 says" in answer
    assert "Everything is perishing except His Face" in answer
