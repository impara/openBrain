from openbrain.domain.search import (
    SEARCH_INTENT_REFERENCE,
    _best_matching_segment,
    build_candidates,
    detect_search_intent,
    rank_evidence,
    synthesize_answer,
)
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
    assert "49:13" not in answer


def test_detect_search_intent_classifies_citation_queries_as_reference():
    assert detect_search_intent("What does 28:88 say") == SEARCH_INTENT_REFERENCE


def test_rank_evidence_drops_non_citation_items_for_reference_queries():
    evidence = build_candidates(
        "What does 28:88 say",
        vector_results=[
            {
                "content": "Everything is perishing except His Face. (28:88)",
                "score": 0.01,
                "metadata": {"origin": "capture_thought", "ingest_mode": "raw"},
            },
            {
                "content": "Indeed, We began creation and We will repeat it, so that you may return. (21:104)",
                "score": 0.02,
                "metadata": {"origin": "capture_thought", "ingest_mode": "raw"},
            },
        ],
        relations=[],
        raw_matches=[],
    )
    ranked, debug = rank_evidence("What does 28:88 say", evidence, intent=SEARCH_INTENT_REFERENCE)
    assert len(ranked) == 1
    assert "28:88" in ranked[0].text
    assert any(item["reason"] == "missing_citation" for item in debug["dropped"])


def test_best_matching_segment_does_not_merge_previous_verse_when_target_segment_is_complete():
    text = (
        "“We have made you into nations and tribes so that you may know one another.” (49:13)\n\n"
        "“Everything is perishing except His Face.” (28:88)\n\n"
        "Diversity is not to be worshipped, nor dismissed."
    )
    segment = _best_matching_segment(
        text,
        query_tokens=tokenize("What does 28:88 say"),
        query_citations=["28:88"],
    )
    assert "Everything is perishing except His Face" in segment
    assert "49:13" not in segment


def test_synthesize_answer_prefers_managed_memory_when_present():
    evidence = build_candidates(
        "what counterpoints",
        vector_results=[
            {
                "content": "Provide counterpoints.",
                "score": 0.02,
                "metadata": {"origin": "capture_thought", "ingest_mode": "fact"},
            }
        ],
        relations=[],
        raw_matches=[],
        managed_results=[
            {
                "id": 1,
                "kind": "directive",
                "topic": "conversation style",
                "topic_key": "conversation-style",
                "canonical_text": (
                    "Act as an intellectual sparring partner. Analyze assumptions, provide counterpoints, "
                    "test reasoning, offer alternatives, and prioritize truth over agreement"
                ),
                "score": 0.01,
            }
        ],
    )
    answer = synthesize_answer("what counterpoints", evidence)
    assert "Based on your active directive" in answer
    assert "intellectual sparring partner" in answer
