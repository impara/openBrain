from openbrain.infrastructure.extraction import LLMManagedMemoryExtractor


class FakeStructuredProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate_json(self, *, system_prompt: str, user_prompt: str):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return dict(self.payload)


def test_managed_memory_extractor_merges_style_clause_into_preference_and_avoids_extra_directive():
    provider = FakeStructuredProvider(
        {
            "memories": [
                {
                    "kind": "preference",
                    "topic": "response style",
                    "canonical_text": "I prefer concise answers by default.",
                    "evidence_text": "/remember I prefer concise answers by default...",
                },
                {
                    # This is the failure case: should not become a separate directive.
                    "kind": "directive",
                    "topic": "response style",
                    "canonical_text": "Only go deep when I explicitly ask for detail.",
                    "evidence_text": "/remember I prefer concise answers by default...",
                },
            ]
        }
    )
    extractor = LLMManagedMemoryExtractor(provider)
    results = extractor.extract("I prefer concise answers by default. Only go deep when I explicitly ask for detail.")
    assert len(results) == 1
    assert results[0].kind == "preference"
    assert "concise answers" in results[0].canonical_text.lower()
    assert "only go deep" in results[0].canonical_text.lower()

