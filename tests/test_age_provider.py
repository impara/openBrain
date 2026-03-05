"""
Unit tests for ApacheAGEProvider.

These tests mock the database connection pool — they don't require a running Postgres instance.
Run with:  python -m pytest tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from age_provider import ApacheAGEProvider, _LABEL_RE, _PROP_VALUE_MAX_LEN


# ── Fixtures ──────────────────────────────────


@pytest.fixture
def mock_pool():
    """Create a mock connection pool that yields mock conn/cursor."""
    with patch("age_provider.pool.ThreadedConnectionPool") as MockPool:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.closed = False

        pool_instance = MockPool.return_value
        pool_instance.getconn.return_value = mock_conn

        yield pool_instance, mock_conn, mock_cursor


@pytest.fixture
def provider(mock_pool):
    """Create an ApacheAGEProvider with mocked infrastructure."""
    _, mock_conn, mock_cursor = mock_pool
    provider = ApacheAGEProvider(
        dbname="test_db",
        user="test_user",
        password="test_pass",
    )
    return provider


# ── Graph Name Validation ─────────────────────


class TestGraphNameValidation:
    def test_valid_graph_name(self):
        """Alphanumeric graph names should be accepted."""
        with patch("age_provider.pool.ThreadedConnectionPool"):
            p = ApacheAGEProvider(
                dbname="db", user="u", password="p", graph_name="brain_graph"
            )
            assert p.graph_name == "brain_graph"

    def test_rejects_injection_in_graph_name(self):
        """Graph names with special chars should be rejected at construction."""
        with pytest.raises(ValueError, match="Invalid graph_name"):
            with patch("age_provider.pool.ThreadedConnectionPool"):
                ApacheAGEProvider(
                    dbname="db",
                    user="u",
                    password="p",
                    graph_name="brain'; DROP TABLE users;--",
                )

    def test_rejects_empty_graph_name(self):
        with pytest.raises(ValueError):
            with patch("age_provider.pool.ThreadedConnectionPool"):
                ApacheAGEProvider(dbname="db", user="u", password="p", graph_name="")


# ── Label Validation ──────────────────────────


class TestLabelValidation:
    def test_valid_labels(self):
        assert ApacheAGEProvider._validate_label("Entity") == "Entity"
        assert ApacheAGEProvider._validate_label("PersonNode") == "PersonNode"
        assert ApacheAGEProvider._validate_label("type_2") == "type_2"

    def test_rejects_sql_injection_label(self):
        with pytest.raises(ValueError, match="Invalid Cypher label"):
            ApacheAGEProvider._validate_label("Entity; DROP TABLE")

    def test_rejects_cypher_injection_label(self):
        with pytest.raises(ValueError, match="Invalid Cypher label"):
            ApacheAGEProvider._validate_label("Entity})-[:X]->(")

    def test_rejects_empty_label(self):
        with pytest.raises(ValueError):
            ApacheAGEProvider._validate_label("")

    def test_rejects_numeric_start(self):
        with pytest.raises(ValueError):
            ApacheAGEProvider._validate_label("123Entity")

    def test_rejects_overly_long_label(self):
        with pytest.raises(ValueError):
            ApacheAGEProvider._validate_label("A" * 65)


# ── Cypher String Escaping ────────────────────


class TestCypherStringEscaping:
    def test_escapes_single_quotes(self):
        result = ApacheAGEProvider._escape_cypher_string("it's a test")
        assert result == "it\\'s a test"

    def test_escapes_backslashes_first(self):
        """Backslashes must be escaped before quotes to prevent double-escaping."""
        result = ApacheAGEProvider._escape_cypher_string("path\\to\\file")
        assert result == "path\\\\to\\\\file"

    def test_handles_backslash_quote_combo(self):
        """The classic injection vector: backslash followed by quote."""
        # Input: one backslash + one single-quote
        input_str = chr(92) + chr(39)
        result = ApacheAGEProvider._escape_cypher_string(input_str)
        # Expected: two backslashes + one backslash + one quote
        expected = chr(92) + chr(92) + chr(92) + chr(39)
        assert result == expected, f"Got: {result!r}, expected: {expected!r}"

    def test_enforces_max_length(self):
        with pytest.raises(ValueError, match="max length"):
            ApacheAGEProvider._escape_cypher_string("A" * (_PROP_VALUE_MAX_LEN + 1))

    def test_allows_max_length_exactly(self):
        result = ApacheAGEProvider._escape_cypher_string("A" * _PROP_VALUE_MAX_LEN)
        assert len(result) == _PROP_VALUE_MAX_LEN

    def test_normal_string_unchanged(self):
        result = ApacheAGEProvider._escape_cypher_string("hello world 123")
        assert result == "hello world 123"


# ── Sanitize Props ────────────────────────────


class TestSanitizeProps:
    def test_empty_dict_returns_empty(self, provider):
        assert provider._sanitize_props({}) == ""
        assert provider._sanitize_props(None) == ""

    def test_strips_invalid_key_chars(self, provider):
        result = provider._sanitize_props({"na-me!@#": "value"})
        assert "name" in result
        assert "!" not in result

    def test_escapes_values(self, provider):
        result = provider._sanitize_props({"name": "O'Brien"})
        assert "O\\'Brien" in result

    def test_multiple_props(self, provider):
        result = provider._sanitize_props({"name": "Alice", "age": "30"})
        assert "name: 'Alice'" in result
        assert "age: '30'" in result


# ── Search Term Validation ────────────────────


class TestSearchTermValidation:
    def test_valid_terms(self):
        assert ApacheAGEProvider._validate_search_term("hello") == "hello"
        assert ApacheAGEProvider._validate_search_term("test-123") == "test-123"
        assert ApacheAGEProvider._validate_search_term("user@email") == "user@email"

    def test_rejects_cypher_injection(self):
        with pytest.raises(ValueError, match="Invalid search term"):
            ApacheAGEProvider._validate_search_term("test') OR 1=1--")

    def test_truncates_long_terms(self):
        long_term = "A" * 300
        result = ApacheAGEProvider._validate_search_term(long_term)
        assert len(result) == 200  # _SEARCH_TERM_MAX_LEN

    def test_strips_whitespace(self):
        result = ApacheAGEProvider._validate_search_term("  hello  ")
        assert result == "hello"


# ── Agtype Parsing ────────────────────────────


class TestAgtypeParsing:
    def test_parses_vertex(self):
        raw = '{"id": 1, "label": "Person", "properties": {"name": "Alice"}}::vertex'
        result = ApacheAGEProvider._parse_agtype(raw)
        assert result["label"] == "Person"
        assert result["properties"]["name"] == "Alice"

    def test_parses_edge(self):
        raw = '{"id": 1, "label": "KNOWS", "start_id": 1, "end_id": 2}::edge'
        result = ApacheAGEProvider._parse_agtype(raw)
        assert result["label"] == "KNOWS"

    def test_handles_none(self):
        assert ApacheAGEProvider._parse_agtype(None) == {}

    def test_handles_malformed_json(self):
        result = ApacheAGEProvider._parse_agtype("not-json::vertex")
        assert "raw" in result

    def test_handles_no_type_suffix(self):
        raw = '{"id": 1, "label": "Test"}'
        result = ApacheAGEProvider._parse_agtype(raw)
        assert result["label"] == "Test"


# ── capture_thought Bug Regression ────────────


class TestCaptureThoughtRegression:
    """Regression test for the critical bug where the literal string 'thought'
    was passed instead of the variable."""

    def test_thought_variable_is_passed_not_literal(self):
        """Verify open_brain_mcp passes the actual thought content to Mem0."""
        # Read the source and verify the fix
        with open("open_brain_mcp.py", "r") as f:
            source = f.read()

        # The buggy version had: "content": "thought" (string literal)
        # The fixed version has:  "content": thought  (variable reference)
        assert '"content": "thought"' not in source, (
            "BUG: capture_thought still passes the literal string 'thought' "
            "instead of the variable"
        )
        assert '"content": thought' not in source or "'content': thought" not in source or (
            # Check for the correct pattern (variable, not string)
            'content": thought}' in source or "content': thought}" in source
        ), "capture_thought should pass the thought variable"
