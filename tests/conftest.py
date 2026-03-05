"""
conftest.py — Mock external dependencies that aren't installed in the test environment.

This allows unit tests to import age_provider.py without needing the full
mem0 library installed, since we only need to test our own code.
"""

import sys
from unittest.mock import MagicMock

# Mock the mem0 module hierarchy so age_provider.py can import BaseGraphProvider
mem0_mock = MagicMock()
mem0_mock.storage = MagicMock()
mem0_mock.storage.graph = MagicMock()
mem0_mock.storage.graph.base = MagicMock()
mem0_mock.storage.graph.base.BaseGraphProvider = type("BaseGraphProvider", (), {})

sys.modules["mem0"] = mem0_mock
sys.modules["mem0.storage"] = mem0_mock.storage
sys.modules["mem0.storage.graph"] = mem0_mock.storage.graph
sys.modules["mem0.storage.graph.base"] = mem0_mock.storage.graph.base
