"""Shared pytest fixtures for ml-automation-llm."""
import pytest


@pytest.fixture
def mock_llm_response():
    """LLM response-shaped dict for testing."""
    return {
        "id": "chatcmpl-123456",
        "object": "text_completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "text": "This is a sample LLM response.",
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def sample_dataset():
    """10 row dicts dataset for testing."""
    return [
        {"id": i, "text": f"Sample text {i}", "label": i % 2}
        for i in range(10)
    ]


@pytest.fixture
def temp_workspace(tmp_path):
    """Temporary workspace directory for testing."""
    return tmp_path
