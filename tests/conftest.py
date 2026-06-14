import pytest


@pytest.fixture(autouse=True)
def default_fast_keyword_provider(monkeypatch):
    monkeypatch.setenv("KEYWORD_PROVIDER", "statistical")
