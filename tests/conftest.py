import pytest

from actionman.features import load_tokenizer
from actionman.schema import Call, Example, Param, Tool


@pytest.fixture(scope="session")
def tok():
    return load_tokenizer("answerdotai/ModernBERT-large")


def make_example(query="find ramen near shibuya top 3", q="best ramen near shibuya", k=3, extra_calls=()):
    tools = [Tool("web_search", "Search the web", [Param("q", "str", False, None, "query"),
                                                   Param("k", "int", True, 5, "")]),
             Tool("get_weather", "Weather for a city", [Param("city", "str", False, None, "")])]
    calls = [Call("web_search", {"q": q, "k": k}), *extra_calls]
    return Example(query, tools, calls, id="t", source="test")


@pytest.fixture
def example():
    return make_example()
