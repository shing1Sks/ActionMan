from actionman.audit import copy_span


def _text(tok, q, span):
    offs = tok(q, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    s, e = span
    return q[offs[s][0]: offs[e][1]].strip()


def test_copy_span_token_aligned_case_insensitive(tok):
    q = "Find sushi in boston, MA tonight"
    assert _text(tok, q, copy_span(tok, q, "Boston, MA")).lower() == "boston, ma"


def test_copy_span_rejects_absent(tok):
    assert copy_span(tok, "weather in NYC", "London") is None
    assert copy_span(tok, "weather in NYC", "") is None


def test_copy_span_rejects_partial_token(tok):
    q = "weather in NYC"
    toks = tok(q, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
    if any(q[a:b].strip() == "NYC" for a, b in toks):      # only meaningful if NYC is one token
        assert copy_span(tok, q, "NY") is None


def test_copy_span_non_ascii_round_trips(tok):
    q = "Book a table at Café Müller in Zürich"
    assert _text(tok, q, copy_span(tok, q, "Café Müller")) == "Café Müller"
