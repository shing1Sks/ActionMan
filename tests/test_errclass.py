from actionman.errclass import classify_arg


def test_classes():
    q = "prices for Tesla and Microsoft, meeting at 11 PM with Lionel Messi"
    assert classify_arg("Teslala", ["Tesla"], q) == "garbled"
    assert classify_arg("Lion Messi", ["Lionel Messi"], q) == "garbled"
    assert classify_arg("Micro", ["Microsoft"], q) == "truncated"
    assert classify_arg("Tesla,Microsoft", [["Tesla", "Microsoft"]], q) == "list_format"
    assert classify_arg(11, [23], q) == "reasoning"
    assert classify_arg("Messi", ["Lionel Messi"], q) == "span_length"
    assert classify_arg(None, ["Tesla"], q) == "missing"
