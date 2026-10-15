from actionman.distill import merge_teacher
from actionman.schema import Call


def test_merge_takes_teacher_surface_form_keeps_original_truth():
    orig = [Call("f", {"airline": "american", "city": "Paris", "q": "Berlin, Germany"})]
    teach = [Call("f", {"airline": "American Airlines", "city": "London", "q": "Berlin"})]
    calls, st = merge_teacher(orig, teach, "fly American Airlines to Paris, then Berlin")
    assert calls[0].args == {"airline": "American Airlines", "city": "Paris", "q": "Berlin"} and st["changed"] == 2


def test_merge_rejects_fact_changes_hidden_by_containment():
    # final-review finding: "1" in "10", invented day, ticker appended, substring-matched added arg, undeclared arg
    orig = [Call("f", {"date": "2023-05", "page": "1", "sym": "AAPL"})]
    teach = [Call("f", {"date": "2023-05-01", "page": "10", "sym": "AAPL,MSFT", "limit": 1, "bogus": "top"})]
    calls, st = merge_teacher(orig, teach, "show top 10 results for AAPL in 2023-05", params={"f": {"date", "page", "sym", "limit"}})
    assert calls[0].args == {"date": "2023-05", "page": "1", "sym": "AAPL"} and st["changed"] == 0 and st["added"] == 0


def test_merge_rejects_different_tools():
    calls, st = merge_teacher([Call("f", {"x": 1})], [Call("g", {"x": 1})], "q")
    assert calls[0].name == "f" and calls[0].args == {"x": 1} and st["rejected_call"] == 1


def test_merge_keeps_original_type_for_numbers():
    calls, st = merge_teacher([Call("f", {"n": 5})], [Call("f", {"n": "5"})], "five")
    assert calls[0].args == {"n": 5} and st["changed"] == 0


def test_merge_lists_elementwise_and_extra_copyable_arg():
    orig = [Call("f", {"cos": ["Apple", "Tesla"]})]
    teach = [Call("f", {"cos": ["Apple Inc.", "Tesla"], "unit": "USD", "bogus": "zzz"})]
    calls, st = merge_teacher(orig, teach, "prices of Apple Inc. and Tesla in USD", params={"f": {"cos", "unit"}})
    assert calls[0].args == {"cos": ["Apple Inc.", "Tesla"], "unit": "USD"} and st["added"] == 1


def test_merge_none_teacher_keeps_original():
    calls, st = merge_teacher([Call("f", {"x": "a"})], None, "q")
    assert calls[0].args == {"x": "a"} and st["rejected_call"] == 1
