"""Span pointer decoding for the v3 copy head."""
from __future__ import annotations

import torch


def best_span(start: torch.Tensor, end: torch.Tensor, max_len: int = 32):
    """Highest p(start) * p(end) with start <= end < start + max_len -> (s, e, prob)."""
    ps, pe = start.float().softmax(-1), end.float().softmax(-1)
    Q = ps.shape[0]
    i = torch.arange(Q, device=ps.device)
    ok = (i[None, :] >= i[:, None]) & (i[None, :] < i[:, None] + max_len)
    score = (ps[:, None] * pe[None, :]).masked_fill(~ok, 0)
    flat = int(score.argmax())
    return flat // Q, flat % Q, float(score.flatten()[flat])


def _cls(ch: str) -> str:
    return "a" if ch.isalpha() else "d" if ch.isdigit() else ""


def snap_word(text: str, s: int, e: int) -> tuple[int, int]:
    """Widen the char span [s, e) so it never cuts a word or number in half ("US Doll" -> "US Dollars").
    Only letter-letter and digit-digit edges are extended, so "100" stays "100" inside "100km"."""
    while s > 0 and s < e and _cls(text[s]) and _cls(text[s - 1]) == _cls(text[s]):
        s -= 1
    while e < len(text) and e > s and _cls(text[e - 1]) and _cls(text[e]) == _cls(text[e - 1]):
        e += 1
    return s, e
