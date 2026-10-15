"""Inline data.json (+ trace.json) into data.js so index.html works from file:// (browsers block fetch() there).

  python build_data_js.py            -> data.js  (window.FILM = data.json, window.FILM.trace = trace.json)

Benchmark numbers are copied into data.json by hand on purpose, so every on-screen number is reviewed once.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
trace_p = HERE / "trace.json"
data["trace"] = json.loads(trace_p.read_text(encoding="utf-8")) if trace_p.exists() else None
(HERE / "data.js").write_text("window.FILM = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
print("data.js written", "(with trace)" if data["trace"] else "(no trace.json: mechanism scene uses its built-in fallback)",
      "status:", data.get("_status"))
