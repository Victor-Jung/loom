"""benchmarks/run.py: successive halving over compiled candidates."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bench_run", ROOT / "benchmarks/run.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_halving_keeps_the_faster_half_and_retimes_survivors(monkeypatch) -> None:
    # Round 1 says [2] is fastest; on re-timing [3] turns out faster.
    later = {"f2": 1.10, "f3": 0.90, "f1": 1.50}
    calls = []

    def fake_run(program, kdir, shapes, iters, *rest):
        calls.append((kdir.name, iters))
        return {"device_ms": later[kdir.name]}

    monkeypatch.setattr(R, "run_candidate_in_process", fake_run)
    monkeypatch.setattr(R, "arg_shapes", lambda out, func: [])
    rows = [{"stage": "ok", "device_ms": ms, "index": i, "function": f"fn{i}", "kdir": f"k/f{i}"}
            for i, ms in ((1, 1.2), (2, 1.0), (3, 1.05), (4, 1.4))]
    rows.append({"stage": "run", "error": "hang", "index": 5})
    a = types.SimpleNamespace(iters=5, warmup=1, timeout=10, fidelity="HiFi4", program="p",
                              timing="trace", per_sample=20)
    sel = R.successive_halving(a, rows, Path("out"), True)
    # 4 passed -> 2 survivors ([2], [3]) re-timed with 10 samples -> [3].
    assert sorted(calls) == [("f2", 10), ("f3", 10)]
    assert sel["index"] == 3 and sel["device_ms"] == 0.90 and sel["rounds"] == 2
    assert sel["solver_first_ms"] == 1.2
