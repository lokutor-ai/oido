#!/usr/bin/env python3
"""Tests for board_bench.parse: the RTF of a streaming clip must come from the STREAM line, not from the UTT line.

usage: python3 esp32/tools/test_board_bench_parse.py   (no board and no pyserial needed)
"""
import importlib.util
import os
import sys
import types

sys.modules.setdefault("serial", types.ModuleType("serial"))
spec = importlib.util.spec_from_file_location("board_bench", os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_bench.py"))
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_file_mode_wall_clock():
    utts, _ = bb.parse("UTT 0 | audio 5.00s | cycles 1200000000 | RTF 1.00 | wall 10000000 us\nHYP: hello\n")
    assert len(utts) == 1 and utts[0]["wall_us"] == 10000000
    assert close(utts[0]["rtf"], 2.0)


def test_file_mode_cycles_only():
    utts, _ = bb.parse("UTT 0 | audio 5.00s | cycles 1200000000\nHYP: hello\n")
    assert utts[0]["wall_us"] is None and close(utts[0]["rtf"], 1.0)


def test_stream_uses_feed_and_finish_wall_time():
    text = ("UTT 0 | audio 5.13s | cycles 531 | wall 1 us\n"
            "STREAM 0 | feed cycles 833123289 | finish cycles 43137660 | feed wall 3471347 us | finish wall 179740 us\n"
            "SHYP: hello\n")
    utts, _ = bb.parse(text)
    u = utts[0]
    assert u["stream_feed_us"] == 3471347 and u["stream_finish_us"] == 179740
    assert u["wall_us"] == 3471347 + 179740
    assert close(u["rtf"], (3471347 + 179740) / 1e6 / 5.13)
    assert u["rtf"] > 0.5


def test_stream_without_wall_time_uses_cycles():
    text = ("UTT 0 | audio 5.00s | cycles 531\n"
            "STREAM 0 | feed cycles 1000000000 | finish cycles 200000000 |\n")
    utts, _ = bb.parse(text)
    assert close(utts[0]["rtf"], 1200000000 / 240e6 / 5.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
