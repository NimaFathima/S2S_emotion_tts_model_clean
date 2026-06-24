"""
test_brow_gate.py — proves BrowTemporalGate is safe.

Run: python processes/test_brow_gate.py   (from emotion_module/)

Checks:
  1. IDENTITY      — temporal=False reproduces the prior single-frame gate exactly.
  2. STABILITY     — temporal=True suppresses single-frame flicker.
  3. NO-DEGRADE    — temporal=True still registers a sustained question, and never
                     turns a stable statement into a question.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processes.brow_gate import BrowTemporalGate
from config.settings import (
    BROW_RAISE_THRESHOLD as RT,
    BROW_FURROW_THRESHOLD as FT,
    AFFECT_CONFOUND_THRESHOLD as CT,
)


def _old_inline(rel, sur, ang):
    raw_raise = rel > RT
    raw_furrow = rel < FT
    raise_gram = (sur is None) or (sur < CT)
    furrow_gram = (ang is None) or (ang < CT)
    return (raw_raise and raise_gram,
            raw_furrow and furrow_gram,
            (raw_raise and not raise_gram) or (raw_furrow and not furrow_gram))


def test_identity():
    g = BrowTemporalGate(RT, FT, CT, temporal=False)
    for rel in [-0.2, -0.06, -0.05, 0.0, 0.05, 0.08, 0.09, 0.3]:
        for sur in [None, 0.0, 0.29, 0.30, 0.5]:
            for ang in [None, 0.0, 0.29, 0.30, 0.5]:
                assert g.update(rel, sur, ang) == _old_inline(rel, sur, ang)
    print("[OK] identity: temporal=False matches the prior single-frame gate")


def _run(seq, temporal):
    g = BrowTemporalGate(RT, FT, CT, temporal=temporal, stable_frames=3)
    flips, prev, yn_frames = 0, (False, False, False), 0
    for rel, sur, ang in seq:
        out = g.update(rel, sur, ang)
        if out != prev:
            flips += 1
        if out[0]:
            yn_frames += 1
        prev = out
    return flips, yn_frames


def test_stability():
    # statement with brief 1-frame surprise spikes
    stmt = [(0.09, 0.6, 0.0) if i % 9 == 0 else (0.0, 0.02, 0.02) for i in range(60)]
    f_single, yn_single = _run(stmt, False)
    f_temporal, yn_temporal = _run(stmt, True)
    assert f_temporal < f_single, (f_temporal, f_single)
    assert yn_single == 0 and yn_temporal == 0          # never a false question
    print(f"[OK] stability: flicker flips {f_single} -> {f_temporal}, "
          f"false-question frames stay 0")


def test_no_degrade():
    yn = [(0.12, 0.02, 0.02)] * 20                      # sustained grammatical Y/N
    _, yn_single = _run(yn, False)
    _, yn_temporal = _run(yn, True)
    assert yn_single > 0 and yn_temporal > 0            # question still registers
    print(f"[OK] no-degrade: sustained Y/N registers (single={yn_single}, "
          f"temporal={yn_temporal})")


if __name__ == "__main__":
    test_identity()
    test_stability()
    test_no_degrade()
    print("\nAll BrowTemporalGate checks passed.")
