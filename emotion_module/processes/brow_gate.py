"""
brow_gate.py — grammatical-vs-affective brow decision (pure, testable).

Separates the brow CLASSIFICATION logic from MediaPipe I/O so it can be unit-
tested and driven by synthetic frame sequences (see evaluate_nmm.py).

Modes:
  temporal=False  reproduces the single-frame confound gate EXACTLY (the prior
                  inline logic in NMMClassifier) — used as the safe baseline.
  temporal=True   adds HYSTERESIS: a change in the grammatical/affective decision
                  must persist `stable_frames` consecutive frames before it takes
                  effect. This suppresses single-frame flicker (steadier output)
                  and, crucially, can NEVER turn a *stable* statement into a
                  question — a sustained no-brow state simply stays. Brief real
                  events are already gated downstream (SUSTAIN_FRAMES_REQUIRED),
                  so hysteresis adds no new recall loss.
"""


class BrowTemporalGate:
    def __init__(self, raise_thr, furrow_thr, confound_thr,
                 temporal=False, stable_frames=3):
        self.raise_thr = raise_thr
        self.furrow_thr = furrow_thr
        self.confound_thr = confound_thr
        self.temporal = temporal
        self.stable_frames = stable_frames
        self._last = (False, False, False)   # last emitted (yn, wh, affective)
        self._pending = None
        self._pending_count = 0

    def reset(self):
        self._last = (False, False, False)
        self._pending = None
        self._pending_count = 0

    def _single_frame(self, relative_raise, surprise_act, anger_act):
        """Single-frame decision — identical to the prior inline gate."""
        raw_raise = relative_raise > self.raise_thr
        raw_furrow = relative_raise < self.furrow_thr
        # surprise/anger None => blendshapes unavailable => assume grammatical
        raise_gram = (surprise_act is None) or (surprise_act < self.confound_thr)
        furrow_gram = (anger_act is None) or (anger_act < self.confound_thr)
        is_yn = raw_raise and raise_gram
        is_wh = raw_furrow and furrow_gram
        affective = ((raw_raise and not raise_gram) or
                     (raw_furrow and not furrow_gram))
        return (is_yn, is_wh, affective)

    def update(self, relative_raise, surprise_act=None, anger_act=None):
        """Return (is_yn_question, is_wh_question, brow_affective)."""
        decision = self._single_frame(relative_raise, surprise_act, anger_act)
        if not self.temporal:
            return decision

        # Hysteresis: adopt a new decision only after it persists stable_frames.
        if decision == self._last:
            self._pending = None
            self._pending_count = 0
            return self._last

        if decision == self._pending:
            self._pending_count += 1
        else:
            self._pending = decision
            self._pending_count = 1

        if self._pending_count >= self.stable_frames:
            self._last = self._pending
            self._pending = None
            self._pending_count = 0
        return self._last
