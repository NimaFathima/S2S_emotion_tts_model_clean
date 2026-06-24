# Affective vs. Grammatical Expression — Design

> The core risk of a sign-to-speech system that conveys *tone* is misrepresenting
> what the signer meant. In sign languages, facial expression is **layered**: the
> same muscles carry both **grammar** (non-manual markers — NMMs) and **emotion**
> (affect). A raised brow can mark a yes/no question *or* signal surprise. If the
> system confuses the two, it will add a question intonation to a statement, or
> flatten a signer's genuine feeling as if it were grammar. This document explains
> how Signet Aid separates the two.

## The problem

Eyebrow position in particular is grammatically loaded in ASL:

| Brow | Grammatical meaning | Emotional confound |
|------|--------------------|--------------------|
| Raised | Yes/No question, topic marking | Surprise, fear |
| Furrowed | WH-question, conditional | Anger, concentration, concern |

A naive system that reads "brows raised → surprise" (a generic facial-emotion
model) will mis-speak every yes/no question as excited surprise. A naive system
that reads "brows raised → question" will turn every shocked statement into a
question. **Both fail.** The brow alone is ambiguous.

## The principle that makes it solvable

Grammar and affect are not actually inseparable — they differ on measurable axes.
We exploit two of them, plus the text the translation module already produces:

1. **The linguistic channel (text).** For ASL→English, the translated text
   reliably carries WH-questions (a WH-word appears) and lexical negation
   (`not/never/no…`). It does **not** reliably carry yes/no questions — in ASL
   "YOU GO STORE" + brow-raise often translates to the same words as the
   statement. So *text owns WH and negation; the brow is only load-bearing for
   yes/no.* This concentrates all face-based risk on one narrow case.

2. **The face concordance axis (which muscles co-activate).** A *grammatical*
   brow movement is **brow-isolated** — the rest of the face stays relatively
   neutral. A *genuine emotion* recruits the **whole face** (Duchenne
   co-variation): surprise widens the eyes and drops the jaw; anger sneers the
   nose, presses the lips and squints the eyes. We read these co-markers from
   MediaPipe **blendshapes** and use them, per-direction, to decide whether a
   brow movement is grammatical or affective.

## The three-channel design

```
         ┌─ TEXT CHANNEL (translated English) ───────────────────────┐
         │  WH-word? lexical negation? trailing '?'? fronted aux?    │ → owns WH / neg / explicit Y/N
         └───────────────────────────────────────────────────────────┘
                              │
 brow event ──► CONFOUND GATE (blendshapes) ──► grammatical vs affective
                              │     raise  is grammatical unless SURPRISE markers (eyeWide, jawOpen)
                              │     furrow is grammatical unless ANGER markers (noseSneer, mouthPress, eyeSquint)
                              ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ FUSION (resolve_sentence_type)                                     │
   │   text says WH/neg/Y/N  → use it; brow movement explained as grammar│
   │   text silent + gated Y/N brow → yes/no question                    │
   │   text silent + brow failed gate → statement, brow kept as EMOTION  │
   └───────────────────────────────────────────────────────────────────┘
                              │
        punctuation ('?') from the RESOLVED type, never the raw brow
        emotion dampened ONLY for genuine brow-isolated grammar
```

### Where each piece lives

| Concern | Where |
|---------|-------|
| Text → sentence type | `processes/audio_consumer.py` → `sentence_type_from_text()` |
| Fusion (text + gated brow) | `processes/audio_consumer.py` → `resolve_sentence_type()` |
| Punctuation from resolved type | `processes/audio_consumer.py` → `VAToChatterbox.modify_text()` |
| Confound gate (blendshapes) | `processes/nmm_classifier.py` → `_confound_scores()`; decision in `processes/brow_gate.py` → `BrowTemporalGate` |
| Conditional emotion dampening | `processes/nmm_classifier.py` → `apply_dampening()` |
| Thresholds | `config/settings.py` |

## How the hard cases resolve

| Case | Outcome |
|------|---------|
| **Surprised statement** ("That just happened!", shocked brows) | gate sees eyeWide/jawOpen → `brow_affective`, no Y/N flag; text silent → **statement, emotion kept** (no false "?") |
| **Angry WH-question** ("WHY did you do that?!") | text has WH → "?" added; gate sees anger markers → furrow is affective → **anger preserved** |
| **Calm grammatical Y/N** ("You go store?", isolated raise) | gate sees quiet lower face → Y/N flag; text silent → "?"; emotion dampened toward neutral → **neutral question** |
| **Smiling Y/N question** | smile does not trigger surprise markers → raise still grammatical → **question detected + happy tone kept** |

## Known limitation (the honest residual)

A **surprise-toned yes/no question** (genuine wide-eyed shock *and* a real
question, with no WH-word in the text) is the one case the single-frame gate
cannot fully resolve: the surprise markers make the gate reject the brow, so the
question mark may be missed (the emotion is still preserved). This is an inherent
limit of frame-level disambiguation. It is **measured, not hidden** — see the
`question_recall` metric in the eval harness.

**Why a face-only temporal trick does not fix it:** grammatical NMMs are sharply
time-locked to the *clause*, but knowing the clause boundary requires timing from
the gloss/manual stream, which the upstream supplies only as text. A face-only
attempt to force this recall up would risk the 0% false-question rate, so it is
deliberately not attempted. The temporal layer that *is* implemented
(`BrowTemporalGate`, `TEMPORAL_GATE`) is a **stability** measure (hysteresis to
suppress flicker), not a recall fix. The real fix needs a richer upstream
sentence-type/timing signal — see the roadmap in [PROJECT_SUMMARY](PROJECT_SUMMARY.md).

## Validating it

`evaluate_nmm.py` turns this design into numbers. The headline metric is
**`false_question_rate`** — how often a true statement is wrongly spoken as a
question (the core safety failure). See [README](README.md#evaluation) for usage.
Record clips that stress the trap cases (surprised statements, emotional
questions) and tune `AFFECT_CONFOUND_THRESHOLD` and `BROW_RAISE_THRESHOLD`
against them; report the before/after `false_question_rate` as evidence.
