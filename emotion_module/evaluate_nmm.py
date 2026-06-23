"""
evaluate_nmm.py — Signet Aid NMM / sentence-type evaluation harness

Measures whether the grammatical-vs-affective separation actually works, and
turns the two tunable thresholds (AFFECT_CONFOUND_THRESHOLD, BROW_RAISE_THRESHOLD)
into defensible numbers for the competition.

Two front-ends, one shared metrics engine:

  1) --from-features <csv>   (runnable with NO clips and NO models)
        Each row is one already-extracted clip. Use this to validate and tune
        the TEXT + fusion layer, and to sanity-check the metric maths.
        Columns (header required):
            text,true_type,true_affect,yn_face,wh_face,neg_face,brow_affective
          text          : the translated English sentence
          true_type     : ground truth — one of {yn, wh, neg, statement}
          true_affect   : ground truth — one of {emotional, neutral}
          yn_face       : 0/1 — did the (gated) Y/N brow flag fire on the clip
          wh_face       : 0/1 — gated WH flag (diagnostic only; text owns WH)
          neg_face      : 0/1 — gated negation flag (diagnostic only)
          brow_affective: 0/1 — was a brow movement classified as affective

  2) --from-video <manifest_csv>   (needs recorded clips + MediaPipe installed)
        Runs the real NMMClassifier over each clip to PRODUCE the per-clip
        features above, then evaluates. Manifest columns:
            clip_path,text,true_type,true_affect
        Clips should START with a ~0.5s neutral expression so the per-session
        baseline calibrates correctly (see --calib-frames).

Headline metrics:
  - false_question_rate : among true statements, how often we wrongly emit a
                          question ('?'). THE core safety number — lower is better.
  - question_recall     : among true questions, how often we emit the right type.
  - affect_preservation : among emotional statements, fraction kept as statements.
  - brow_affective P/R   : does the gate flag emotional brows as affective.

Usage:
    python evaluate_nmm.py --from-features eval_samples/sample_features.csv
    python evaluate_nmm.py --from-video manifest.csv --calib-frames 15 --out results.csv
"""

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict

from processes.audio_consumer import resolve_sentence_type

log = logging.getLogger("eval_nmm")

TYPES = ["yn", "wh", "neg", "statement"]
QUESTION_TYPES = {"yn", "wh"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared metrics engine
# ─────────────────────────────────────────────────────────────────────────────
def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def evaluate(rows: list) -> dict:
    """
    rows: list of dicts with keys
        text, true_type, true_affect, yn_face, wh_face, neg_face, brow_affective
    Returns a dict with per-row predictions and aggregate metrics.
    """
    per_row = []
    confusion = defaultdict(Counter)   # confusion[true][pred] = count

    # counters for headline metrics
    stmt_total = stmt_as_question = 0
    q_total = q_correct = 0
    emo_stmt_total = emo_stmt_kept = 0
    # brow_affective vs (true_affect == emotional)
    ba_tp = ba_fp = ba_fn = ba_tn = 0

    for r in rows:
        text = r["text"]
        true_type = r["true_type"].strip().lower()
        true_affect = r.get("true_affect", "").strip().lower()
        yn_face = _as_bool(r.get("yn_face", 0))
        brow_affective = _as_bool(r.get("brow_affective", 0))

        pred_type = resolve_sentence_type(text, yn_face)

        confusion[true_type][pred_type] += 1
        per_row.append({**r, "pred_type": pred_type})

        # false question rate (true statements predicted as a question)
        if true_type == "statement":
            stmt_total += 1
            if pred_type in QUESTION_TYPES:
                stmt_as_question += 1

        # question recall
        if true_type in QUESTION_TYPES:
            q_total += 1
            if pred_type == true_type:
                q_correct += 1

        # affect preservation: emotional statements should stay statements
        if true_type == "statement" and true_affect == "emotional":
            emo_stmt_total += 1
            if pred_type == "statement":
                emo_stmt_kept += 1

        # brow_affective detection vs emotional ground truth
        is_emo = true_affect == "emotional"
        if brow_affective and is_emo:       ba_tp += 1
        elif brow_affective and not is_emo: ba_fp += 1
        elif not brow_affective and is_emo: ba_fn += 1
        else:                                ba_tn += 1

    def _safe(n, d):
        return (n / d) if d else float("nan")

    metrics = {
        "n": len(rows),
        "type_accuracy": _safe(
            sum(confusion[t][t] for t in confusion), len(rows)
        ),
        "false_question_rate": _safe(stmt_as_question, stmt_total),
        "false_question_count": (stmt_as_question, stmt_total),
        "question_recall": _safe(q_correct, q_total),
        "question_recall_count": (q_correct, q_total),
        "affect_preservation": _safe(emo_stmt_kept, emo_stmt_total),
        "affect_preservation_count": (emo_stmt_kept, emo_stmt_total),
        "brow_affective_precision": _safe(ba_tp, ba_tp + ba_fp),
        "brow_affective_recall": _safe(ba_tp, ba_tp + ba_fn),
    }
    return {"per_row": per_row, "confusion": confusion, "metrics": metrics}


def print_report(result: dict) -> None:
    m = result["metrics"]
    conf = result["confusion"]

    print("\n" + "=" * 60)
    print("Signet Aid - NMM / sentence-type evaluation")
    print("=" * 60)
    print(f"clips evaluated: {m['n']}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    header = "true \\ pred".ljust(14) + "".join(t.rjust(11) for t in TYPES)
    print(header)
    for t in TYPES:
        row = t.ljust(14) + "".join(str(conf[t][p]).rjust(11) for p in TYPES)
        print(row)

    def pct(x):
        return "n/a" if x != x else f"{x*100:5.1f}%"   # x!=x → NaN

    print("\nHeadline metrics:")
    print(f"  sentence-type accuracy   : {pct(m['type_accuracy'])}")
    fq_n, fq_d = m["false_question_count"]
    print(f"  FALSE-QUESTION RATE      : {pct(m['false_question_rate'])}  "
          f"({fq_n}/{fq_d} true statements turned into questions)  <-- core safety metric")
    qr_n, qr_d = m["question_recall_count"]
    print(f"  question recall          : {pct(m['question_recall'])}  ({qr_n}/{qr_d})")
    ap_n, ap_d = m["affect_preservation_count"]
    print(f"  affect preservation      : {pct(m['affect_preservation'])}  "
          f"({ap_n}/{ap_d} emotional statements kept as statements)")
    print(f"  brow_affective precision : {pct(m['brow_affective_precision'])}")
    print(f"  brow_affective recall    : {pct(m['brow_affective_recall'])}")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Front-end 1: features CSV (no models needed)
# ─────────────────────────────────────────────────────────────────────────────
def load_features_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"text", "true_type", "true_affect"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        return list(reader)


# ─────────────────────────────────────────────────────────────────────────────
# Front-end 2: video manifest (runs the real NMMClassifier)
# ─────────────────────────────────────────────────────────────────────────────
def features_from_video(manifest_path: str,
                        calib_frames: int,
                        active_fraction: float) -> list:
    """
    Run NMMClassifier over each clip to derive per-clip flags.
    A flag is considered active for the clip if it fires in >= active_fraction
    of the evaluated (post-calibration) frames.
    """
    import cv2
    from processes.nmm_classifier import NMMClassifier

    with open(manifest_path, newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    rows = []
    for entry in manifest:
        clip = entry["clip_path"]
        cap = cv2.VideoCapture(clip)
        if not cap.isOpened():
            log.warning(f"Cannot open clip, skipping: {clip}")
            continue

        clf = NMMClassifier()          # fresh calibration per clip
        counts = Counter()
        evaluated = 0
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ctx = clf.classify(frame)
            frame_idx += 1
            if frame_idx <= calib_frames:
                continue               # let the per-session baseline settle
            evaluated += 1
            if ctx.is_yn_question:  counts["yn"] += 1
            if ctx.is_wh_question:  counts["wh"] += 1
            if ctx.is_negation:     counts["neg"] += 1
            if ctx.brow_affective:  counts["affect"] += 1
        cap.release()
        clf.close()

        if evaluated == 0:
            log.warning(f"Clip too short for calib_frames={calib_frames}: {clip}")

        def active(key):
            return 1 if (evaluated and counts[key] / evaluated >= active_fraction) else 0

        rows.append({
            "text": entry["text"],
            "true_type": entry["true_type"],
            "true_affect": entry.get("true_affect", ""),
            "yn_face": active("yn"),
            "wh_face": active("wh"),
            "neg_face": active("neg"),
            "brow_affective": active("affect"),
        })
        log.info(f"{clip}: evaluated {evaluated} frames -> "
                 f"yn={active('yn')} wh={active('wh')} neg={active('neg')} "
                 f"affect={active('affect')}")
    return rows


def write_results_csv(per_row: list, path: str) -> None:
    if not per_row:
        return
    cols = ["text", "true_type", "true_affect", "yn_face", "wh_face",
            "neg_face", "brow_affective", "pred_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in per_row:
            w.writerow(r)
    print(f"Per-clip results written to {path}")


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Signet Aid NMM evaluation harness")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-features", metavar="CSV",
                     help="Pre-extracted per-clip features (no models needed).")
    src.add_argument("--from-video", metavar="MANIFEST_CSV",
                     help="Run NMMClassifier over clips listed in the manifest.")
    ap.add_argument("--calib-frames", type=int, default=15,
                    help="Frames to skip per clip for baseline calibration (video mode).")
    ap.add_argument("--active-fraction", type=float, default=0.30,
                    help="Fraction of frames a flag must fire to count as active (video mode).")
    ap.add_argument("--out", metavar="CSV", help="Write per-clip predictions here.")
    args = ap.parse_args(argv)

    if args.from_features:
        rows = load_features_csv(args.from_features)
    else:
        rows = features_from_video(args.from_video, args.calib_frames, args.active_fraction)

    if not rows:
        print("No rows to evaluate.")
        return 1

    result = evaluate(rows)
    print_report(result)
    if args.out:
        write_results_csv(result["per_row"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
