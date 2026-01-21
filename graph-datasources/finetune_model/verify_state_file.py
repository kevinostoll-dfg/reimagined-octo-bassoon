"""
Comprehensive verifier for PKL state files produced by llm_classify.py.
Shows counts, distribution, text stats, recency, and samples.
"""

import argparse
import os
import pickle
import statistics
from collections import Counter
from datetime import datetime
from typing import Dict, Any, Iterable, Tuple, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect llm_classify PKL state.")
    parser.add_argument(
        "--path",
        type=str,
        default="train_state.pkl",
        help="Path to PKL state file (default: train_state.pkl)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=5,
        help="Number of sample entries to display for each category",
    )
    return parser.parse_args()


def load_state(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        raise SystemExit(f"Missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def pct(part: int, total: int) -> float:
    return (part / total * 100) if total else 0.0


def text_stats(entries: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    lengths = [len(e.get("text", "") or "") for e in entries]
    if not lengths:
        return {}
    lengths_sorted = sorted(lengths)
    return {
        "min": lengths_sorted[0],
        "max": lengths_sorted[-1],
        "mean": statistics.mean(lengths_sorted),
        "median": statistics.median(lengths_sorted),
        "p95": lengths_sorted[int(0.95 * (len(lengths_sorted) - 1))],
    }


def updated_at_stats(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    timestamps: List[datetime] = []
    missing = 0
    for e in entries:
        ts = e.get("updated_at")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts))
            except Exception:
                missing += 1
        else:
            missing += 1
    if not timestamps:
        return {"count": 0, "missing": missing}
    ts_sorted = sorted(timestamps)
    return {
        "count": len(timestamps),
        "missing": missing,
        "earliest": ts_sorted[0].isoformat(),
        "latest": ts_sorted[-1].isoformat(),
        "median": ts_sorted[len(ts_sorted) // 2].isoformat(),
    }


def sample_entries(items: Iterable[Tuple[str, Dict[str, Any]]], n: int) -> List[Tuple[str, Dict[str, Any]]]:
    sampled = []
    for idx, item in enumerate(items):
        if idx >= n:
            break
        sampled.append(item)
    return sampled


def fmt_entry(entry_id: str, entry: Dict[str, Any]) -> str:
    return f"{entry_id} | label={entry.get('label')} processed={entry.get('processed')} text[:80]={entry.get('text','')[:80]}"


def main():
    args = parse_args()
    state = load_state(args.path)

    total = len(state)
    processed = sum(1 for v in state.values() if v.get("processed") and isinstance(v.get("label"), int))
    pending = total - processed
    errors = sum(1 for v in state.values() if v.get("error"))

    label_counts = Counter(v.get("label") for v in state.values() if isinstance(v.get("label"), int))
    label_dist = {lbl: f"{cnt} ({pct(cnt, processed):.2f}%)" for lbl, cnt in sorted(label_counts.items())}

    processed_entries = (v for v in state.values() if v.get("processed") and isinstance(v.get("label"), int))
    pending_entries = (v for v in state.values() if not (v.get("processed") and isinstance(v.get("label"), int)))

    text_stats_all = text_stats(state.values())
    text_stats_processed = text_stats(processed_entries)
    text_stats_pending = text_stats(pending_entries)

    updated_stats_all = updated_at_stats(state.values())
    updated_stats_processed = updated_at_stats(v for v in state.values() if v.get("processed"))

    print(f"file: {args.path}")
    print(f"entries: {total}")
    print(f"processed: {processed} ({pct(processed, total):.2f}%)")
    print(f"pending: {pending} ({pct(pending, total):.2f}%)")
    print(f"errors: {errors}")
    print(f"label_counts: {dict(label_counts)}")
    print(f"label_distribution: {label_dist}")
    print()
    print("text_length stats (all):", text_stats_all)
    print("text_length stats (processed):", text_stats_processed)
    print("text_length stats (pending):", text_stats_pending)
    print()
    print("updated_at stats (all):", updated_stats_all)
    print("updated_at stats (processed):", updated_stats_processed)
    print()

    # Samples
    processed_samples = sample_entries(
        ((k, v) for k, v in state.items() if v.get("processed") and isinstance(v.get("label"), int)),
        args.sample,
    )
    pending_samples = sample_entries(
        ((k, v) for k, v in state.items() if not (v.get("processed") and isinstance(v.get("label"), int))),
        args.sample,
    )
    error_samples = sample_entries(((k, v) for k, v in state.items() if v.get("error")), args.sample)

    print(f"sample processed ({len(processed_samples)}):")
    for k, v in processed_samples:
        print("  ", fmt_entry(k, v))
    print(f"\nsample pending ({len(pending_samples)}):")
    for k, v in pending_samples:
        print("  ", fmt_entry(k, v))
    print(f"\nsample with errors ({len(error_samples)}):")
    for k, v in error_samples:
        print("  ", fmt_entry(k, v))


if __name__ == "__main__":
    main()