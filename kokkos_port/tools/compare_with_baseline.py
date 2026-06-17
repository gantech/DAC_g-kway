#!/usr/bin/env python3

import argparse
import csv
import pathlib
import sys


def read_lines(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def compare_out_file(baseline_path: pathlib.Path, candidate_path: pathlib.Path):
    base = read_lines(baseline_path)
    cand = read_lines(candidate_path)
    if len(base) != len(cand):
        return False, f"out.out line mismatch: baseline={len(base)} candidate={len(cand)}"

    diffs = 0
    for b, c in zip(base, cand):
        if b != c:
            diffs += 1

    return True, f"out.out compared: lines={len(base)} differing_lines={diffs}"


def compare_levels_file(baseline_path: pathlib.Path, candidate_path: pathlib.Path):
    with baseline_path.open("r", encoding="utf-8") as bh, candidate_path.open("r", encoding="utf-8") as ch:
        b_reader = list(csv.reader(bh))
        c_reader = list(csv.reader(ch))

    if not b_reader or not c_reader:
        return False, "levels file empty"

    b_header = b_reader[0]
    c_header = c_reader[0]
    if b_header != c_header:
        return False, f"levels header mismatch: baseline={b_header} candidate={c_header}"

    if len(b_reader) != len(c_reader):
        return False, f"levels row mismatch: baseline={len(b_reader)} candidate={len(c_reader)}"

    diffs = 0
    for b_row, c_row in zip(b_reader[1:], c_reader[1:]):
        if b_row != c_row:
            diffs += 1

    return True, f"out.levels compared: rows={len(b_reader)-1} differing_rows={diffs}"


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and candidate partition artifacts")
    parser.add_argument("--baseline-prefix", required=True, help="Path prefix for baseline files")
    parser.add_argument("--candidate-prefix", required=True, help="Path prefix for candidate files")
    args = parser.parse_args()

    baseline_prefix = pathlib.Path(args.baseline_prefix)
    candidate_prefix = pathlib.Path(args.candidate_prefix)

    pairs = [
        (baseline_prefix.with_suffix(".out"), candidate_prefix.with_suffix(".out"), compare_out_file),
        (baseline_prefix.with_suffix(".levels"), candidate_prefix.with_suffix(".levels"), compare_levels_file),
    ]

    ok = True
    for b_path, c_path, comparator in pairs:
        if not b_path.exists() or not c_path.exists():
            print(f"MISSING: baseline={b_path} candidate={c_path}")
            ok = False
            continue
        status, message = comparator(b_path, c_path)
        print(message)
        ok = ok and status

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
