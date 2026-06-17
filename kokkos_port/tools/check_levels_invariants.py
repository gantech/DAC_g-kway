#!/usr/bin/env python3

import argparse
import csv
import pathlib
import sys


def fail(message: str):
    print(f"FAIL: {message}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Validate invariants for out.levels CSV")
    parser.add_argument("--levels", required=True, help="Path to *.levels CSV")
    parser.add_argument("--num-partitions", type=int, default=None)
    args = parser.parse_args()

    path = pathlib.Path(args.levels)
    if not path.exists():
        fail(f"levels file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        fail("levels file must include header and at least one data row")

    header = rows[0]
    if not header or header[0] != "PartitionID":
        fail("header must start with PartitionID")
    if header[-1] != "L0":
        fail("header must end with L0")

    expected_cols = len(header)
    seen_l0 = set()

    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != expected_cols:
            fail(f"row {idx}: expected {expected_cols} columns, found {len(row)}")

        try:
            values = [int(val) for val in row]
        except ValueError:
            fail(f"row {idx}: non-integer value found")

        partition = values[0]
        if args.num_partitions is not None and (partition < 0 or partition >= args.num_partitions):
            fail(f"row {idx}: PartitionID {partition} out of [0, {args.num_partitions - 1}]")

        l0 = values[-1]
        if l0 <= 0:
            fail(f"row {idx}: L0 must be positive")
        if l0 in seen_l0:
            fail(f"row {idx}: duplicate L0 id {l0}")
        seen_l0.add(l0)

    print(f"PASS: rows={len(rows)-1} columns={expected_cols}")


if __name__ == "__main__":
    main()
