#!/usr/bin/env python3

import argparse
import csv
import pathlib
import sys
from collections import defaultdict


def fail(message: str):
    print(f"FAIL: {message}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Validate multilevel consistency in out.levels")
    parser.add_argument("--levels", required=True, help="Path to *.levels CSV")
    args = parser.parse_args()

    levels_path = pathlib.Path(args.levels)
    if not levels_path.exists():
        fail(f"levels file does not exist: {levels_path}")

    with levels_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        fail("levels file must include header and data rows")

    header = rows[0]
    if header != ["PartitionID", "L1", "L0"]:
        fail(f"expected header ['PartitionID', 'L1', 'L0'], found {header}")

    coarse_to_partition = {}
    coarse_to_fine = defaultdict(list)

    for idx, row in enumerate(rows[1:], start=2):
        if len(row) != 3:
            fail(f"row {idx}: expected 3 columns, found {len(row)}")

        try:
            partition_id = int(row[0])
            level1_id = int(row[1])
            level0_id = int(row[2])
        except ValueError:
            fail(f"row {idx}: all values must be integers")

        if level1_id <= 0:
            fail(f"row {idx}: L1 must be positive")
        if level0_id <= 0:
            fail(f"row {idx}: L0 must be positive")

        if level1_id not in coarse_to_partition:
            coarse_to_partition[level1_id] = partition_id
        elif coarse_to_partition[level1_id] != partition_id:
            fail(
                f"row {idx}: coarse vertex L1={level1_id} maps to multiple partitions "
                f"({coarse_to_partition[level1_id]} and {partition_id})"
            )

        coarse_to_fine[level1_id].append(level0_id)

    max_group_size = 0
    for level1_id, fine_vertices in coarse_to_fine.items():
        fine_vertices_sorted = sorted(fine_vertices)
        group_size = len(fine_vertices_sorted)
        max_group_size = max(max_group_size, group_size)

        if group_size > 2:
            fail(f"L1={level1_id}: expected at most 2 fine vertices, found {group_size}")

        if group_size == 2 and fine_vertices_sorted[1] - fine_vertices_sorted[0] != 1:
            fail(
                f"L1={level1_id}: expected pairwise consecutive L0 ids, found {fine_vertices_sorted}"
            )

    print(
        f"PASS: coarse_vertices={len(coarse_to_fine)} max_group_size={max_group_size} "
        f"partition_consistency=ok"
    )


if __name__ == "__main__":
    main()