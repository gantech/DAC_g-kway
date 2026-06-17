#!/usr/bin/env python3

import argparse
import csv
import pathlib
import sys
from collections import defaultdict


def fail(message: str):
    print(f"FAIL: {message}")
    sys.exit(1)


def parse_graph_weights(graph_path: pathlib.Path):
    with graph_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("%")] 

    if not lines:
        fail("graph file is empty")

    header = lines[0].split()
    if len(header) < 2:
        fail("graph header must contain at least <num_vertices> <num_edges>")

    try:
        num_vertices = int(header[0])
    except ValueError:
        fail("graph header has non-integer num_vertices")

    if num_vertices <= 0:
        fail("graph header num_vertices must be positive")

    if len(lines) < num_vertices + 1:
        fail(f"graph body has {len(lines) - 1} rows, expected {num_vertices}")

    weights = [0] * num_vertices
    for vertex_idx in range(num_vertices):
        tokens = lines[vertex_idx + 1].split()
        if not tokens:
            fail(f"graph row {vertex_idx + 2} is empty")

        try:
            weights[vertex_idx] = int(tokens[0])
        except ValueError:
            fail(f"graph row {vertex_idx + 2}: first token must be integer vertex weight")

    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Check that coarse grouping in .levels conserves fine vertex weights"
    )
    parser.add_argument("--graph", required=True, help="Path to METIS-like graph input")
    parser.add_argument("--levels", required=True, help="Path to *.levels CSV")
    args = parser.parse_args()

    graph_path = pathlib.Path(args.graph)
    levels_path = pathlib.Path(args.levels)
    if not graph_path.exists():
        fail(f"graph file does not exist: {graph_path}")
    if not levels_path.exists():
        fail(f"levels file does not exist: {levels_path}")

    fine_weights = parse_graph_weights(graph_path)
    num_vertices = len(fine_weights)
    expected_num_coarse_vertices = (num_vertices + 1) // 2

    with levels_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        fail("levels file must include header and at least one data row")

    header = rows[0]
    if len(header) < 3 or header[0] != "PartitionID" or header[-1] != "L0":
        fail(f"expected header to start with PartitionID and end with L0, found {header}")

    seen_l0 = set()
    coarse_weight = defaultdict(int)

    for idx, row in enumerate(rows[1:], start=2):
        if len(row) < 3:
            fail(f"row {idx}: expected at least 3 columns, found {len(row)}")

        try:
            l1 = int(row[-2])
            l0 = int(row[-1])
        except ValueError:
            fail(f"row {idx}: L1 and L0 must be integers")

        if l0 <= 0 or l0 > num_vertices:
            fail(f"row {idx}: L0 out of graph vertex range: {l0}")

        if l1 <= 0:
            fail(f"row {idx}: L1 must be positive")

        if l0 in seen_l0:
            fail(f"row {idx}: duplicate L0 found: {l0}")
        seen_l0.add(l0)

        expected_l1 = ((l0 - 1) // 2) + 1
        if l1 != expected_l1:
            fail(f"row {idx}: expected L1={expected_l1} for L0={l0}, found {l1}")

        coarse_weight[l1] += fine_weights[l0 - 1]

    if len(seen_l0) != num_vertices:
        fail(f"levels contain {len(seen_l0)} unique L0 values, expected {num_vertices}")

    if len(coarse_weight) != expected_num_coarse_vertices:
        fail(
            f"coarse vertex count mismatch: got {len(coarse_weight)}, "
            f"expected {expected_num_coarse_vertices}"
        )

    total_fine_weight = sum(fine_weights)
    total_coarse_weight = sum(coarse_weight.values())
    if total_coarse_weight != total_fine_weight:
        fail(
            f"weight not conserved: fine={total_fine_weight}, coarse={total_coarse_weight}"
        )

    print(
        f"PASS: fine_vertices={num_vertices} coarse_vertices={len(coarse_weight)} "
        f"fine_total_weight={total_fine_weight} coarse_total_weight={total_coarse_weight}"
    )


if __name__ == "__main__":
    main()