#!/usr/bin/env python3
"""
Combine times.dat files from performance calculations into separate CSV files for float and double.

Usage: python combine_times.py <input_folder> [output_prefix]
  input_folder: Path to folder containing performance calculations
  output_prefix: Prefix for output CSV files (default: combined_times)
                 Outputs: {prefix}_float.csv and {prefix}_double.csv
"""

import sys
import os
import csv
from pathlib import Path
from collections import defaultdict

def find_times_dat_files(root_dir):
    """Find all times.dat files recursively."""
    root_path = Path(root_dir)
    times_files = list(root_path.rglob("times.dat"))
    return times_files

def extract_omp_threads(file_path, root_dir):
    """Extract OMP thread count from file path."""
    # For OMP, path structure is: root/omp/{threads}/.../times.dat
    # For CPU/CUDA, path structure is: root/{arch}/.../times.dat
    rel_path = file_path.relative_to(root_dir)
    parts = rel_path.parts

    if len(parts) >= 2 and parts[0] == "omp":
        # Extract thread count from omp/{threads}/...
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None

def parse_times_dat(file_path):
    """Parse a times.dat file and return the data row."""
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return None

def combine_times(input_folder, output_prefix="combined_times"):
    """Combine all times.dat files into separate CSV files for float and double."""
    input_path = Path(input_folder)

    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist", file=sys.stderr)
        return 1

    # Find all times.dat files
    times_files = find_times_dat_files(input_path)

    if not times_files:
        print(f"Error: No times.dat files found in '{input_folder}'", file=sys.stderr)
        return 1

    print(f"Found {len(times_files)} times.dat files")

    # Dictionary to store data: key = (arch, cores, size), value = {solver_prec: {...}}
    # Separate dictionaries for float and double
    float_dict = defaultdict(dict)
    double_dict = defaultdict(dict)

    # Process each times.dat file
    for times_file in times_files:
        row = parse_times_dat(times_file)
        if row is None:
            continue

        solver = row.get("solver", "")
        prec = row.get("prec", "")
        arch = row.get("arch", "")
        float_type = row.get("float_type", "")
        size = row.get("size", "")

        # Extract OMP thread count
        cores = extract_omp_threads(times_file, input_path)
        if cores is None:
            # For CPU/CUDA, use 1 for consistency
            cores = 1

        # Create key: (arch, cores, size)
        key = (arch, cores, size)

        # Create solver_prec identifier
        solver_prec = f"{solver}_{prec}"

        # Store data in appropriate dictionary
        if float_type == "f":
            float_dict[key][solver_prec] = row
        elif float_type == "d":
            double_dict[key][solver_prec] = row

    # Define output columns
    base_columns = ["arch", "cores", "size"]
    solver_prec_combinations = ["jacobi_diag", "gmres_diag", "jacobi_mg", "gmres_mg"]
    metrics = ["iters_n", "time", "reduction_rate"]

    output_columns = base_columns.copy()
    for solver_prec in solver_prec_combinations:
        for metric in metrics:
            output_columns.append(f"{solver_prec}({metric})")

    # Write float CSV
    float_output = Path(f"{output_prefix}_float.csv")
    with open(float_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()

        # Sort keys for consistent output
        sorted_keys = sorted(float_dict.keys())

        for key in sorted_keys:
            arch, cores, size = key
            solver_prec_data = float_dict[key]

            output_row = {
                "arch": arch,
                "cores": cores,
                "size": size,
            }

            # Fill in data for each solver_prec combination
            for solver_prec in solver_prec_combinations:
                if solver_prec in solver_prec_data:
                    data = solver_prec_data[solver_prec]
                    output_row[f"{solver_prec}(iters_n)"] = data.get("iters_n", "")
                    output_row[f"{solver_prec}(time)"] = data.get("time(ms)", "")
                    output_row[f"{solver_prec}(reduction_rate)"] = data.get("reduction_rate", "")
                else:
                    output_row[f"{solver_prec}(iters_n)"] = ""
                    output_row[f"{solver_prec}(time)"] = ""
                    output_row[f"{solver_prec}(reduction_rate)"] = ""

            writer.writerow(output_row)

    print(f"Float: Combined {len(sorted_keys)} unique configurations")
    print(f"Float output written to: {float_output}")

    # Write double CSV
    double_output = Path(f"{output_prefix}_double.csv")
    with open(double_output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()

        # Sort keys for consistent output
        sorted_keys = sorted(double_dict.keys())

        for key in sorted_keys:
            arch, cores, size = key
            solver_prec_data = double_dict[key]

            output_row = {
                "arch": arch,
                "cores": cores,
                "size": size,
            }

            # Fill in data for each solver_prec combination
            for solver_prec in solver_prec_combinations:
                if solver_prec in solver_prec_data:
                    data = solver_prec_data[solver_prec]
                    output_row[f"{solver_prec}(iters_n)"] = data.get("iters_n", "")
                    output_row[f"{solver_prec}(time)"] = data.get("time(ms)", "")
                    output_row[f"{solver_prec}(reduction_rate)"] = data.get("reduction_rate", "")
                else:
                    output_row[f"{solver_prec}(iters_n)"] = ""
                    output_row[f"{solver_prec}(time)"] = ""
                    output_row[f"{solver_prec}(reduction_rate)"] = ""

            writer.writerow(output_row)

    print(f"Double: Combined {len(sorted_keys)} unique configurations")
    print(f"Double output written to: {double_output}")

    return 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python combine_times.py <input_folder> [output_prefix]", file=sys.stderr)
        print("  input_folder: Path to folder containing performance calculations", file=sys.stderr)
        print("  output_prefix: Prefix for output CSV files (default: combined_times)", file=sys.stderr)
        print("                 Outputs: {prefix}_float.csv and {prefix}_double.csv", file=sys.stderr)
        return 1

    input_folder = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "combined_times"

    return combine_times(input_folder, output_prefix)

if __name__ == "__main__":
    sys.exit(main())
