#!/usr/bin/env python3
"""
Process nvprof CSV files and generate summary files with TFLOP counts.

This script:
1. Scans a directory for nvprof CSV files (nvprof_*.csv)
2. Extracts solver, preconditioner, size, and precision from each file
3. Sums all FLOP operations from the "Avg" column
4. Reads iteration count from times.dat in corresponding folders
5. Divides total FLOP by iteration count to get per-iteration TFLOP
6. Generates profile_summary_float.csv and profile_summary_double.csv
"""

import csv
import os
import re
import sys
from pathlib import Path
from collections import defaultdict


def parse_filename(filename):
    """
    Parse filename like 'nvprof_gmres_diag_double.csv' or 'nvprof_jacobi_mg_float.csv'
    Returns: (solver, prec, precision) or None if pattern doesn't match
    """
    # Pattern: nvprof_<solver>_<prec>_<float|double>.csv
    match = re.match(r'nvprof_(\w+)_(\w+)_(float|double)\.csv', filename)
    if match:
        return match.groups()  # (solver, prec, precision)
    return None


def extract_size_from_csv(csv_path):
    """
    Extract grid size from the command line in the CSV file.
    Looks for pattern like: ./test_biharmonic_cuda_d.bin gmres diag 256 ...
    Returns size as integer or None if not found.
    """
    try:
        with open(csv_path, 'r') as f:
            # Read first few lines to find the command
            for i, line in enumerate(f):
                if i > 5:  # Don't read too many lines
                    break
                # Look for command line pattern
                match = re.search(r'\.bin\s+\w+\s+\w+\s+(\d+)', line)
                if match:
                    return int(match.group(1))
    except Exception as e:
        print(f"Warning: Could not extract size from {csv_path}: {e}", file=sys.stderr)
    return None


def find_matching_folder(directory, solver, prec, precision, size):
    """
    Find the folder matching the pattern: <solver>_<prec>_cuda_<precision>_<size>_*
    Returns Path to the folder or None if not found.
    """
    directory = Path(directory)
    # Pattern: <solver>_<prec>_cuda_<float|double>_<size>_*
    pattern = f"{solver}_{prec}_cuda_{precision}_{size}_*"

    matching_folders = list(directory.glob(pattern))
    if not matching_folders:
        return None
    if len(matching_folders) > 1:
        print(f"Warning: Multiple folders match pattern {pattern}, using first: {matching_folders[0]}", file=sys.stderr)
    return matching_folders[0]


def read_iterations_from_times_dat(folder_path):
    """
    Read the number of iterations from times.dat file in the given folder.
    Returns iters_n as integer or None if not found.
    """
    times_dat_path = Path(folder_path) / 'times.dat'
    if not times_dat_path.exists():
        return None

    try:
        with open(times_dat_path, 'r') as f:
            reader = csv.reader(f)
            # Read header
            header = next(reader)
            if 'iters_n' not in header:
                print(f"Warning: 'iters_n' column not found in {times_dat_path}", file=sys.stderr)
                return None

            iters_n_idx = header.index('iters_n')

            # Read data row
            data_row = next(reader)
            if len(data_row) > iters_n_idx:
                iters_n = int(float(data_row[iters_n_idx]))
                return iters_n
    except Exception as e:
        print(f"Warning: Could not read iterations from {times_dat_path}: {e}", file=sys.stderr)

    return None


def process_csv_file(csv_path, directory):
    """
    Process a single nvprof CSV file and return summary data.
    Returns: (solver, prec, size, precision, total_flop, iters_n) or None if error
    """
    filename = os.path.basename(csv_path)
    parsed = parse_filename(filename)

    if not parsed:
        print(f"Warning: Could not parse filename {filename}, skipping", file=sys.stderr)
        return None

    solver, prec, precision = parsed

    # Extract size from CSV content
    size = extract_size_from_csv(csv_path)
    if size is None:
        print(f"Warning: Could not extract size from {filename}, skipping", file=sys.stderr)
        return None

    # Find matching folder and read iterations
    matching_folder = find_matching_folder(directory, solver, prec, precision, size)
    iters_n = None
    if matching_folder:
        iters_n = read_iterations_from_times_dat(matching_folder)
        if iters_n is None:
            print(f"Warning: Could not read iterations from {matching_folder}/times.dat, using 1", file=sys.stderr)
            iters_n = 1
    else:
        print(f"Warning: Could not find matching folder for {filename}, using 1 iteration", file=sys.stderr)
        iters_n = 1

    # Read CSV and sum FLOPs from "Avg" column multiplied by "Invocations"
    total_flop = 0
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            # Skip header lines (lines starting with == or empty)
            header_found = False
            invocations_idx = None
            avg_idx = None
            for row in reader:
                if not row:
                    continue

                # Look for the actual header row
                if not header_found and len(row) > 0 and 'Avg' in row:
                    header_found = True
                    # Find the index of "Avg" and "Invocations" columns
                    try:
                        avg_idx = row.index('Avg')
                        invocations_idx = row.index('Invocations')
                    except ValueError as e:
                        print(f"Warning: Could not find required column in {filename}: {e}", file=sys.stderr)
                        return None
                    continue

                # Process data rows (skip header rows and empty rows)
                if header_found and len(row) > max(avg_idx, invocations_idx):
                    try:
                        # Get Invocations and Avg values
                        invocations_str = row[invocations_idx].strip()
                        avg_value_str = row[avg_idx].strip()

                        # Check if both are numeric
                        if (invocations_str and invocations_str.replace('.', '').replace('-', '').isdigit() and
                            avg_value_str and avg_value_str.replace('.', '').replace('-', '').isdigit()):
                            invocations = int(float(invocations_str))  # Convert to int (invocations should be integer)
                            avg_value = float(avg_value_str)
                            # Total FLOP = Avg FLOP per invocation * number of invocations
                            total_flop += avg_value * invocations
                    except (ValueError, IndexError) as e:
                        # Skip rows that can't be parsed
                        continue

    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        return None

    return (solver, prec, size, precision, total_flop, iters_n)


def process_directory(directory):
    """
    Process all nvprof CSV files in the given directory.
    Returns: dict with keys 'float' and 'double', each containing list of tuples (solver, prec, size, tflop)
    """
    results = defaultdict(list)
    directory = Path(directory)

    # Find all nvprof CSV files
    csv_files = list(directory.glob('nvprof_*.csv'))

    if not csv_files:
        print(f"Warning: No nvprof CSV files found in {directory}", file=sys.stderr)
        return results

    print(f"Found {len(csv_files)} CSV files to process...", file=sys.stderr)

    for csv_file in csv_files:
        data = process_csv_file(csv_file, directory)
        if data:
            solver, prec, size, precision, total_flop, iters_n = data
            # Divide by number of iterations to get per-iteration FLOP
            flop_per_iter = total_flop / iters_n
            tflop = flop_per_iter / 1e12  # Convert to TFLOP
            results[precision].append((solver, prec, size, tflop))
            print(f"  Processed {csv_file.name}: {solver}, {prec}, size={size}, {precision}, {iters_n} iters, {tflop:.6f} TFLOP/iter", file=sys.stderr)

    return results


def write_summary_file(results, precision, output_path):
    """
    Write summary CSV file for a given precision.
    """
    if precision not in results or not results[precision]:
        print(f"Warning: No {precision} precision results to write", file=sys.stderr)
        return

    # Sort by solver, then prec, then size
    sorted_results = sorted(results[precision], key=lambda x: (x[0], x[1], x[2]))

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['solver', 'prec', 'size', 'TFLOP'])

        # Write data
        for solver, prec, size, tflop in sorted_results:
            writer.writerow([solver, prec, size, f'{tflop:.6f}'])

    print(f"Written {len(sorted_results)} entries to {output_path}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print("Usage: process_profiles.py <directory>", file=sys.stderr)
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Process all CSV files
    results = process_directory(directory)

    # Write summary files
    output_dir = Path(directory)
    write_summary_file(results, 'float', output_dir / 'profile_summary_float.csv')
    write_summary_file(results, 'double', output_dir / 'profile_summary_double.csv')

    print(f"\nSummary:", file=sys.stderr)
    print(f"  Float entries: {len(results.get('float', []))}", file=sys.stderr)
    print(f"  Double entries: {len(results.get('double', []))}", file=sys.stderr)


if __name__ == '__main__':
    main()
