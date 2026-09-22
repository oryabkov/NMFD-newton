#!/usr/bin/env python3
"""
Script to check Cahn-Hilliard solution files for corruption.

Usage:
    python check_files.py <folder> [--num-files N] [--from-end]

Example:
    # Check last 100 files from the end
    python check_files.py data/ch_128_100k_5e5_20260126_112055 --num-files 100 --from-end

    # Check all files
    python check_files.py data/ch_128_100k_5e5_20260126_112055
"""

import numpy as np
import argparse
import glob
from pathlib import Path


def load_solution(filename):
    """Load solution from binary file saved by C++ code."""
    with open(filename, 'rb') as f:
        dims = np.frombuffer(f.read(12), dtype=np.int32)
        n_components = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(), dtype=np.float64)
        data = data.reshape((dims[2], dims[1], dims[0], n_components))
        # Transpose to (N, N, N, 2) with x as first axis
        data = np.transpose(data, (2, 1, 0, 3))
    return data, dims[0]


def check_file(filename):
    """Try to load a file and return (success, error_message)."""
    try:
        data, N = load_solution(filename)
        # Basic validation
        if data.size == 0:
            return False, "File is empty (data.size == 0)"
        if N <= 0:
            return False, f"Invalid grid size: N = {N}"
        expected_size = N * N * N * 2  # Assuming 2 components
        if data.size != expected_size:
            return False, f"Size mismatch: expected {expected_size}, got {data.size}"
        return True, None
    except FileNotFoundError:
        return False, "File not found"
    except IOError as e:
        return False, f"IO error: {e}"
    except ValueError as e:
        return False, f"Value error: {e}"
    except IndexError as e:
        return False, f"Index error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description='Check Cahn-Hilliard solution files for corruption',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check last 100 files from the end
  python check_files.py data/ch_128_100k_5e5_20260126_112055 --num-files 100 --from-end

  # Check all files
  python check_files.py data/ch_128_100k_5e5_20260126_112055

  # Check first 50 files from the beginning
  python check_files.py data/ch_128_100k_5e5_20260126_112055 --num-files 50
        """
    )

    parser.add_argument('folder', type=str,
                       help='Path to folder containing numerical_*.bin files')
    parser.add_argument('--num-files', type=int, default=None,
                       help='Number of files to check. If not specified, checks all files.')
    parser.add_argument('--from-end', action='store_true',
                       help='If specified, check files from the end (last N files). '
                            'Otherwise, check from the beginning (first N files).')

    args = parser.parse_args()

    folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return 1

    # Find all numerical_*.bin files
    numerical_files = sorted(glob.glob(str(folder_path / 'numerical_*.bin')),
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if len(numerical_files) == 0:
        print(f"Error: No numerical_*.bin files found in {folder_path}")
        return 1

    print(f"Found {len(numerical_files)} files in {folder_path}")

    # Determine which files to check
    if args.num_files is not None:
        if args.from_end:
            files_to_check = numerical_files[-args.num_files:]
            print(f"Checking last {len(files_to_check)} files (from end)...")
        else:
            files_to_check = numerical_files[:args.num_files]
            print(f"Checking first {len(files_to_check)} files (from beginning)...")
    else:
        files_to_check = numerical_files
        print(f"Checking all {len(files_to_check)} files...")

    print("-" * 80)

    broken_files = []
    checked_count = 0

    for filename in files_to_check:
        checked_count += 1
        file_path = Path(filename)
        file_name = file_path.name

        success, error_msg = check_file(filename)

        if success:
            # Extract file info for display
            try:
                data, N = load_solution(filename)
                status = f"OK (N={N}, shape={data.shape})"
            except:
                status = "OK"
            print(f"[{checked_count}/{len(files_to_check)}] {file_name}: {status}")
        else:
            broken_files.append((file_name, error_msg))
            print(f"[{checked_count}/{len(files_to_check)}] {file_name}: BROKEN - {error_msg}")

    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Total files checked: {checked_count}")
    print(f"  OK: {checked_count - len(broken_files)}")
    print(f"  Broken: {len(broken_files)}")

    if broken_files:
        print(f"\nBroken files:")
        for file_name, error_msg in broken_files:
            print(f"  - {file_name}: {error_msg}")
        return 1
    else:
        print("\nAll checked files are OK!")
        return 0


if __name__ == '__main__':
    exit(main())
