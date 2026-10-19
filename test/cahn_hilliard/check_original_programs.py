"""Run the original CH programs without changing their numerical settings."""

import argparse
import math
from pathlib import Path
import re
import shlex
import subprocess


def run(binary, arguments, directory, launcher=()):
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [*launcher, str(binary), *arguments],
        cwd=directory, capture_output=True, text=True, timeout=120,
    )
    output = result.stdout + result.stderr
    (directory / "output.log").write_text(output)
    if result.returncode or re.search(r"ERROR|(?i:\b(?:nan|inf(?:inity)?)\b)", output):
        raise RuntimeError(f"{binary.name} failed in {directory}:\n{output}")
    print(f"{directory.name}: passed")
    return output


def solution_error(output):
    if not re.search(r"Converged:\s+yes", output):
        raise RuntimeError(f"Original biharmonic solve did not converge:\n{output}")
    return float(re.search(r"Relative error:\s+(\S+)", output)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--mpi-exec", default="")
    parser.add_argument("--ranks", type=int, default=2)
    options = parser.parse_args()
    build = options.build_dir.resolve()
    if options.mpi_exec:
        launcher = shlex.split(options.mpi_exec)
        if options.ranks < 1:
            parser.error("--ranks must be positive")
        run(build / "test_rect_mpi.bin", [], build / "runs/rect_mpi",
            [*launcher, "-n", str(options.ranks)])
        for preconditioner in ("diag", "mg"):
            errors = []
            for ranks in (1, options.ranks):
                output = run(
                    build / "test_biharmonic_mpi.bin",
                    ["gmres", preconditioner, "8"],
                    build / f"runs/mpi_{ranks}_{preconditioner}",
                    [*launcher, "-n", str(ranks)],
                )
                errors.append(solution_error(output))
            if not math.isclose(*errors, rel_tol=1e-6, abs_tol=1e-12):
                raise RuntimeError(f"Serial/distributed CH errors differ: {errors}")
        return
    run(build / "test_rect.bin", [], build / "runs/rect")
    for preconditioner in ("diag", "mg"):
        output = run(build / "test_biharmonic.bin", ["gmres", preconditioner, "8"],
                     build / f"runs/biharmonic_{preconditioner}")
        solution_error(output)
    run(build / "test_cahn_hilliard.bin", ["gmres", "mg", "8"], build / "runs/stationary")
    for name in ("test_time_cahn_hilliard", "test_cahn_hilliard_nonlinear"):
        run(build / f"{name}.bin", ["gmres", "mg", "8", "--max-time-steps", "2"],
            build / f"runs/{name}")
    run(build / "test_random_init_cahn_hilliard.bin", ["8", "--max-time-steps", "2"],
        build / "runs/random")


if __name__ == "__main__":
    main()
