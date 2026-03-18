#!/usr/bin/env python3
"""
Create GIF animations of 3D isosurfaces from Cahn–Hilliard solution files.

The script loads `numerical_*.bin` (time-dependent) or `numerical.bin`
from a folder and, for each selected frame, extracts an isosurface
of the phi component using marching cubes and renders it as an opaque,
shaded surface to convey volume.

Usage:
    python animate_isosurface_3d.py <folder> [--fps <fps>] [--level <value>] [--output <gif>]

Example:
    python animate_isosurface_3d.py data/circle_test_20260316_173610 --fps 2 --level 0.0
"""

import argparse
import glob
import os
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def load_solution(filename):
    """Load solution from binary file saved by C++ code."""
    with open(filename, "rb") as f:
        dims = np.frombuffer(f.read(12), dtype=np.int32)
        n_components = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(), dtype=np.float64)
        data = data.reshape((dims[2], dims[1], dims[0], n_components))
        # Transpose to (N, N, N, 2) with x as first axis
        data = np.transpose(data, (2, 1, 0, 3))
    return data, int(dims[0])


def format_simulation_time(frame_idx, dt=5e-5):
    """Format simulation time string from frame index (same style as visual tools)."""
    time = frame_idx * dt
    if frame_idx == 0:
        return "t = 0"
    if time < 1e-4:
        return f"t = {time:.2e}"
    if time < 0.001:
        return f"t = {time:.5f}"
    if time < 0.01:
        return f"t = {time:.4f}"
    if time < 0.1:
        return f"t = {time:.3f}"
    if time < 1.0:
        return f"t = {time:.2f}"
    if time < 10.0:
        return f"t = {time:.1f}"
    if time == int(time):
        return f"t = {int(time)}"
    return f"t = {time:.1f}"


def select_frames_linear(num_frames):
    """Simple linear frame selection: use all available frames."""
    return list(range(num_frames))


def compute_shaded_colors(normals, cmap_name="viridis", light_dir=(1.0, 1.0, 1.0)):
    """
    Compute per-face colors with simple Lambertian shading.

    Parameters
    ----------
    normals : (F, 3) array
        Per-face normals (unit or non-unit).
    cmap_name : str
        Matplotlib colormap name.
    light_dir : tuple
        Direction of a distant light source.
    """
    light = np.array(light_dir, dtype=float)
    light /= np.linalg.norm(light)

    # Ensure normals has shape (F, 3)
    n = normals
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_norm[n_norm == 0.0] = 1.0
    n_unit = n / n_norm

    intensity = np.clip(np.einsum("ij,j->i", n_unit, light), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(intensity)
    colors[:, 3] = 1.0  # fully opaque
    return colors


def create_animation(folder_path, fps=2.0, level=0.0, output_filename=None,
                     azimuth=45.0, elevation=30.0):
    """
    Create a GIF animation of 3D isosurfaces for the phi component.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing numerical_*.bin files.
    fps : float
        Frames per second for the animation.
    level : float
        Isosurface value for phi (default: 0.0).
    output_filename : str, optional
        Output filename. If None, auto-generated from folder and level.
    azimuth : float
        Camera azimuth angle in degrees (default: 45.0).
    elevation : float
        Camera elevation angle in degrees (default: 30.0).
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Time-dependent (numerical_*.bin) or stationary (numerical.bin)
    numerical_files = sorted(
        glob.glob(str(folder_path / "numerical_*.bin")),
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )
    if len(numerical_files) == 0:
        numerical_files = [str(folder_path / "numerical.bin")]
        if not os.path.exists(numerical_files[0]):
            raise FileNotFoundError(
                f"Could not find numerical solution file in {folder_path}"
            )

    total_frames = len(numerical_files)
    if total_frames == 0:
        raise FileNotFoundError(
            f"Could not find numerical solution file in {folder_path}"
        )

    frame_indices = select_frames_linear(total_frames)

    print(f"Total available frames: {total_frames}")
    print(f"Selected frames for animation: {len(frame_indices)}")
    print(f"Isosurface level: {level}")

    # Load selected frames
    numerical_solutions = []
    for frame_idx in frame_indices:
        filename = numerical_files[frame_idx]
        data, N = load_solution(filename)
        numerical_solutions.append(data)

    numerical = np.array(numerical_solutions)
    num_selected_frames = len(frame_indices)

    if numerical.ndim == 4:
        numerical = numerical[np.newaxis, ...]
        num_selected_frames = 1

    print(f"Loaded solutions with grid size N = {N}")

    # Cell-centered coordinates
    h = 1.0 / N

    frames = []
    print("Generating isosurface frames...")

    for anim_frame_idx, source_frame_idx in enumerate(frame_indices):
        phi_data = numerical[anim_frame_idx, :, :, :, 1]  # phi component

        # Extract isosurface at given level
        try:
            verts, faces, normals, values = measure.marching_cubes(
                phi_data, level=level, spacing=(h, h, h)
            )
        except ValueError as e:
            print(
                f"Warning: could not extract isosurface (frame {source_frame_idx}): {e}"
            )
            # Create an empty frame with a message
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                f"No isosurface (phi={level:g}) in frame {source_frame_idx}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.axis("off")
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3]
            frames.append(frame)
            plt.close(fig)
            continue

        # Compute per-face normals (average of vertex normals for each face)
        face_normals = normals[faces].mean(axis=1)
        face_colors = compute_shaded_colors(face_normals, cmap_name="viridis")

        # Plot opaque shaded isosurface
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Thin mesh edges for a less bulky wireframe look
        mesh = Poly3DCollection(
            verts[faces],
            facecolors=face_colors,
            edgecolor="k",
            linewidths=0.2,
        )
        mesh.set_alpha(1.0)
        ax.add_collection3d(mesh)

        # Axis limits: full cube [0, 1]^3
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_zlabel("z", fontsize=14)

        # Camera settings (user‑controlled via CLI)
        ax.view_init(elev=elevation, azim=azimuth)

        title = f"Cahn-Hilliard: φ = {level:g} isosurface"
        if num_selected_frames > 1:
            if source_frame_idx == 0:
                title += ", Initial approximation"
            else:
                time_str = format_simulation_time(source_frame_idx)
                title += f", {time_str}"
        ax.set_title(title, fontsize=14)

        plt.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3]
        frames.append(frame)
        plt.close(fig)

        if (anim_frame_idx + 1) % 10 == 0 or anim_frame_idx == num_selected_frames - 1:
            print(f"  Generated {anim_frame_idx + 1}/{num_selected_frames} frames")

    # Output filename
    if output_filename is None:
        folder_name = folder_path.name
        safe_level = f"{level}".replace(".", "p").replace("-", "m")
        output_filename = folder_path / f"animation_isosurf_phi_{safe_level}.gif"
    else:
        output_filename = Path(output_filename)

    print(f"Saving animation to {output_filename}...")
    imageio.mimsave(str(output_filename), frames, fps=fps)
    print("Animation saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D isosurface GIF animation from Cahn–Hilliard solution files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default isosurface at phi = 0
  python animate_isosurface_3d.py data/circle_test_20260316_173610

  # Different level and fps
  python animate_isosurface_3d.py data/circle_test_20260316_173610 --level 0.5 --fps 3
        """,
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing numerical_*.bin (or numerical.bin) files",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for the animation (default: 2.0)",
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.0,
        help="Isosurface value for phi (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF filename. If not specified, auto-generated.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=45.0,
        help="Camera azimuth angle in degrees (default: 45.0)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=30.0,
        help="Camera elevation angle in degrees (default: 30.0)",
    )

    args = parser.parse_args()

    try:
        create_animation(
            folder_path=args.folder,
            fps=args.fps,
            level=args.level,
            output_filename=args.output,
            azimuth=args.azimuth,
            elevation=args.elevation,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

