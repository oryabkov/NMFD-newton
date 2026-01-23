#!/usr/bin/env python3
"""
Script to create GIF animations from Cahn-Hilliard solution files.

Usage:
    python animate_solution.py <folder> --axis <x|y|z> --layer <N> [--fps <fps>] [--component <0|1>]

Example:
    python animate_solution.py data/random_init_ch_test_20260123_181946 --axis z --layer 32 --fps 2
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
from pathlib import Path

try:
    import imageio
except ImportError:
    print("Error: imageio is required. Install it with: pip install imageio")
    exit(1)


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


def create_animation(folder_path, slice_axis, layer_idx, fps=2, component=1, output_filename=None):
    """
    Create a GIF animation from solution files.

    Parameters:
    -----------
    folder_path : str
        Path to folder containing numerical_*.bin files
    slice_axis : str
        Axis to slice along ('x', 'y', or 'z')
    layer_idx : int
        Layer index along the slice axis (0 to N-1)
    fps : float
        Frames per second for the animation (default: 2)
    component : int
        Component to visualize (0=psi, 1=phi, default: 1)
    output_filename : str, optional
        Output filename. If None, auto-generated from folder and parameters.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Check for time-dependent solutions (numerical_0.bin, numerical_1.bin, ...)
    numerical_files = sorted(glob.glob(str(folder_path / 'numerical_*.bin')),
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if len(numerical_files) == 0:
        # Time-independent case: single numerical.bin file
        numerical_files = [str(folder_path / 'numerical.bin')]
        if not os.path.exists(numerical_files[0]):
            raise FileNotFoundError(f"Could not find numerical solution file in {folder_path}")

    # Load all numerical solutions
    numerical_solutions = []
    for filename in numerical_files:
        data, N = load_solution(filename)
        numerical_solutions.append(data)

    # Convert to numpy array: shape will be (num_frames, N, N, N, 2)
    numerical = np.array(numerical_solutions)
    num_frames = len(numerical_solutions)

    # Ensure numerical always has 5 dimensions (frames, N, N, N, components)
    if numerical.ndim == 4:
        numerical = numerical[np.newaxis, ...]
        num_frames = 1

    print(f"Loaded solutions with grid size N = {N}")
    print(f"Number of frames: {num_frames}")
    print(f"Component: {'ψ (psi)' if component == 0 else 'φ (phi)'}")

    # Validate layer index
    if layer_idx < 0 or layer_idx >= N:
        raise ValueError(f"Layer index {layer_idx} is out of range [0, {N-1}]")

    # Map axis name to index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if slice_axis.lower() not in axis_map:
        raise ValueError(f"Invalid axis '{slice_axis}'. Must be 'x', 'y', or 'z'")
    slice_axis_idx = axis_map[slice_axis.lower()]

    # Calculate cell-centered coordinate values: h*(0.5 + i) for i = 0, 1, ..., N-1
    h = 1.0 / N
    cell_centered_coords = np.array([h * (0.5 + i) for i in range(N)])

    # Extract 2D slice based on slice_axis and layer_idx
    axis_names = ['x', 'y', 'z']
    component_names = ['ψ (psi)', 'φ (phi)']

    # Determine coordinate labels and slice data extraction
    if slice_axis_idx == 0:  # Slice along x-axis: show y-z plane
        x_coords = cell_centered_coords  # y coordinates
        y_coords = cell_centered_coords  # z coordinates
        xlabel = 'y'
        ylabel = 'z'
        title_axis = f'x = {cell_centered_coords[layer_idx]:.4f}'
    elif slice_axis_idx == 1:  # Slice along y-axis: show x-z plane
        x_coords = cell_centered_coords  # x coordinates
        y_coords = cell_centered_coords  # z coordinates
        xlabel = 'x'
        ylabel = 'z'
        title_axis = f'y = {cell_centered_coords[layer_idx]:.4f}'
    else:  # slice_axis_idx == 2, Slice along z-axis: show x-y plane
        x_coords = cell_centered_coords  # x coordinates
        y_coords = cell_centered_coords  # y coordinates
        xlabel = 'x'
        ylabel = 'y'
        title_axis = f'z = {cell_centered_coords[layer_idx]:.4f}'

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

    # Generate frames
    print("Generating frames...")
    frames = []

    for frame_idx in range(num_frames):
        numerical_frame = numerical[frame_idx]

        # Extract 2D slice
        if slice_axis_idx == 0:
            slice_data = numerical_frame[layer_idx, :, :, component]
        elif slice_axis_idx == 1:
            slice_data = numerical_frame[:, layer_idx, :, component]
        else:  # slice_axis_idx == 2
            slice_data = numerical_frame[:, :, layer_idx, component]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap with automatic scaling for each frame
        im = ax.pcolormesh(X, Y, slice_data,
                          cmap='viridis',
                          shading='gouraud')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(component_names[component], fontsize=14)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        # Title
        title = f'Cahn-Hilliard: {component_names[component]} slice at {title_axis}'
        if num_frames > 1:
            if frame_idx == 0:
                title += ', Initial approximation'
            else:
                title += f', After iteration {frame_idx}'
        ax.set_title(title, fontsize=14)

        ax.set_aspect('equal')
        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()

        # Get the RGBA buffer and convert to RGB
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        # Convert RGBA to RGB
        frame = frame[:, :, :3]
        frames.append(frame)

        plt.close(fig)

        if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
            print(f"  Generated {frame_idx + 1}/{num_frames} frames")

    # Generate output filename if not provided
    if output_filename is None:
        folder_name = folder_path.name
        output_filename = folder_path / f'animation_{slice_axis}_layer{layer_idx}_comp{component}.gif'
    else:
        output_filename = Path(output_filename)

    # Save as GIF
    print(f"Saving animation to {output_filename}...")
    imageio.mimsave(str(output_filename), frames, fps=fps)
    print(f"Animation saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Create GIF animation from Cahn-Hilliard solution files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create animation slicing along z-axis at layer 32
  python animate_solution.py data/random_init_ch_test_20260123_181946 --axis z --layer 32

  # Create animation with custom FPS and component
  python animate_solution.py data/random_init_ch_test_20260123_181946 --axis z --layer 32 --fps 5 --component 0

  # Create animation with custom output filename
  python animate_solution.py data/random_init_ch_test_20260123_181946 --axis z --layer 32 --output my_animation.gif
        """
    )

    parser.add_argument('folder', type=str,
                       help='Path to folder containing numerical_*.bin files')
    parser.add_argument('--axis', type=str, choices=['x', 'y', 'z'], required=True,
                       help='Axis along which to slice (x, y, or z)')
    parser.add_argument('--layer', type=int, required=True,
                       help='Layer index along the slice axis (0 to N-1)')
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Frames per second for the animation (default: 2.0)')
    parser.add_argument('--component', type=int, choices=[0, 1], default=1,
                       help='Component to visualize: 0=psi, 1=phi (default: 1)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename. If not specified, auto-generated.')

    args = parser.parse_args()

    try:
        create_animation(
            folder_path=args.folder,
            slice_axis=args.axis,
            layer_idx=args.layer,
            fps=args.fps,
            component=args.component,
            output_filename=args.output
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
