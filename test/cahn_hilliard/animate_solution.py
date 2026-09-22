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


def format_simulation_time(frame_idx, dt=5e-5):
    """
    Format simulation time from frame index with reasonable rounding.

    Parameters:
        frame_idx: Frame index (0-based)
        dt: Integration time step (default: 5e-5)

    Returns:
        Formatted time string
    """
    time = frame_idx * dt

    if frame_idx == 0:
        return "t = 0"

    # Format based on magnitude for readability
    if time < 1e-4:
        # Very small: use scientific notation with 1-2 significant digits
        return f"t = {time:.2e}"
    elif time < 0.001:
        # Small: use 5-6 decimal places
        return f"t = {time:.5f}"
    elif time < 0.01:
        # Medium-small: use 4 decimal places
        return f"t = {time:.4f}"
    elif time < 0.1:
        # Medium: use 3 decimal places
        return f"t = {time:.3f}"
    elif time < 1.0:
        # Medium-large: use 2 decimal places
        return f"t = {time:.2f}"
    elif time < 10.0:
        # Large: use 1 decimal place
        return f"t = {time:.1f}"
    else:
        # Very large: use integer or 1 decimal place
        if time == int(time):
            return f"t = {int(time)}"
        else:
            return f"t = {time:.1f}"


def calculate_frame_indices(num_frames, fps, use_non_linear=True, frac=0.0, video_length=None):
    """
    Calculate which frame indices to use for animation.

    Parameters:
    -----------
    num_frames : int
        Total number of available frames
    fps : float
        Frames per second for the animation
    use_non_linear : bool
        If True, use cubic polynomial for frame selection.
        If False, use all frames linearly.
    frac : float
        Controls the frame number at T/2: y(T/2) = frac*N (0.0 to 1.0)
    video_length : float, optional
        Desired video length in seconds. If None, calculated automatically.

    Returns:
    --------
    frame_indices : list of int
        List of frame indices to use (sorted, unique)
    """
    if not use_non_linear or num_frames <= 1:
        # Linear: use all frames
        return list(range(num_frames))

    # Cubic polynomial: y = ax³ + bx² + cx + d
    # Conditions:
    # 1. y(0) = 0 → d = 0
    # 2. y(dt) = 1 → a(dt)³ + b(dt)² + c(dt) = 1
    # 3. y(T/2) = frac*N → a(T/2)³ + b(T/2)² + c(T/2) = frac*N
    # 4. y(T) = N → aT³ + bT² + cT = N

    dt = 1.0 / fps
    N = num_frames - 1  # Frame indices go from 0 to N

    # If video_length not specified, estimate it based on fps and num_frames
    # Default: use all frames at the given fps
    if video_length is None:
        video_length = N / fps

    T = video_length

    # Set up the linear system: A * [a, b, c]^T = [1, frac*N, N]^T
    # where A is a 3x3 matrix with rows:
    # [dt³, dt², dt]
    # [(T/2)³, (T/2)², T/2]
    # [T³, T², T]

    A = np.array([
        [dt ** 3, dt ** 2, dt],
        [(T / 2) ** 3, (T / 2) ** 2, T / 2],
        [T ** 3, T ** 2, T]
    ])

    b_vec = np.array([1.0, frac * N, N])

    # Solve the linear system
    try:
        coeffs = np.linalg.solve(A, b_vec)
        a, b, c = coeffs
        print(a, b, c)
    except np.linalg.LinAlgError:
        # If system is singular or ill-conditioned, fall back to linear
        print("Warning: Could not solve linear system, falling back to linear frame selection")
        return list(range(num_frames))

    def frame_number(t):
        """Calculate frame number at time t using cubic polynomial."""
        return a * (t ** 3) + b * (t ** 2) + c * t

    # Generate frame indices by sampling simulation time at regular intervals dt
    selected_frames = set()

    # Always include frame 0
    selected_frames.add(0)

    # Sample frames at regular time intervals dt
    t = dt
    while t <= T:
        # Calculate frame number using cubic polynomial
        frame_num = round(frame_number(t))
        # Clamp to valid range
        frame_num = max(0, min(frame_num, N))
        selected_frames.add(frame_num)
        t += dt

    # Always include the last frame
    if num_frames > 1:
        selected_frames.add(N)

    # Return sorted list
    return sorted(selected_frames)


def create_animation(folder_path, slice_axis, layer_idx, fps=2, component=1, output_filename=None, use_non_linear=False, frac=0.0, video_length=None):
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
    use_non_linear : bool
        If True, use adaptive frame selection based on cubic polynomial (default: False)
    frac : float
        Controls frame number at T/2: y(T/2) = frac*N (0.0 to 1.0). Only used when use_non_linear=True.
    video_length : float, optional
        Desired video length in seconds. If None, calculated automatically.
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

    # Determine total number of available frames
    total_frames = len(numerical_files)
    if total_frames == 0:
        raise FileNotFoundError(f"Could not find numerical solution file in {folder_path}")

    # Calculate which frames to use
    frame_indices = calculate_frame_indices(total_frames, fps, use_non_linear, frac, video_length)

    print(f"Total available frames: {total_frames}")
    print(f"Selected frames for animation: {len(frame_indices)}")
    if use_non_linear:
        if video_length is not None:
            print(f"Using cubic polynomial frame selection (video length: {video_length:.2f}s, frac={frac:.2f})")
        else:
            print(f"Using cubic polynomial frame selection (auto video length, frac={frac:.2f})")
        print(f"Selected frame indices: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''}")
    else:
        print(f"Using all frames linearly")
    print(f"Component: {'ψ (psi)' if component == 0 else 'φ (phi)'}")

    # Load only the selected frames
    numerical_solutions = []
    for frame_idx in frame_indices:
        filename = numerical_files[frame_idx]
        try:
            data, N = load_solution(filename)
            numerical_solutions.append(data)
        except Exception as e:
            raise RuntimeError(f"Failed to load solution file {filename} (frame index {frame_idx}): {e}")

    if len(numerical_solutions) == 0:
        raise RuntimeError(f"No solutions were successfully loaded. Check that files exist and are readable.")

    # Convert to numpy array: shape will be (num_selected_frames, N, N, N, 2)
    try:
        numerical = np.array(numerical_solutions)
    except Exception as e:
        # Check if arrays have inconsistent shapes
        if len(numerical_solutions) > 0:
            shapes = [arr.shape for arr in numerical_solutions]
            raise RuntimeError(f"Failed to convert solutions to numpy array. "
                             f"Array shapes: {shapes}. "
                             f"This might indicate inconsistent data shapes. Error: {e}")
        raise RuntimeError(f"Failed to convert solutions to numpy array: {e}")
    num_selected_frames = len(frame_indices)

    # Ensure numerical always has 5 dimensions (frames, N, N, N, components)
    if numerical.ndim == 4:
        numerical = numerical[np.newaxis, ...]
        num_selected_frames = 1

    # Validate that we actually loaded data
    if numerical.size == 0:
        raise RuntimeError(f"Loaded numerical array is empty. Expected {num_selected_frames} frames but got 0.")

    if len(numerical_solutions) != num_selected_frames:
        raise RuntimeError(f"Mismatch: expected {num_selected_frames} frames but loaded {len(numerical_solutions)} frames.")

    # Validate array shape
    if numerical.shape[0] == 0:
        raise RuntimeError(f"numerical array has 0 frames on axis 0. Shape: {numerical.shape}, "
                         f"expected at least {num_selected_frames} frames.")

    print(f"Loaded solutions with grid size N = {N}")
    print(f"numerical array shape: {numerical.shape}")

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

    for anim_frame_idx, source_frame_idx in enumerate(frame_indices):
        if anim_frame_idx >= numerical.shape[0]:
            raise IndexError(f"Frame index {anim_frame_idx} is out of bounds. "
                           f"numerical array has shape {numerical.shape}, "
                           f"but trying to access frame {anim_frame_idx}. "
                           f"Expected {num_selected_frames} frames but got {numerical.shape[0]}.")
        numerical_frame = numerical[anim_frame_idx]

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
        if num_selected_frames > 1:
            if source_frame_idx == 0:
                title += ', Initial approximation'
            else:
                time_str = format_simulation_time(source_frame_idx)
                title += f', {time_str}'
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

        if (anim_frame_idx + 1) % 10 == 0 or anim_frame_idx == num_selected_frames - 1:
            print(f"  Generated {anim_frame_idx + 1}/{num_selected_frames} frames")

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
    parser.add_argument('--non-lin', dest='non_lin', action='store_true',
                       help='Enable adaptive frame selection using cubic polynomial y=ax³+bx²+cx. '
                            'The polynomial satisfies: y(0)=0, y(dt)=1, y(T/2)=frac*N, y(T)=N.')
    parser.add_argument('--frac', type=float, default=0.0,
                       help='Controls frame number at T/2: y(T/2) = frac*N (0.0 to 1.0). '
                            'Only used with --non-lin. Lower values make beginning faster. (default: 0.0)')
    parser.add_argument('--video-length', type=float, default=None,
                       help='Desired video length in seconds. Only used with --non-lin. '
                            'If not specified, calculated automatically based on fps and number of frames.')

    args = parser.parse_args()

    # Validate frac parameter
    if args.non_lin and (args.frac < 0.0 or args.frac > 1.0):
        parser.error("--frac must be between 0.0 and 1.0")

    # Validate video_length parameter
    if args.video_length is not None and args.video_length <= 0:
        parser.error("--video-length must be positive")

    # video_length only makes sense with --non-lin
    if args.video_length is not None and not args.non_lin:
        parser.error("--video-length can only be used with --non-lin")

    try:
        create_animation(
            folder_path=args.folder,
            slice_axis=args.axis,
            layer_idx=args.layer,
            fps=args.fps,
            component=args.component,
            output_filename=args.output,
            use_non_linear=args.non_lin,
            frac=args.frac,
            video_length=args.video_length
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
