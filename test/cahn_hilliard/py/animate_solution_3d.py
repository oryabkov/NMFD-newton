#!/usr/bin/env python3
"""
Script to create GIF animations from Cahn-Hilliard solution files with 3D cube face visualization.

The phi function values are displayed on the 6 faces of the cube using the viridis colormap.

Usage:
    python animate_solution_3d.py <folder> [--fps <fps>] [--azimuth <deg>] [--elevation <deg>] [--rotation-angle <deg>]

Example:
    python animate_solution_3d.py data/random_init_ch_test_20260123_181946 --fps 2 --azimuth 45 --elevation 30 --rotation-angle 360
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def create_animation(folder_path, fps=2, output_filename=None, use_non_linear=False, frac=0.0,
                     video_length=None, azimuth=45, elevation=30, rotation_angle=360):
    """
    Create a GIF animation from solution files with 3D cube face visualization.

    The phi function values are displayed on the 6 faces of the cube (x=0, x=1, y=0, y=1, z=0, z=1)
    using the viridis colormap, matching the 2D visualization style.

    Parameters:
    -----------
    folder_path : str
        Path to folder containing numerical_*.bin files
    fps : float
        Frames per second for the animation (default: 2)
    output_filename : str, optional
        Output filename. If None, auto-generated from folder and parameters.
    use_non_linear : bool
        If True, use adaptive frame selection based on cubic polynomial (default: False)
    frac : float
        Controls frame number at T/2: y(T/2) = frac*N (0.0 to 1.0). Only used when use_non_linear=True.
    video_length : float, optional
        Desired video length in seconds. If None, calculated automatically.
    azimuth : float
        Initial azimuth angle for camera position in degrees (default: 45)
    elevation : float
        Initial elevation angle for camera position in degrees (default: 30)
    rotation_angle : float
        Total rotation angle for camera during the entire video in degrees (default: 360)
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
    print(f"Camera: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, rotation={rotation_angle:.1f}°")

    # Load only the selected frames
    numerical_solutions = []
    for frame_idx in frame_indices:
        filename = numerical_files[frame_idx]
        data, N = load_solution(filename)
        numerical_solutions.append(data)

    # Convert to numpy array: shape will be (num_selected_frames, N, N, N, 2)
    numerical = np.array(numerical_solutions)
    num_selected_frames = len(frame_indices)

    # Ensure numerical always has 5 dimensions (frames, N, N, N, components)
    if numerical.ndim == 4:
        numerical = numerical[np.newaxis, ...]
        num_selected_frames = 1

    print(f"Loaded solutions with grid size N = {N}")

    # Rescale coordinates: map cell-centered coordinates [h/2, ..., 1-h/2] to [0, 1]
    # This eliminates the need for interpolation at boundaries
    rescaled_coords = np.linspace(0.0, 1.0, N)

    # Precompute global min/max for consistent colorbar scaling across all frames
    phi_min = numerical[:, :, :, :, 1].min()
    phi_max = numerical[:, :, :, :, 1].max()

    # Normalize function for color mapping
    def normalize_phi(phi_data):
        """Normalize phi data to [0, 1] range for colormap."""
        if phi_max == phi_min:
            return np.zeros_like(phi_data)
        return (phi_data - phi_min) / (phi_max - phi_min)

    # Create coordinate grids for the faces
    Y_face, Z_face = np.meshgrid(rescaled_coords, rescaled_coords, indexing='ij')
    X_face, Z_face_xz = np.meshgrid(rescaled_coords, rescaled_coords, indexing='ij')
    X_face_xy, Y_face_xy = np.meshgrid(rescaled_coords, rescaled_coords, indexing='ij')

    # Calculate angular velocity for camera rotation
    # Total video time = num_selected_frames / fps
    if num_selected_frames > 1:
        total_video_time = num_selected_frames / fps
        angular_velocity = rotation_angle / total_video_time  # degrees per second
    else:
        angular_velocity = 0.0

    # Generate frames
    print("Generating frames...")
    frames = []

    for anim_frame_idx, source_frame_idx in enumerate(frame_indices):
        numerical_frame = numerical[anim_frame_idx]

        # Get phi component (component 1) for the selected frame
        phi_data = numerical_frame[:, :, :, 1]  # Shape: (N, N, N)

        # Calculate current camera azimuth (rotates during animation)
        current_time = anim_frame_idx / fps
        current_azimuth = azimuth + angular_velocity * current_time

        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Enable perspective projection (matplotlib 3D uses perspective by default)
        # Adjust viewing distance to make perspective more pronounced
        ax.dist = 10  # Smaller values = more perspective distortion

        # Face 1: x = 0 (y-z plane)
        face_data = phi_data[0, :, :]
        face_colors = plt.cm.viridis(normalize_phi(face_data))
        ax.plot_surface(np.zeros_like(Y_face), Y_face, Z_face, facecolors=face_colors,
                       shade=False, alpha=1.0, cstride=1, rstride=1, antialiased=False)

        # Face 2: x = 1 (y-z plane)
        face_data = phi_data[-1, :, :]
        face_colors = plt.cm.viridis(normalize_phi(face_data))
        ax.plot_surface(np.ones_like(Y_face), Y_face, Z_face, facecolors=face_colors,
                       shade=False, alpha=1.0, cstride=1, rstride=1, antialiased=False)

        # Face 3: y = 0 (x-z plane)
        face_data = phi_data[:, 0, :]
        face_colors = plt.cm.viridis(normalize_phi(face_data))
        ax.plot_surface(X_face, np.zeros_like(X_face), Z_face_xz, facecolors=face_colors,
                       shade=False, alpha=1.0, cstride=1, rstride=1, antialiased=False)

        # Face 4: y = 1 (x-z plane)
        face_data = phi_data[:, -1, :]
        face_colors = plt.cm.viridis(normalize_phi(face_data))
        ax.plot_surface(X_face, np.ones_like(X_face), Z_face_xz, facecolors=face_colors,
                       shade=False, alpha=1.0, cstride=1, rstride=1, antialiased=False)

        # Face 5: z = 0 (x-y plane)
        face_data = phi_data[:, :, 0]
        face_colors = plt.cm.viridis(normalize_phi(face_data))
        ax.plot_surface(X_face_xy, Y_face_xy, np.zeros_like(X_face_xy), facecolors=face_colors,
                       shade=False, alpha=1.0, cstride=1, rstride=1, antialiased=False)

        # Face 6: z = 1 (x-y plane)
        face_data = phi_data[:, :, -1]
        face_colors = plt.cm.viridis(normalize_phi(face_data))
        ax.plot_surface(X_face_xy, Y_face_xy, np.ones_like(X_face_xy), facecolors=face_colors,
                       shade=False, alpha=1.0, cstride=1, rstride=1, antialiased=False)

        # Set axis limits to show the full cube [0, 1]^3
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # Set labels
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_zlabel('z', fontsize=14)

        # Set camera position
        ax.view_init(elev=elevation, azim=current_azimuth)

        # Title
        title = 'Cahn-Hilliard: φ on cube faces'
        if num_selected_frames > 1:
            if source_frame_idx == 0:
                title += ', Initial approximation'
            else:
                time_str = format_simulation_time(source_frame_idx)
                title += f', {time_str}'
        ax.set_title(title, fontsize=14)

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
        output_filename = folder_path / f'animation_3d_comp1_az{azimuth:.0f}_el{elevation:.0f}_rot{rotation_angle:.0f}.gif'
    else:
        output_filename = Path(output_filename)

    # Save as GIF
    print(f"Saving animation to {output_filename}...")
    imageio.mimsave(str(output_filename), frames, fps=fps)
    print(f"Animation saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D cube face GIF animation from Cahn-Hilliard solution files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 3D animation with default camera settings
  python animate_solution_3d.py data/random_init_ch_test_20260123_181946

  # Create animation with custom camera position and rotation
  python animate_solution_3d.py data/random_init_ch_test_20260123_181946 --azimuth 45 --elevation 30 --rotation-angle 360

  # Create animation with non-linear frame selection
  python animate_solution_3d.py data/random_init_ch_test_20260123_181946 --non-lin --frac 0.5 --video-length 10
        """
    )

    parser.add_argument('folder', type=str,
                       help='Path to folder containing numerical_*.bin files')
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Frames per second for the animation (default: 2.0)')
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
    parser.add_argument('--azimuth', type=float, default=45.0,
                       help='Initial azimuth angle for camera position in degrees (default: 45.0)')
    parser.add_argument('--elevation', type=float, default=30.0,
                       help='Initial elevation angle for camera position in degrees (default: 30.0)')
    parser.add_argument('--rotation-angle', type=float, default=360.0,
                       help='Total rotation angle for camera during the entire video in degrees (default: 360.0)')

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
            fps=args.fps,
            output_filename=args.output,
            use_non_linear=args.non_lin,
            frac=args.frac,
            video_length=args.video_length,
            azimuth=args.azimuth,
            elevation=args.elevation,
            rotation_angle=args.rotation_angle
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
