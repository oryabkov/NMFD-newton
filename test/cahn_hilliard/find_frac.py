#!/usr/bin/env python3
"""
Script to find the optimal frac parameter for matching animation curves.

Given:
- Old animation: 10k steps, T=10, frac=0.1, fps=24
- New animation: 87629 steps, T=15, fps=24, frac=???

Find frac such that the new curve best matches the old curve over time [0, 10].
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, root_scalar


def compute_cubic_coefficients(num_frames, fps, T, frac):
    """
    Compute cubic polynomial coefficients for frame selection.

    Returns:
        (a, b, c) - coefficients of y = ax³ + bx² + cx
    """
    dt = 1.0 / fps
    N = num_frames - 1

    # Set up linear system: A * [a, b, c]^T = [1, frac*N, N]^T
    A = np.array([
        [dt ** 3, dt ** 2, dt],
        [(T / 2) ** 3, (T / 2) ** 2, T / 2],
        [T ** 3, T ** 2, T]
    ])

    b_vec = np.array([1.0, frac * N, N])

    try:
        coeffs = np.linalg.solve(A, b_vec)
        return coeffs[0], coeffs[1], coeffs[2]
    except np.linalg.LinAlgError:
        raise ValueError(f"Cannot solve linear system for frac={frac}")


def evaluate_curve(t, a, b, c):
    """Evaluate cubic polynomial y = ax³ + bx² + cx at time t."""
    return a * (t ** 3) + b * (t ** 2) + c * t


def find_time_for_frame_value(num_frames, fps, T, frac, target_frame_value, t_max=None):
    """
    Find the time at which the curve reaches a target frame value.

    Parameters:
        num_frames: number of frames
        fps: frames per second
        T: video length
        frac: frac parameter
        target_frame_value: target frame number to find time for
        t_max: maximum time to search within (default: T)

    Returns:
        Time value where curve reaches target_frame_value
    """
    if t_max is None:
        t_max = T

    # Compute coefficients
    a, b, c = compute_cubic_coefficients(num_frames, fps, T, frac)

    # Define function to find root: ax³ + bx² + cx - target = 0
    def f(t):
        return evaluate_curve(t, a, b, c) - target_frame_value

    # Check bounds
    y_at_0 = evaluate_curve(0, a, b, c)
    y_at_max = evaluate_curve(t_max, a, b, c)

    if target_frame_value < y_at_0:
        return 0.0
    if target_frame_value > y_at_max:
        return t_max

    # Find root using Brent's method
    try:
        result = root_scalar(f, bracket=[0.0, t_max], method='brentq')
        return result.root
    except ValueError:
        # If bracket doesn't work, try bisection with wider range
        try:
            result = root_scalar(f, bracket=[0.0, t_max * 2], method='brentq')
            return min(result.root, t_max)
        except:
            # Fallback: binary search
            t_low, t_high = 0.0, t_max
            for _ in range(100):  # Max iterations
                t_mid = (t_low + t_high) / 2
                y_mid = evaluate_curve(t_mid, a, b, c)
                if abs(y_mid - target_frame_value) < 1e-6:
                    return t_mid
                if y_mid < target_frame_value:
                    t_low = t_mid
                else:
                    t_high = t_mid
            return (t_low + t_high) / 2


def compute_curve_mismatch(frac_new, num_frames_old, num_frames_new, fps, T_old, T_new, t_eval):
    """
    Compute mismatch between old and new curves over time range t_eval.
    Uses real frame numbers (not normalized) for matching.

    Parameters:
        frac_new: frac parameter for new animation
        num_frames_old: number of frames in old animation
        num_frames_new: number of frames in new animation
        fps: frames per second
        T_old: video length for old animation
        T_new: video length for new animation
        t_eval: array of time points to evaluate at

    Returns:
        Mean squared error between real frame number curves
    """
    # Compute coefficients for old curve
    frac_old = 0.1
    a_old, b_old, c_old = compute_cubic_coefficients(num_frames_old, fps, T_old, frac_old)

    # Compute coefficients for new curve
    a_new, b_new, c_new = compute_cubic_coefficients(num_frames_new, fps, T_new, frac_new)

    # Evaluate both curves (real frame numbers)
    y_old = np.array([evaluate_curve(t, a_old, b_old, c_old) for t in t_eval])
    y_new = np.array([evaluate_curve(t, a_new, b_new, c_new) for t in t_eval])

    # Compute mean squared error using real frame numbers
    mse = np.mean((y_old - y_new) ** 2)
    mae = np.mean(np.abs(y_old - y_new))

    return mae


def find_optimal_frac(num_frames_old, num_frames_new, fps, T_old, T_new, t_max=10.0):
    """
    Find optimal frac parameter for new animation.

    Parameters:
        num_frames_old: number of frames in old animation (10000)
        num_frames_new: number of frames in new animation (87629)
        fps: frames per second (24)
        T_old: video length for old animation (10)
        T_new: video length for new animation (15)
        t_max: maximum time to match over (default: 10, matching T_old)

    Returns:
        (optimal_frac, error) - tuple of optimal frac value and final error
    """
    # Create time evaluation points (dense sampling over [0, t_max])
    t_eval = np.linspace(0, t_max, 1000)

    # Objective function: minimize mismatch
    def objective(frac):
        return compute_curve_mismatch(
            frac, num_frames_old, num_frames_new, fps, T_old, T_new, t_eval
        )

    # Search for optimal frac in [0, 1]
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')

    final_error = result.fun
    return result.x, final_error


def compute_error_metrics(frac_new, num_frames_old, num_frames_new, fps, T_old, T_new, t_eval):
    """
    Compute detailed error metrics between old and new curves.

    Returns:
        dict with 'mse', 'rmse', 'mae', 'max_error' keys
    """
    # Compute coefficients
    frac_old = 0.1
    a_old, b_old, c_old = compute_cubic_coefficients(num_frames_old, fps, T_old, frac_old)
    a_new, b_new, c_new = compute_cubic_coefficients(num_frames_new, fps, T_new, frac_new)

    # Evaluate curves
    N_old = num_frames_old - 1
    N_new = num_frames_new - 1

    y_old = np.array([evaluate_curve(t, a_old, b_old, c_old) for t in t_eval])
    y_new = np.array([evaluate_curve(t, a_new, b_new, c_new) for t in t_eval])

    # Normalize to [0, 1] range
    y_old_norm = y_old
    y_new_norm = y_new

    # Compute error metrics
    errors = y_old_norm - y_new_norm
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'y_old': y_old_norm,
        'y_new': y_new_norm,
        't': t_eval
    }


def plot_curves(num_frames_old, num_frames_new, fps, T_old, T_new, frac_old, frac_new, t_max=10.0, target_frame_value=None, output_file=None):
    """
    Plot comparison of old and new curves over their full time ranges.
    Old curve: 0 to T_old, New curve: 0 to T_new.

    Parameters:
        t_max: Maximum time for comparison region
        target_frame_value: If provided, show vertical line at time where old curve reaches this value
        output_file: Optional filename to save the plot. If None, displays interactively.
    """
    # Create time evaluation points for both curves
    t_eval_old = np.linspace(0, T_old, 1000)
    t_eval_new = np.linspace(0, T_new, 1000)
    t_eval_match = np.linspace(0, t_max, 1000)  # For error computation

    # Compute coefficients
    a_old, b_old, c_old = compute_cubic_coefficients(num_frames_old, fps, T_old, frac_old)
    a_new, b_new, c_new = compute_cubic_coefficients(num_frames_new, fps, T_new, frac_new)

    # Evaluate curves over their full ranges
    y_old_full = np.array([evaluate_curve(t, a_old, b_old, c_old) for t in t_eval_old])
    y_new_full = np.array([evaluate_curve(t, a_new, b_new, c_new) for t in t_eval_new])

    # Compute error metrics for matching region
    metrics = compute_error_metrics(frac_new, num_frames_old, num_frames_new, fps, T_old, T_new, t_eval_match)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot curves over their full time ranges
    ax.plot(t_eval_old, y_old_full, 'b-', linewidth=2, label=f'Old: {num_frames_old} frames, T={T_old}s, frac={frac_old}')
    ax.plot(t_eval_new, y_new_full, 'r--', linewidth=2, label=f'New: {num_frames_new} frames, T={T_new}s, frac={frac_new:.6f}')

    # Add vertical line at matching region end
    if target_frame_value is not None:
        # Show where old curve reaches target frame value
        t_target = find_time_for_frame_value(num_frames_old, fps, T_old, frac_old, target_frame_value, T_old)
        ax.axvline(x=t_target, color='green', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Old curve reaches {target_frame_value} frames at t={t_target:.3f}s')
        ax.axhline(y=target_frame_value, color='green', linestyle=':', linewidth=1, alpha=0.5)
    else:
        # Fallback to T_old
        ax.axvline(x=t_max, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'Matching region end (t={t_max:.3f}s)')

    # Labels and title
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Frame Number', fontsize=14)
    ax.set_title('Frame Selection Curve Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)

    # Add text box with error metrics
    error_text = f'Error Metrics (over [0, {t_max:.1f}s]):\n'
    error_text += f'MSE: {metrics["mse"]:.2e}\n'
    error_text += f'RMSE: {metrics["rmse"]:.2f}\n'
    error_text += f'MAE: {metrics["mae"]:.2f}\n'
    error_text += f'Max Error: {metrics["max_error"]:.2f}'
    ax.text(0.02, 0.98, error_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Find optimal frac parameter for matching animation curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default target frame value (3000)
  python find_frac.py

  # Compare until old curve reaches 5000 frames
  python find_frac.py --target-frames 5000

  # Compare over fixed time range instead
  python find_frac.py --time-range 5.0
        """
    )

    parser.add_argument('--target-frames', type=float, default=3000,
                       help='Target frame value: compare curves until old curve reaches this value (default: 3000)')
    parser.add_argument('--time-range', type=float, default=None,
                       help='Alternative: use fixed time range [0, time_range] for comparison (overrides --target-frames)')
    parser.add_argument('--num-frames-old', type=int, default=10000,
                       help='Number of frames in old animation (default: 10000)')
    parser.add_argument('--num-frames-new', type=int, default=87661,
                       help='Number of frames in new animation (default: 87661)')
    parser.add_argument('--fps', type=float, default=24,
                       help='Frames per second (default: 24)')
    parser.add_argument('--T-old', type=float, default=10.0,
                       help='Video length for old animation (default: 10.0)')
    parser.add_argument('--T-new', type=float, default=18.0,
                       help='Video length for new animation (default: 18.0)')
    parser.add_argument('--frac-old', type=float, default=0.1,
                       help='Frac parameter for old animation (default: 0.1)')

    args = parser.parse_args()

    # Parameters from user's description
    num_frames_old = args.num_frames_old
    num_frames_new = args.num_frames_new
    fps = args.fps
    T_old = args.T_old
    T_new = args.T_new
    frac_old = args.frac_old

    # Determine comparison range
    if args.time_range is not None:
        # Use fixed time range
        t_max = args.time_range
        target_frame_value = None
        print("Finding optimal frac parameter...")
        print(f"Old animation: {num_frames_old} frames, T={T_old}s, frac={frac_old}")
        print(f"New animation: {num_frames_new} frames, T={T_new}s, frac=???")
        print(f"Matching curves over time [0, {t_max:.6f}s]")
    else:
        # Use target frame value
        target_frame_value = args.target_frames
        print("Finding optimal frac parameter...")
        print(f"Old animation: {num_frames_old} frames, T={T_old}s, frac={frac_old}")
        print(f"New animation: {num_frames_new} frames, T={T_new}s, frac=???")

        # Find time where old curve reaches target frame value
        t_max = find_time_for_frame_value(num_frames_old, fps, T_old, frac_old, target_frame_value, T_old)
        print(f"Matching curves over time [0, {t_max:.6f}s] (until old curve reaches {target_frame_value} frames)")
    print()

    # Find optimal frac
    optimal_frac, final_error = find_optimal_frac(num_frames_old, num_frames_new, fps, T_old, T_new, t_max=t_max)

    print(f"Optimal frac: {optimal_frac:.6f}")
    print(f"Final MSE (optimization objective): {final_error:.2e}")
    print()

    # Compute detailed error metrics
    t_eval = np.linspace(0, t_max, 1000)
    metrics = compute_error_metrics(optimal_frac, num_frames_old, num_frames_new, fps, T_old, T_new, t_eval)

    print("Approximation Error Metrics:")
    print(f"  Mean Squared Error (MSE):     {metrics['mse']:.2e}")
    print(f"  Root Mean Squared Error (RMSE): {metrics['rmse']:.6f}")
    print(f"  Mean Absolute Error (MAE):   {metrics['mae']:.6f}")
    print(f"  Maximum Absolute Error:      {metrics['max_error']:.6f}")
    print()

    # Create and save plot
    plot_filename = 'curve_comparison.png'
    plot_curves(num_frames_old, num_frames_new, fps, T_old, T_new, frac_old, optimal_frac,
                t_max=t_max, target_frame_value=target_frame_value, output_file=plot_filename)

    print()
    print(f"Use this command:")
    print(f"  python animate_solution.py <folder> --axis z --layer <N> --non-lin --fps {fps} --video-length {T_new} --frac {optimal_frac:.6f}")


if __name__ == '__main__':
    main()
