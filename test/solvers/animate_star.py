#!/usr/bin/env python3
"""
Script to animate the continuation method walking around a star-shaped curve.
Reads a log file and creates a GIF animation of the point trajectory.

The curve is defined by:
F(x, y) = sqrt(x^2 + y^2) - 1 - 0.2 * 4xy(x^2 - y^2) / (x^2 + y^2)^2 = 0

In polar form: r = 1 + 0.2 * sin(4*theta)

Usage: python animate_star.py <path_to_log_file> [output.gif]
"""

import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def parse_log_file(filepath):
    """Parse the log file and extract (x, lambda) coordinates from Step lines."""
    pattern = r"Step\s+\d+:\s+\(x,\s*lambda\)\s*=\s*\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)"

    points = []
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                points.append((x, y))

    return points


def star_curve(theta):
    """
    Compute the radius of the star curve at angle theta.
    r = 1 + 0.2 * sin(4*theta)
    """
    return 1 + 0.2 * np.sin(4 * theta)


def create_animation(points, output_path='star_animation.gif'):
    """Create an animated GIF of the point moving around the star curve."""

    # Set up the figure with specified size
    fig, ax = plt.subplots(figsize=(8, 6))

    # Style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Draw the star curve in polar coordinates
    theta = np.linspace(0, 2 * np.pi, 500)
    r = star_curve(theta)
    star_x = r * np.cos(theta)
    star_y = r * np.sin(theta)

    # Colors - elegant palette
    curve_color = '#8E44AD'       # Purple for the star curve
    line_color = '#E67E22'        # Orange for trajectory
    point_color = '#2ECC71'       # Green for current point
    trail_color = '#E67E22'       # Orange for trail points
    start_color = '#3498DB'       # Blue for start point

    def init():
        ax.clear()
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        return []

    def animate(frame):
        ax.clear()

        # Set up axes
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')

        # Draw the star curve
        ax.plot(star_x, star_y, color=curve_color, linewidth=2.5,
                # label='Star curve: r = 1 + 0.2·sin(4θ)', zorder=1)
                label='Star curve', zorder=1)

        # Also draw a reference unit circle (dashed)
        circle_theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(circle_theta), np.sin(circle_theta),
                color='#95A5A6', linewidth=1, linestyle='--', alpha=0.5,
                label='Unit circle', zorder=0)

        # Get points up to current frame
        current_points = points[:frame + 1]

        if len(current_points) > 0:
            xs = [p[0] for p in current_points]
            ys = [p[1] for p in current_points]

            # Draw the trajectory line
            if len(current_points) > 1:
                ax.plot(xs, ys, color=line_color, linewidth=1.8,
                        linestyle='-', alpha=0.8, zorder=2)

            # Draw trail points (smaller)
            if len(current_points) > 1:
                ax.scatter(xs[:-1], ys[:-1], color=trail_color, s=20,
                          alpha=0.6, zorder=3, edgecolors='white', linewidths=0.5)

            # Draw start point (blue, larger)
            ax.scatter([xs[0]], [ys[0]], color=start_color, s=80,
                      zorder=5, edgecolors='white', linewidths=1.5,
                      label='Start')

            # Draw current point (larger, prominent)
            ax.scatter([xs[-1]], [ys[-1]], color=point_color, s=100,
                      zorder=6, edgecolors='white', linewidths=2,
                      label=f'Step {frame + 1}')

        # Title with step info
        if len(current_points) > 0:
            x, y = current_points[-1]
            ax.set_title(f'Continuation on Star Curve: Step {frame + 1}\n'
                        f'(x, y) = ({x:.4f}, {y:.4f})',
                        fontsize=14, fontweight='bold')

        ax.legend(loc='upper right', fontsize=9)

        return []

    # Create animation
    num_frames = len(points)
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=num_frames, interval=400, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=2.5)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()

    print(f"Animation saved to: {output_path}")
    print(f"Total frames: {num_frames}")


def plot_star_curve_static(output_path='star_curve.png'):
    """Create a static plot of the star curve for reference."""

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the star curve
    theta = np.linspace(0, 2 * np.pi, 500)
    r = star_curve(theta)
    star_x = r * np.cos(theta)
    star_y = r * np.sin(theta)

    ax.plot(star_x, star_y, color='#8E44AD', linewidth=3,
            label=r'$r = 1 + 0.2\sin(4\theta)$')

    # Draw unit circle for reference
    circle_theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta),
            color='#95A5A6', linewidth=1.5, linestyle='--', alpha=0.7,
            label='Unit circle')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('Star-Shaped Curve\n' +
                r'$F(x,y) = \sqrt{x^2+y^2} - 1 - 0.2\cdot\frac{4xy(x^2-y^2)}{(x^2+y^2)^2} = 0$',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    # plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Static plot saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python animate_star.py <path_to_log_file> [output.gif]")
        print("       python animate_star.py --static  (to generate static curve plot)")
        print("Example: python animate_star.py log.txt star_animation.gif")
        sys.exit(1)

    if sys.argv[1] == '--static':
        plot_star_curve_static()
        sys.exit(0)

    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'star_animation.gif'

    print(f"Reading log file: {log_file}")
    points = parse_log_file(log_file)

    if not points:
        print("Error: No valid Step lines found in the log file.")
        sys.exit(1)

    print(f"Found {len(points)} points")
    for i, (x, y) in enumerate(points[:5]):  # Show first 5 points
        print(f"  Step {i+1}: ({x:.6f}, {y:.6f})")
    if len(points) > 5:
        print(f"  ... ({len(points) - 5} more points)")

    print(f"\nCreating animation...")
    create_animation(points, output_file)

    # Also create a static plot of the curve
    plot_star_curve_static('star_curve.png')


if __name__ == '__main__':
    main()

