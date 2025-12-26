#!/usr/bin/env python3
"""
Script to animate the continuation method walking around a unit circle.
Reads a log file and creates a GIF animation of the point trajectory.

Usage: python animate_circle.py <path_to_log_file> [output.gif]
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


def create_animation(points, output_path='circle_animation.gif'):
    """Create an animated GIF of the point moving around the circle."""

    # Set up the figure with specified size
    fig, ax = plt.subplots(figsize=(8, 6))

    # Style settings
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Draw the unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Colors - elegant palette
    circle_color = '#2C3E50'      # Dark blue-gray for circle
    line_color = '#E74C3C'        # Soft red for trajectory
    point_color = '#3498DB'       # Blue for current point
    trail_color = '#E74C3C'       # Red for trail points
    start_color = '#27AE60'       # Green for start point

    def init():
        ax.clear()
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('λ', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
        return []

    def animate(frame):
        ax.clear()

        # Set up axes
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('λ', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')

        # Draw the unit circle
        ax.plot(circle_x, circle_y, color=circle_color, linewidth=2,
                label='Unit circle', zorder=1)

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
                ax.scatter(xs[:-1], ys[:-1], color=trail_color, s=25,
                          alpha=0.6, zorder=3, edgecolors='white', linewidths=0.5)

            # Draw start point (green, larger)
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
            ax.set_title(f'Continuation Method: Step {frame + 1}\n'
                        f'(x, λ) = ({x:.4f}, {y:.4f})',
                        fontsize=14, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10)

        return []

    # Create animation
    num_frames = len(points)
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=num_frames, interval=500, blit=False)

    # Save as GIF
    writer = PillowWriter(fps=2)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()

    print(f"Animation saved to: {output_path}")
    print(f"Total frames: {num_frames}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python animate_circle.py <path_to_log_file> [output.gif]")
        print("Example: python animate_circle.py log.txt circle_animation.gif")
        sys.exit(1)

    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'circle_animation.gif'

    print(f"Reading log file: {log_file}")
    points = parse_log_file(log_file)

    if not points:
        print("Error: No valid Step lines found in the log file.")
        sys.exit(1)

    print(f"Found {len(points)} points")
    for i, (x, y) in enumerate(points):
        print(f"  Step {i+1}: ({x:.6f}, {y:.6f})")

    print(f"\nCreating animation...")
    create_animation(points, output_file)


if __name__ == '__main__':
    main()

