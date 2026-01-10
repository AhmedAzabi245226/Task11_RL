"""
corners.py (Task 9 - OT-2 Digital Twin)

This script finds the OT-2 pipette work envelope (workspace limits).

What it does:
1) Starts the OT-2 simulation (PyBullet digital twin).
2) Sends velocity commands to move the pipette in 8 diagonal directions.
3) When the pipette stops moving (hits a limit), it records that position as a corner.
4) Prints observations (target corner name, pipette position, small movement per step, stability counter).
5) Saves a GIF (corners.gif) where the observations are drawn on the frames.

How to run:
- Activate your environment.
- Install dependencies: pip install numpy imageio opencv-python
- Run: python corners.py

Output:
- corners.gif  (robot motion + on-screen observations)
- Printed corner coordinates in the terminal
"""

from sim_class import Simulation
import numpy as np
import imageio
import cv2


# ----------------------------
# Settings you can tweak
# ----------------------------
OUTPUT_GIF = "corners.gif"
FPS = 15

MAX_STEPS = 1200         # maximum simulation steps per corner direction
STABLE_REQUIRED = 5      # how many consecutive "almost no movement" steps = stable
TOL = 1e-4               # movement threshold for stable check


def euclid(a, b) -> float:
    """
    Return Euclidean distance between two 3D points.
    """
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def annotate_frame(frame, lines):
    """
    Draw a small text overlay (observations) onto a simulation frame.

    - Makes sure the frame is RGB uint8.
    - Draws a small black box at the bottom-center.
    - Draws white text with a small shadow so it stays readable in a GIF.
    """
    img = np.array(frame)

    # If frame has alpha channel (RGBA), drop it (keep RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Make sure pixels are uint8 (0..255)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    img = img.copy()
    h, w = img.shape[:2]

    # Smaller text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38          # smaller font
    thickness = 1
    dy = 12               # tighter line spacing

    # Compact box size (bottom-center)
    box_w = 260
    box_h = dy * (len(lines) + 1)

    margin_bottom = 3
    x0 = max(0, (w - box_w) // 2)
    x1 = min(w - 1, x0 + box_w)
    y1 = h - margin_bottom
    y0 = max(0, y1 - box_h)

    # Background box
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)

    # Text start position
    tx = x0 + 6
    ty = y0 + 14

    for i, text in enumerate(lines):
        y = ty + i * dy

        # Shadow (helps GIF readability)
        cv2.putText(img, text, (tx + 1, y + 1), font, scale, (0, 0, 0),
                    thickness + 1, cv2.LINE_8)
        # Text
        cv2.putText(img, text, (tx, y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_8)

    return img


def main():
    """
    Main routine:
    - Start the simulation
    - Move to all 8 corners
    - Print and store corner coordinates
    - Save GIF with overlay text
    """
    # Start simulation with rendering + RGB frames
    sim = Simulation(num_agents=1, render=True, rgb_array=True)

    gif_frames = []

    def move_until_stable(vel, name):
        """
        Move in a given velocity direction until the pipette stops moving.
        The stop is detected when movement per step (delta) is below TOL
        for STABLE_REQUIRED consecutive steps.

        Returns:
            list[pip_x, pip_y, pip_z] at the stable (corner) position
        """
        print(f"\n===== Moving toward {name} =====")

        last_pip = None
        stable_count = 0

        for step in range(MAX_STEPS):
            state = sim.run([[vel[0], vel[1], vel[2], 0]])
            pip = tuple(state["robotId_1"]["pipette_position"])

            # delta/step is a simple velocity proxy (distance moved per sim step)
            delta = 0.0 if last_pip is None else euclid(pip, last_pip)

            # stability check
            if last_pip is not None and delta < TOL:
                stable_count += 1
            else:
                stable_count = 0

            # Print observations in terminal
            print(
                f"step={step:04d} cmd_vel={vel} pip={pip} "
                f"delta_per_step={delta:.6f} stable={stable_count}/{STABLE_REQUIRED}"
            )

            # Overlay small observations onto frame for GIF
            if getattr(sim, "current_frame", None) is not None:
                lines = [
                    f"{name}",
                    f"pip:[{pip[0]:+.3f},{pip[1]:+.3f},{pip[2]:+.3f}]",
                    f"d:{delta:.5f} s:{stable_count}/{STABLE_REQUIRED}",
                ]
                annotated = annotate_frame(sim.current_frame, lines)
                gif_frames.append(annotated)

            # If stable long enough, we assume we hit the workspace limit
            if stable_count >= STABLE_REQUIRED:
                print(f"Reached corner {name} at {pip}")
                return list(pip)

            last_pip = pip

        print(f"Reached max steps for {name}, final pos: {pip}")
        return list(pip)

    # 8 diagonal directions to push toward workspace limits
    directions = {
        "x_min_y_min_z_min": [-0.3, -0.3, -0.3],
        "x_min_y_min_z_max": [-0.3, -0.3, +0.3],
        "x_min_y_max_z_min": [-0.3, +0.3, -0.3],
        "x_min_y_max_z_max": [-0.3, +0.3, +0.3],
        "x_max_y_min_z_min": [+0.3, -0.3, -0.3],
        "x_max_y_min_z_max": [+0.3, -0.3, +0.3],
        "x_max_y_max_z_min": [+0.3, +0.3, -0.3],
        "x_max_y_max_z_max": [+0.3, +0.3, +0.3],
    }

    true_corners = {}

    # Find all 8 reachable corners
    for name, vel in directions.items():
        true_corners[name] = move_until_stable(vel, name)

    # Print final corners
    print("\n\n WORK ENVELOPE CORNERS")
    for name, pos in true_corners.items():
        print(name, pos)

    # Save final GIF
    imageio.mimsave(OUTPUT_GIF, gif_frames, fps=FPS)
    print(f"\nGIF saved as {OUTPUT_GIF}")


if __name__ == "__main__":
    main()
