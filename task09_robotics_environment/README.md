# Task 09 – Robotics Environment Interaction
**Opentrons OT-2 Digital Twin – Workspace Exploration**

This task explores how to control a simulated **Opentrons OT-2** robot using velocity commands, observe its behavior, and determine the pipette’s **3D work envelope**.  
The robot is controlled through the provided `sim_class.py` PyBullet environment.

---

## **Objectives**

- Move the robot’s pipette around the workspace  
- Understand how the robot moves in x, y, and z  
- Find the *maximum* and *minimum* positions the pipette can reach  
- Move the pipette to all **8 corners** of the workspace  
- Save a GIF showing how the robot moved  
 
The robot is commanded through `sim_class.py`, and a custom script (`corners.py`) moves the pipette to all **eight extreme corners** of its reachable workspace while capturing frames and generating a GIF.

---

##  Project Structure

```
task09_robotics_environment/
│
├── meshes/                     # OT-2 3D model files
├── textures/                   # Plate and environment textures
│
├── sim_class.py                # Simulation engine (provided)
├── corners.py                  # Script that finds the 8 workspace corners
├── corners.gif                 # GIF of robot moving to all corners
├── ot_2_simulation_v6.urdf     # OT-2 model
├── custom.urdf                 # Specimen/plate URDF
└── README.md                   # (this file)
```

---

##  Environment Setup

### 1. Install Dependencies

Create/activate your conda or venv environment, then install:

```
pip install pybullet
pip install imageio
pip install numpy
pip install opencv-python

```

---

## How to Run the Simulation
To explore the robot’s work envelope and record the GIF:

```
python corners.py
```

This script will:
- Start the PyBullet OT-2 simulation
- Move the pipette in 8 diagonal directions
- Detect when each limit is reached
- Record pipette positions
- Save all captured frames
- Generate `corners.gif`

---

##  GIF Output
The animation is generated using:

```
imageio.mimsave("corners.gif", gif_frames, fps=20)
```
The resulting GIF (`corners.gif`) shows the pipette moving smoothly to all 8 corners of its workspace.  
The GIF includes an on-screen text overlay showing the target corner, pipette position, per-step movement (velocity proxy), and stability state for each simulation step.

---
##  Determined Work Envelope (8 Corners)
Measured directly from the simulation:

| Corner | X | Y | Z |
|--------|--------|--------|--------|
| **x_min_y_min_z_min** | -0.187  | -0.1706 | 0.1195 |
| **x_min_y_min_z_max** | -0.187  | -0.1706 | 0.2895 |
| **x_min_y_max_z_min** | -0.187  | 0.2195  | 0.1695 |
| **x_min_y_max_z_max** | -0.187  | 0.2195  | 0.2895 |
| **x_max_y_min_z_min** | 0.2531 | -0.1705 | 0.1695 |
| **x_max_y_min_z_max** | 0.2530 | -0.1706 | 0.2896 |
| **x_max_y_max_z_min** | 0.2530 | 0.2196  | 0.1695 |
| **x_max_y_max_z_max** | 0.2530 | 0.2196  | 0.2896 |

These values form the OT-2 pipette’s **reachable 3D workspace**.


Method for Determining Corners

I used velocity-based diagonal movement rather than target-position commands.
Why diagonal velocity works better
The OT-2 arm geometry means moving in X can also slightly change Z
Velocity control avoids issues with inverse kinematics
The robot naturally decelerates near physical joint limits
The simulation becomes unstable if you try to force exact XYZ coordinates
A movement is considered “complete” when the pipette position stops changing for several frames.

---

## Key Findings

1. Minor variation in Z-minimum values across corners is due to geometric and collision constraints of the OT-2 model; the lowest reachable Z depends on the X/Y position rather than forming a perfectly flat plane

2. Joint limits cause natural deceleration

The pipette slows as it approaches its limits, allowing detection based on stability.

3. The workspace is not a perfect cube

Changing one axis influences others due to the robot’s kinematics.

4. Direct coordinate targeting is unreliable

Attempting to move directly to extreme values caused:
premature stopping
freezing
PyBullet disconnections
incorrect work envelope measurements

5. Velocity control is ideal for exploration

Diagonal velocity pushes the arm to its real physical boundaries.

6. Pipette tip is the origin of measurements

Values are returned for the pipette tip, not the robot base.












