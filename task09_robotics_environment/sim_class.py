import os

p = None
if os.environ.get("CLEARML_TASK_ID") or os.environ.get("CLEARML_WORKER_ID"):
    import pybullet as p

import time
import pybullet_data
import math
import logging
import os
import random

#logging.basicConfig(level=logging.INFO)

class Simulation:
    def __init__(self, num_agents, render=True, rgb_array=False):
        self.render = render
        self.rgb_array = rgb_array

        # --------- ADDED FOR RGB CAPTURE ---------
        self.current_frame = None  # buffer for most recent camera frame
        # ------------------------------------------

        if render:
            mode = p.GUI # for graphical version
        else:
            mode = p.DIRECT # for non-graphical version

        # Set up the simulation
        self.physicsClient = p.connect(mode)
        # Hide the default GUI components
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)

        texture_list = os.listdir("textures")
        random_texture = random.choice(texture_list[:-1])
        random_texture_index = texture_list.index(random_texture)
        self.plate_image_path = f'textures/_plates/{os.listdir("textures/_plates")[random_texture_index]}'
        self.textureId = p.loadTexture(f'textures/{random_texture}')

        cameraDistance = 1.1*(math.ceil((num_agents)**0.3))
        cameraYaw = 90  
        cameraPitch = -35  
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents**0.5)/2)+0.5, 0.1]

        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        self.baseplaneId = p.loadURDF("plane.urdf")

        self.pipette_offset = [0.073, 0.0895, 0.0895]

        self.pipette_positions = {}

        self.create_robots(num_agents)

        self.sphereIds = []

        self.droplet_positions = {}

    # method to create n robots in a grid pattern
    def create_robots(self, num_agents):
        spacing = 1  
        grid_size = math.ceil(num_agents ** 0.5) 

        self.robotIds = []
        self.specimenIds = []
        agent_count = 0  

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count < num_agents:
                    position = [-spacing * i, -spacing * j, 0.03]
                    robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0,0,0,1],
                                        flags=p.URDF_USE_INERTIA_FROM_FILE)
                    start_position, start_orientation = p.getBasePositionAndOrientation(robotId)
                    p.createConstraint(parentBodyUniqueId=robotId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=start_position,
                                    childFrameOrientation=start_orientation)

                    offset = [0.18275-0.00005, 0.163-0.026, 0.057]
                    position_with_offset = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                    rotate_90 = p.getQuaternionFromEuler([0, 0, -math.pi/2])
                    planeId = p.loadURDF("custom.urdf", position_with_offset, rotate_90)
                    p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)
                    spec_position, spec_orientation = p.getBasePositionAndOrientation(planeId)

                    p.createConstraint(parentBodyUniqueId=planeId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=spec_position,
                                    childFrameOrientation=spec_orientation)

                    p.changeVisualShape(planeId, -1, textureUniqueId=self.textureId)

                    self.robotIds.append(robotId)
                    self.specimenIds.append(planeId)

                    agent_count += 1  

                    pipette_position = self.get_pipette_position(robotId)
                    self.pipette_positions[f'robotId_{robotId}'] = pipette_position

    def get_pipette_position(self, robotId):
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        robot_position = list(robot_position)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]

        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2]

        pipette_position = [robot_position[0]+x_offset, robot_position[1]+y_offset, robot_position[2]+z_offset]
        return pipette_position

    def reset(self, num_agents=1):
        for specimenId in self.specimenIds:
            p.changeVisualShape(specimenId, -1, textureUniqueId=-1)

        for robotId in self.robotIds:
            p.removeBody(robotId)
            self.robotIds.remove(robotId)

        for specimenId in self.specimenIds:
            p.removeBody(specimenId)
            self.specimenIds.remove(specimenId)

        for sphereId in self.sphereIds:
            p.removeBody(sphereId)
            self.sphereIds.remove(sphereId)

        self.pipette_positions = {}
        self.sphereIds = []
        self.droplet_positions = {}

        self.create_robots(num_agents)

        return self.get_states()

    def run(self, actions, num_steps=1):
        start = time.time()
        n = 100

        for i in range(num_steps):
            self.apply_actions(actions)
            p.stepSimulation()

            # --------- ADDED FOR RGB CAPTURE ---------
            if self.rgb_array:
                camera_pos = [1, 0, 1]
                camera_target = [-0.3, 0, 0]
                up_vector = [0, 0, 1]
                fov = 50
                aspect = 320/240

                w, h, rgbImg, _, _ = p.getCameraImage(
                    width=320,
                    height=240,
                    viewMatrix=p.computeViewMatrix(camera_pos, camera_target, up_vector),
                    projectionMatrix=p.computeProjectionMatrixFOV(fov, aspect, 0.1, 100.0)
                )

                self.current_frame = rgbImg
            # ------------------------------------------

            for specimenId, robotId in zip(self.specimenIds, self.robotIds):
                self.check_contact(robotId, specimenId)

            if self.render:
                time.sleep(1./240.)

        return self.get_states()
    
    def apply_actions(self, actions):
        for i in range(len(self.robotIds)):
            p.setJointMotorControl2(self.robotIds[i], 0, p.VELOCITY_CONTROL, targetVelocity=-actions[i][0], force=500)
            p.setJointMotorControl2(self.robotIds[i], 1, p.VELOCITY_CONTROL, targetVelocity=-actions[i][1], force=500)
            p.setJointMotorControl2(self.robotIds[i], 2, p.VELOCITY_CONTROL, targetVelocity=actions[i][2], force=800)
            if actions[i][3] == 1:
                self.drop(robotId=self.robotIds[i])

    def drop(self, robotId):
        robot_position = p.getBasePositionAndOrientation(robotId)[0]
        robot_position = list(robot_position)
        joint_states = p.getJointStates(robotId, [0, 1, 2])
        robot_position[0] -= joint_states[0][0]
        robot_position[1] -= joint_states[1][0]
        robot_position[2] += joint_states[2][0]

        x_offset = self.pipette_offset[0]
        y_offset = self.pipette_offset[1]
        z_offset = self.pipette_offset[2]-0.0015

        specimen_position = p.getBasePositionAndOrientation(self.specimenIds[0])[0]

        sphereRadius = 0.003  
        sphereColor = [1, 0, 0, 0.5]  
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius, rgbaColor=sphereColor)

        collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=sphereRadius)
        sphereBody = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=visualShapeId, baseCollisionShapeIndex=collision)

        droplet_position = [robot_position[0]+x_offset, robot_position[1]+y_offset, robot_position[2]+z_offset]

        p.resetBasePositionAndOrientation(sphereBody, droplet_position, [0, 0, 0, 1])

        self.sphereIds.append(sphereBody)
        self.dropped = True

        return droplet_position

    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            raw_joint_states = p.getJointStates(robotId, [0, 1, 2])

            joint_states = {}
            for i, joint_state in enumerate(raw_joint_states):
                joint_states[f'joint_{i}'] = {
                    'position': joint_state[0],
                    'velocity': joint_state[1],
                    'reaction_forces': joint_state[2],
                    'motor_torque': joint_state[3]
                }

            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            robot_position = list(robot_position)

            robot_position[0] -= raw_joint_states[0][0]
            robot_position[1] -= raw_joint_states[1][0]
            robot_position[2] += raw_joint_states[2][0]

            pipette_position = [
                robot_position[0] + self.pipette_offset[0],
                robot_position[1] + self.pipette_offset[1],
                robot_position[2] + self.pipette_offset[2]
            ]
            pipette_position = [round(num, 4) for num in pipette_position]

            states[f'robotId_{robotId}'] = {
                "joint_states": joint_states,
                "robot_position": robot_position,
                "pipette_position": pipette_position
            }

        return states
    
    def check_contact(self, robotId, specimenId):
        for sphereId in self.sphereIds:
            contact_points_specimen = p.getContactPoints(sphereId, specimenId)
            contact_points_robot = p.getContactPoints(sphereId, robotId)

            if contact_points_specimen:
                p.setCollisionFilterPair(sphereId, specimenId, -1, -1, enableCollision=0)
                sphere_position, sphere_orientation = p.getBasePositionAndOrientation(sphereId)
                p.createConstraint(parentBodyUniqueId=sphereId,
                                    parentLinkIndex=-1,
                                    childBodyUniqueId=-1,
                                    childLinkIndex=-1,
                                    jointType=p.JOINT_FIXED,
                                    jointAxis=[0, 0, 0],
                                    parentFramePosition=[0, 0, 0],
                                    childFramePosition=sphere_position,
                                    childFrameOrientation=sphere_orientation)

                if f'specimenId_{specimenId}' in self.droplet_positions:
                    self.droplet_positions[f'specimenId_{specimenId}'].append(sphere_position)
                else:
                    self.droplet_positions[f'specimenId_{specimenId}'] = [sphere_position]

            if contact_points_robot:
                p.removeBody(sphereId)
                self.sphereIds.remove(sphereId)

    def set_start_position(self, x, y, z):
        for robotId in self.robotIds:
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            adjusted_x = x - robot_position[0] - self.pipette_offset[0]
            adjusted_y = y - robot_position[1] - self.pipette_offset[1]
            adjusted_z = z - robot_position[2] - self.pipette_offset[2]

            p.resetJointState(robotId, 0, targetValue=adjusted_x)
            p.resetJointState(robotId, 1, targetValue=adjusted_y)
            p.resetJointState(robotId, 2, targetValue=adjusted_z)

    def get_plate_image(self):
        return self.plate_image_path
    
    def close(self):
        p.disconnect()
