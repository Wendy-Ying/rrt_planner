#!/usr/bin/env python3
import numpy as np
import time
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
import pid_angle_control
import utilities
from collision import CollisionChecker
from rrt import RRTPlanner
from kinematics import NLinkArm

def main():
    # Define positions
    obj = np.array([0.166, 0.148, 0])  # Object position
    obstacle = np.array([0.367, 0.11, 0])  # Obstacle position
    goal = np.array([0.552, 0.131, 0])  # Goal position

    # Robot parameters
    joint_limits = np.array([(205, 150), (205, 150), (210, 150), 
                            (210, 145), (215, 140), (210, 150)]) / 180 * np.pi
    dh_params = [
        [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
        [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
        [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
        [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
    ]

    # Initialize robot and planners
    robot = NLinkArm(dh_params, joint_limits)
    collision_checker = CollisionChecker(dh_params)
    
    # Set up obstacle box (15cm×15cm×15cm)
    boxes_3d = np.array([
        obstacle[0]-0.075,  # x_min 
        obstacle[1]-0.075,  # y_min
        obstacle[2],        # z_min
        obstacle[0]+0.075,  # x_max
        obstacle[1]+0.075,  # y_max 
        obstacle[2]+0.15    # z_max
    ])

    # Initialize RRT planner
    rrt = RRTPlanner(robot, joint_limits, collision_checker, boxes_3d)

    # Initial robot position (in radians)
    init = np.array([357, 21, 150, 272, 320, 273]) / 180 * np.pi

    # Calculate object grasp position (with 5cm z-offset for grasping)
    obj_grasp = obj + np.array([0, 0, 0])

    # Setup connection
    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)

            # Generate and execute path to object
            print("Planning path to object...")
            path1 = rrt.plan(init, obj_grasp)
            if path1 is None:
                print("Failed to plan path to object")
                return

            print("Executing path to object...")
            success = pid_angle_control.execute_path(base, path1)
            if not success:
                print("Failed to reach object")
                return

            print("\nObject reached, closing gripper...")
            pid_angle_control.send_gripper_command(base, 0.3)  
            time.sleep(3.0)  # Wait for gripper to close

            # Generate and execute path to goal
            print("\nPlanning path to goal...")
            goal_pos = goal + np.array([0, 0, 0.05])  # Add z-offset for placing
            path2 = rrt.plan(obj_grasp, goal_pos)
            if path2 is None:
                print("Failed to plan path to goal")
                return

            print("Executing path to goal...")
            success = pid_angle_control.execute_path(base, path2)
            if not success:
                print("Failed to reach goal")
                return

            print("\nGoal reached, opening gripper...")
            pid_angle_control.send_gripper_command(base, 0.9)  # Open gripper
            time.sleep(1.0)  # Wait for gripper to open

            print("\nMotion completed successfully")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
