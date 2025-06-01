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
    # Define positions as numpy arrays
    obj = np.array([0.18343688, 0.13385455, -0.01195908], dtype=float)      # Object position
    obstacle = np.array([0.58765264, 0.10652312, 0.00915751], dtype=float)  # Obstacle position
    goal = np.array([0.38018093, 0.13177095, 0.08623872], dtype=float)     # Goal position

    # Robot parameters
    joint_limits = np.array([(205, 150), (205, 150), (210, 150), 
                            (210, 145), (215, 140), (210, 150)], dtype=float) / 180 * np.pi
    dh_params = np.array([
        [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
        [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
        [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
        [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
    ], dtype=float)

    # Initialize robot and planners
    robot = NLinkArm(dh_params, joint_limits)
    collision_checker = CollisionChecker(dh_params)
    
    # Set up obstacle box (15cm×15cm×15cm)
    box_margin = 0.075 # 7.5cm margin
    height = 0.15      # 15cm height
    boxes_3d = np.array([
        float(obstacle[0] - box_margin/3),  # x_min 
        float(obstacle[1] - box_margin),  # y_min
        float(obstacle[2]),               # z_min
        float(obstacle[0] + box_margin/3),  # x_max
        float(obstacle[1] + box_margin),  # y_max 
        float(obstacle[2] + height)       # z_max
    ])

    # Calculate and print distances from object and goal to obstacle
    obj_to_obstacle = np.linalg.norm([obj[0] - obstacle[0], obj[1] - obstacle[1]])
    goal_to_obstacle = np.linalg.norm([goal[0] - obstacle[0], goal[1] - obstacle[1]])
    print(f"Distance from object to obstacle: {obj_to_obstacle:.3f}m")
    print(f"Distance from goal to obstacle: {goal_to_obstacle:.3f}m")
    # Initialize RRT planner
    rrt = RRTPlanner(robot, joint_limits, collision_checker, boxes_3d)

    # Initial robot position (convert to radians and ensure float type)
    init = np.array([357, 21, 150, 272, 320, 273], dtype=float) / 180.0 * np.pi

    # Calculate grasp positions with offsets (ensure float type)
    grasp_offset = np.array([0, 0, 0], dtype=float)
    z_offset = np.array([0, 0, 0.05], dtype=float)

    # Calculate object grasp position
    obj_grasp = obj + grasp_offset

    # Setup connection
    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)

            # Generate and execute path to object
            print("Planning path to object...")
            path1 = rrt.plan(init, obj_grasp)  # Plan will automatically choose best strategy
            if path1 is None:
                print("Failed to plan path to object")
                return

            print("Executing path to object...")
            path1 = np.array(path1, dtype=float)  # Ensure path is float array
            success = pid_angle_control.execute_path(base, path1)
            if not success:
                print("Failed to reach object")
                return

            # print("\nObject reached, closing gripper...")
            # pid_angle_control.send_gripper_command(base, 0.3)  
            # time.sleep(3.0)  # Wait for gripper to close

            # Plan and execute motion to goal position
            print("\nPlanning path to goal...")
            goal_pos = goal + z_offset  # Add z-offset for placing
            path2 = rrt.plan(obj_grasp, goal_pos)  # Plan will automatically handle obstacles
            if path2 is None:
                print("Failed to plan path to goal")
                return

            print("Executing path to goal...")
            path2 = np.array(path2, dtype=float)  # Ensure path is float array
            success = pid_angle_control.execute_path(base, path2)
            if not success:
                print("Failed to reach goal")
                return

            # print("\nGoal reached, opening gripper...")
            # pid_angle_control.send_gripper_command(base, 0.9)  # Open gripper (90% open)
            # time.sleep(1.0)  # Wait for gripper to open

            print("\nMotion completed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
