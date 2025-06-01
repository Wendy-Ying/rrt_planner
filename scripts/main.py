#!/usr/bin/env python3
import numpy as np
import math
from perception import init_realsense, detect
from collision import CollisionChecker
from obstacle import ObstacleDetector
from rrt import RRTPlanner
from prm import PRMPlanner
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

import pid_angle_control
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

def main():
    pipeline, align = init_realsense()
    
    joint_limits = np.array([(205, 150), (205, 150), (210, 150), (210, 145), (215, 140), (210, 150)]) / 180 * np.pi

    dh_params = [
        [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
        [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
        [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
        [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
    ]

    robot = NLinkArm(dh_params, joint_limits)
    
    collision_checker = CollisionChecker(dh_params)
    detector = ObstacleDetector(pipeline, align)
    boxes_3d = detector.get_world_bounding_boxes(visualize=False)

    # Get positions and ensure they are numpy arrays
    obj, goal, obstacle = detect(pipeline, align)
    obj = np.array(obj)
    goal = np.array(goal)
    obstacle = np.array(obstacle)

    # Convert angles to radians using numpy operations
    init = np.array([357, 21, 150, 272, 320, 273]) / 180 * np.pi
    final = np.array([0, 343, 75, 0, 300, 0]) / 180 * np.pi

    # Calculate grasp and offset positions
    obj_grasp = obj + np.array([0, 0.02, 0])
    obj_offset = obj + np.array([0, 0, 0.05])

    print(f"Detected object: {obj}")
    print(f"Goal position: {goal}")
    print(f"Obstacle position: {obstacle}")

    # Create obstacle bounding box with margin
    margin = 0.02
    boxes_3d = np.array([
        float(obstacle[0] - margin),  # x_min
        float(obstacle[1] - margin),  # y_min
        float(obstacle[2]),           # z_min
        float(obstacle[0] + margin),  # x_max
        float(obstacle[1] + margin),  # y_max
        float(obstacle[2] + 0.2)      # z_max
    ])

    rrt = RRTPlanner(robot, joint_limits, collision_checker, boxes_3d)
    prm = PRMPlanner(robot, joint_limits, boxes_3d)
    optimizer = BSplineOptimizer(robot, degree=3, num_points=5)

    args = utilities.parseConnectionArguments()

    # path = rrt.plan(init, obj_grasp)
    

    # if path:
    #     smooth_path = optimizer.optimize(path)
    #     for i, q in enumerate(smooth_path):
    #         print(f"Step {i}: {q}")
    #     with utilities.DeviceConnection.createTcpConnection(args) as router:
    #         base = BaseClient(router)
    #         pid_angle_control.send_gripper_command(base, 0.1)
    #         success = pid_angle_control.execute_path(base, smooth_path)
    #         pid_angle_control.send_gripper_command(base, 1)
    #         if not success:
    #             print("Path execution failed")
    #         else:
    #             print("Path execution completed successfully")
    
    
    # plot_cartesian_trajectory(smooth_path, robot)

    path = rrt.plan(obj_offset, goal)
    # path = prm.plan(obj_offset, goal)
    if path:
        smooth_path = optimizer.optimize(path)
        for i, q in enumerate(path):
            print(f"Step {i}: {q}")
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            success = pid_angle_control.execute_path(base, path)
            pid_angle_control.send_gripper_command(base, 0)
        if not success:
            print("Path execution failed")
        else:
            print("Path execution completed successfully")

    # plot_cartesian_trajectory(smooth_path, robot)

    # path = rrt.plan(goal, final)
    # if path:
    #     smooth_path = optimizer.optimize(path)
    #     for i, q in enumerate(smooth_path):
    #         print(f"Step {i}: {q}")
    #     with utilities.DeviceConnection.createTcpConnection(args) as router:
    #         base = BaseClient(router)
    #         success = pid_angle_control.execute_path(base, smooth_path)
    #         if not success:
    #             print("Path execution failed")
    #         else:
    #             print("Path execution completed successfully")

    pipeline.stop()

if __name__ == "__main__":
    # try:
    #     main()
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    main()
