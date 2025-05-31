#!/usr/bin/env python3
import numpy as np
import math
from perception import init_realsense, detect
from collision import CollisionChecker
from obstacle import ObstacleDetector
from rrt import RRTPlanner
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
    print(f"box: {boxes_3d}")

    rrt = RRTPlanner(robot, joint_limits, collision_checker, boxes_3d)
    optimizer = BSplineOptimizer(robot, degree=3, num_points=5)

    obj, goal = detect(pipeline, align)
    init=np.array([357, 21, 150, 272, 320, 273]) / 180 * np.pi
    print(f"Detected object: {obj}")
    print(f"Goal position: {goal}")

    args = utilities.parseConnectionArguments()

    # path = rrt.plan(start_q, goal_q)
    path = rrt.plan(init, obj)
    

    if path:
        smooth_path = optimizer.optimize(path)
        for i, q in enumerate(smooth_path):
            print(f"Step {i}: {q}")
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            success = pid_angle_control.execute_path(base, smooth_path)
            pid_angle_control.send_gripper_command(base, 0.1)
            if not success:
                print("Path execution failed")
            else:
                print("Path execution completed successfully")
    
    
    plot_cartesian_trajectory(smooth_path, robot)

    path = rrt.plan(obj, goal)
    if path:
        smooth_path = optimizer.optimize(path)
        for i, q in enumerate(smooth_path):
            print(f"Step {i}: {q}")
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            success = pid_angle_control.execute_path(base, smooth_path)
            pid_angle_control.send_gripper_command(base, 1.0)
        if not success:
            print("Path execution failed")
        else:
            print("Path execution completed successfully")

    plot_cartesian_trajectory(smooth_path, robot)
    
    pipeline.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")