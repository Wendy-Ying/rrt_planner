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

if __name__ == "__main__":
    pipeline, align = init_realsense()
    start_q = np.array([0, 0, 0, 0, 0, 0]) / 180 * np.pi
    goal_q = np.array([10, 20, 0, 0, 0, 0]) / 180 * np.pi
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
    obstacle_detector = ObstacleDetector(pipeline, align)

    rrt = RRTPlanner(robot, joint_limits, collision_checker, obstacle_detector)
    optimizer = BSplineOptimizer(robot, degree=3, num_points=3)

    obj, goal = detect(pipeline, align)
    print(f"Detected object: {obj}")
    print(f"Goal position: {goal}")

    # path = rrt.plan(start_q, goal_q)
    path = rrt.plan(obj, goal)

    if path:
        smooth_path = optimizer.optimize(path)
        
        for i, q in enumerate(smooth_path):
            print(f"Step {i}: {q}")
        
        try:            
            print("Executing optimized path...")
            args = utilities.parseConnectionArguments()
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, smooth_path)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")
                    
        except Exception as e:
            print(f"Error executing path: {str(e)}")
        
        plot_cartesian_trajectory(smooth_path, robot)
    
    pipeline.stop()
