#!/usr/bin/env python3
import numpy as np
import math
from perception import init_realsense, detect
from collision import CollisionChecker
from rrt import RRTPlanner
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

import pid_angle_control
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

if __name__ == "__main__":
    pipeline, align = init_realsense()
    start_q = np.array([357, 20, 150, 272, 320, 273]) / 180 * np.pi
    goal_q = np.array([0, 344, 75, 0, 300, 0]) / 180 * np.pi
    joint_limits = [(0, 2 * np.pi)] * 6

    dh_params = [

        # [theta,      d,      a,          alpha]
        [0,           0,       0.2433,     0],          # Joint 1
        [math.pi/2,   0,       0.01,       math.pi/2],  # Joint 2
        [math.pi,     0.28,    0,          math.pi/2],  # Joint 3
        [math.pi/2,   0,       0.245,      math.pi/2],  # Joint 4
        [math.pi/2,   0,       0.057,      0],          # Joint 5
        [-math.pi/2,  0,       0.235,     -math.pi/2]   # Joint 6
    ]
    robot = NLinkArm(dh_params)
    
    collision_checker = CollisionChecker(dh_params)

    rrt = RRTPlanner(robot, joint_limits)
    optimizer = BSplineOptimizer(robot, degree=3, num_points=20)

    obj, goal = detect(pipeline, align)

    path = rrt.plan(start_q, goal_q)

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
