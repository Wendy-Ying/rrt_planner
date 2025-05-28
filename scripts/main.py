#!/usr/bin/env python3
import numpy as np
import math
from rrt import RRTStar
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

if __name__ == "__main__":
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
    
    # Create collision checker
    from collision import CollisionChecker
    collision_checker = CollisionChecker(dh_params)
    rrt_star = RRTStar(joint_limits, dh_params, collision_checker)
    optimizer = BSplineOptimizer(degree=3, num_points=200)

    path = rrt_star.plan(start_q, goal_q)  

    if path:
        # 优化路径
        smooth_path = optimizer.optimize(path)
        
        # 显示路径信息
        for i, q in enumerate(smooth_path):
            print(f"Step {i}: {q}")
        
        # 执行优化后的路径
        try:
            import pid_angle_control
            import utilities
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            
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
        
        # 显示轨迹可视化
        plot_cartesian_trajectory(smooth_path, robot)
