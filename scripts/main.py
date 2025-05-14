import numpy as np
from rrt import RRTStar
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

if __name__ == "__main__":
    start_q = [0, 0, 0, 0, 0, 0]
    goal_q = [np.pi/2, -np.pi/4, np.pi/3, 0, np.pi/6, -np.pi/2]
    joint_limits = [(-np.pi, np.pi)] * 6

    dh_params = [
        [0, 0.1, 0, -np.pi/2],
        [0, 0, 0.5, 0],
        [0, 0, 0.3, 0],
        [0, 0.2, 0, -np.pi/2],
        [0, 0, 0, np.pi/2],
        [0, 0.1, 0, 0]
    ]
    robot = NLinkArm(dh_params)

    rrt_star = RRTStar(start_q, goal_q, joint_limits)
    optimizer = BSplineOptimizer(degree=3, num_points=200)

    path = rrt_star.plan()

    if path:
        smooth_path = optimizer.optimize(path)
        for i, q in enumerate(smooth_path):
            print(f"Step {i}: {q}")
        plot_cartesian_trajectory(smooth_path, robot)