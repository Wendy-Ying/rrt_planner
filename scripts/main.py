import numpy as np
from perception import init_realsense, detect
from rrt import RRTStar
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

if __name__ == "__main__":
    pipeline, align = init_realsense()
    start_q = np.array([357, 20, 150, 272, 320, 273]) / 180 * np.pi
    goal_q = np.array([0, 344, 75, 0, 300, 0]) / 180 * np.pi
    joint_limits = [(0, 2 * np.pi)] * 6

    dh_params = [
        [0, 0.1, 0, -np.pi/2],
        [0, 0, 0.5, 0],
        [0, 0, 0.3, 0],
        [0, 0.2, 0, -np.pi/2],
        [0, 0, 0, np.pi/2],
        [0, 0.1, 0, 0]
    ]
    robot = NLinkArm(dh_params)

    rrt_star = RRTStar(joint_limits, dh_params)
    optimizer = BSplineOptimizer(degree=3, num_points=200)

    obj, goal = detect(pipeline, align)

    path = rrt_star.plan(start_q, goal_q)

    if path:
        smooth_path = optimizer.optimize(path)
        for i, q in enumerate(smooth_path):
            print(f"Step {i}: {q}")
        plot_cartesian_trajectory(smooth_path, robot)
    
    pipeline.stop()