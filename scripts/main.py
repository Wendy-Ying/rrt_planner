import numpy as np
from perception import init_realsense, detect
from rrt import RRTPlanner
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

if __name__ == "__main__":
    pipeline, align = init_realsense()
    start_q = np.array([357, 20, 150, 272, 320, 273]) / 180 * np.pi
    goal_q = np.array([0, 344, 75, 0, 300, 0]) / 180 * np.pi
    joint_limits = [(0, 2 * np.pi)] * 6

    dh_params = [
        [0, 0, 243.3/1000, 0],
        [np.pi/2, 0, 10/1000, np.pi/2],
        [np.pi, 280/1000, 0, np.pi/2],
        [np.pi/2, 0, 245/1000, np.pi/2],
        [np.pi/2, 0, 57/1000, 0],
        [-np.pi/2, 0, 235/1000, -np.pi/2]
    ]
    robot = NLinkArm(dh_params)

    rrt = RRTPlanner(robot, joint_limits)
    optimizer = BSplineOptimizer(robot, degree=3, num_points=20)

    obj, goal = detect(pipeline, align)

    path = rrt.plan(start_q, goal_q)

    if path:
        smooth_path = optimizer.optimize(path)
        for i, q in enumerate(smooth_path):
            print(f"Step {i}: {q}")
        plot_cartesian_trajectory(smooth_path, robot)
    
    pipeline.stop()