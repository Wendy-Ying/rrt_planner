import numpy as np
from scipy.interpolate import splprep, splev

class BSplineOptimizer:
    def __init__(self, robot, degree=3, num_points=20):
        self.arm = robot
        self.degree = degree
        self.num_points = num_points

    def optimize(self, path):
        # get joint positions
        ee_points = [self.arm.forward_kinematics(q) for q in path]
        ee_points = np.array(ee_points).T

        # bspline interpolation
        tck, _ = splprep(ee_points, s=0, k=min(self.degree, len(path) - 1))
        u_fine = np.linspace(0, 1, self.num_points)
        smoothed_ee = splev(u_fine, tck)
        smoothed_ee = np.array(smoothed_ee).T

        # inverse kinematics for each point
        q_list = []
        q_seed = path[0]
        for ee_target in smoothed_ee:
            q_new = self.arm.inverse_kinematics(ee_target, q_init=q_seed)
            if q_new is None:
                print("IK failed at point", ee_target)
                continue
            q_list.append(q_new)
            q_seed = q_new

        return q_list
