import numpy as np
from scipy.interpolate import splprep, splev

class BSplineOptimizer:
    def __init__(self, degree=3, num_points=100):
        """
        degree: B样条的阶数，默认3为立方B样条
        num_points: 采样的点数，用于平滑后路径
        """
        self.degree = degree
        self.num_points = num_points

    def optimize(self, path):
        """
        path: List of configurations (e.g., joint angles or positions)
        return: Smoothed path as a list of points
        """
        path = np.array(path).T  # shape (n_dof, N)
        n_dim, n_pts = path.shape
        if n_pts <= self.degree:
            return path.T  # 不足以拟合B样条，返回原路径

        # 拟合B样条
        tck, _ = splprep(path, s=0, k=self.degree)

        # 按照均匀参数采样平滑路径
        u_fine = np.linspace(0, 1, self.num_points)
        smoothed_path = splev(u_fine, tck)
        return np.array(smoothed_path).T  # shape (N, n_dof)
