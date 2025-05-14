import numpy as np

class NLinkArm:
    def __init__(self, dh_params):
        self.dh_params = dh_params

    def dh_transform(self, theta, d, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                            0,                           1]
        ])

    def forward_kinematics(self, q):
        T = np.eye(4)
        for i in range(len(q)):
            theta = q[i] + self.dh_params[i][0]
            d = self.dh_params[i][1]
            a = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            T_i = self.dh_transform(theta, d, a, alpha)
            T = np.dot(T, T_i)
        return T[:3, 3]  # Return only position
