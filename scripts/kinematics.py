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

    def get_joint_positions(self, q):
        """Return the 3D positions of each joint (including base and end-effector)."""
        positions = [np.array([0.0, 0.0, 0.0])]  # start with base
        T = np.eye(4)
        for i in range(len(q)):
            theta = q[i] + self.dh_params[i][0]
            d = self.dh_params[i][1]
            a = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            T_i = self.dh_transform(theta, d, a, alpha)
            T = np.dot(T, T_i)
            pos = T[:3, 3]
            positions.append(pos)
        return np.array(positions)  # shape: (n_joints + 1, 3)

    def inverse_kinematics(self, target_pos, q_init=None, max_iter=100, tol=1e-3, alpha=0.1):
        if q_init is None:
            q = np.zeros(len(self.dh_params))
        else:
            q = np.array(q_init)

        for _ in range(max_iter):
            current_pos = self.forward_kinematics(q)
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                return q

            J = self.jacobian_numerical(q)
            try:
                dq = alpha * np.linalg.pinv(J).dot(error)
            except np.linalg.LinAlgError:
                return None

            q = q + dq

            # q = np.clip(q, self.joint_limits[:,0], self.joint_limits[:,1])

        return None

    def jacobian_numerical(self, q, delta=1e-6):
        n = len(q)
        J = np.zeros((3, n))
        f0 = self.forward_kinematics(q)

        for i in range(n):
            dq = np.zeros(n)
            dq[i] = delta
            f1 = self.forward_kinematics(q + dq)
            J[:, i] = (f1 - f0) / delta

        return J