import numpy as np

class NLinkArm:
    def __init__(self, dh_params):
        """
        dh_params: List of DH parameters in the order [alpha, a, d, theta]
        """
        self.dh_params = dh_params
        self.num_joints = len(dh_params)

    def dh_transform(self, alpha, a, d, theta):
        """
        Construct the transformation matrix using the standard DH convention:
        alpha: twist angle
        d: link offset
        a: link length
        theta: joint angle
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)

        return np.array([
            [ct, -st, 0, a],
            [st*ca, ct*ca, -sa, -d*sa],
            [st*sa, ct*sa,  ca,  d*ca],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """
        joint_angles: The list of joint variables (offset to each theta in the table)
        Assumes all joints are revolute. If some are prismatic, you can extend this.
        """
        assert len(joint_angles) == self.num_joints, "Joint angle count must match DH parameters"

        T = np.eye(4)
        for i, (alpha, a, d, theta) in enumerate(self.dh_params):
            T_i = self.dh_transform(alpha, a, d, theta + joint_angles[i])
            T = T @ T_i
        return T[:3, 3]  # Return the position (x, y, z) of the end-effector

    def get_joint_positions(self, q):
        """Return the 3D positions of each joint (including base and end-effector)."""
        positions = [np.array([0.0, 0.0, 0.0])]
        T = np.eye(4)

        for i in range(self.num_joints):
            alpha, a, d, theta_offset = self.dh_params[i]
            theta = theta_offset + q[i]  # apply joint variable offset
            T_i = self.dh_transform(alpha, a, d, theta)
            T = T @ T_i
            positions.append(T[:3, 3])  # extract position
        return np.array(positions)

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