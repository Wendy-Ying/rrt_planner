import numpy as np
from collision import is_in_collision

class Node:
    def __init__(self, q, parent=None, cost=0.0):
        self.q = q  # Joint angles
        self.parent = parent
        self.cost = cost  # Cost from start to this node

class RRTStar:
    def __init__(self, joint_limits, dh_table,
                 step_size=0.1, max_iter=1000, radius=0.5, use_cartesian_cost=False):
        self.joint_limits = joint_limits  # list of (low, high) for each joint
        self.dh_table = dh_table  # (n_joints x 4) -> [theta, d, a, alpha]
        self.step_size = step_size
        self.max_iter = max_iter
        self.radius = radius
        self.use_cartesian_cost = use_cartesian_cost

    def sample(self):
        if np.random.rand() < 0.1:  # 10% goal bias
            return self.goal
        return np.array([np.random.uniform(low, high) for (low, high) in self.joint_limits])

    def nearest(self, q_rand):
        return min(self.nodes, key=lambda node: np.linalg.norm(node.q - q_rand))

    def steer(self, q_near, q_rand):
        direction = q_rand - q_near
        norm = np.linalg.norm(direction)
        if norm == 0:
            return q_near
        direction = direction / norm
        q_new = q_near + direction * min(self.step_size, norm)
        return self.wrap_angles(q_new)

    def wrap_angles(self, q):
        # Ensure angles are wrapped to [0, 2pi] if applicable
        wrapped = []
        for i, angle in enumerate(q):
            low, high = self.joint_limits[i]
            if high > 2 * np.pi - 0.1:  # consider this a revolute joint
                wrapped.append(angle % (2 * np.pi))
            else:
                wrapped.append(np.clip(angle, low, high))
        return np.array(wrapped)

    def get_near_nodes(self, q_new):
        return [node for node in self.nodes if np.linalg.norm(node.q - q_new) <= self.radius]

    def forward_kinematics(self, q):
        """Compute end effector pose from joint angles using D-H parameters"""
        T = np.eye(4)
        for i in range(len(q)):
            theta, d, a, alpha = q[i], self.dh_table[i][1], self.dh_table[i][2], self.dh_table[i][3]
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            A = np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [0,       sa,      ca,      d],
                [0,        0,       0,      1]
            ])
            T = T @ A
        return T

    def cartesian_distance(self, q1, q2):
        p1 = self.forward_kinematics(q1)[:3, 3]
        p2 = self.forward_kinematics(q2)[:3, 3]
        return np.linalg.norm(p1 - p2)

    def cost(self, from_node, to_q):
        if self.use_cartesian_cost:
            return from_node.cost + self.cartesian_distance(from_node.q, to_q)
        else:
            return from_node.cost + np.linalg.norm(from_node.q - to_q)

    def plan(self, start, goal):
        self.start = Node(start)
        self.goal = np.array(goal)
        self.nodes = [self.start]
        for i in range(self.max_iter):
            q_rand = self.sample()
            nearest_node = self.nearest(q_rand)
            q_new = self.steer(nearest_node.q, q_rand)

            if is_in_collision(q_new):
                continue

            near_nodes = self.get_near_nodes(q_new)

            # Choose best parent
            min_cost = self.cost(nearest_node, q_new)
            best_parent = nearest_node
            for node in near_nodes:
                if not is_in_collision(q_new):
                    c = self.cost(node, q_new)
                    if c < min_cost:
                        best_parent = node
                        min_cost = c

            new_node = Node(q_new, best_parent, min_cost)
            self.nodes.append(new_node)

            # Rewire
            for node in near_nodes:
                if node == best_parent:
                    continue
                c = self.cost(new_node, node.q)
                if c < node.cost and not is_in_collision(node.q):
                    node.parent = new_node
                    node.cost = c

            if np.linalg.norm(q_new - self.goal) < self.step_size:
                print(f"Goal reached in {i} iterations.")
                return self.extract_path(new_node)

        print("Failed to find a path.")
        return None

    def extract_path(self, node):
        path = []
        while node:
            path.append(node.q)
            node = node.parent
        return path[::-1]
