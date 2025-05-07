import numpy as np
from collision import is_in_collision

class Node:
    def __init__(self, config, parent=None):
        self.config = config
        self.parent = parent

class RRT:
    def __init__(self, start, goal, max_iter=1000, step_size=0.1, goal_thresh=0.2):
        self.start = Node(start)
        self.goal = Node(goal)
        self.nodes = [self.start]
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_thresh = goal_thresh
        self.joint_limits = [
            (-np.pi, np.pi),       # Joint 1
            (-np.pi / 2, np.pi / 2), # Joint 2
            (-np.pi, np.pi),       # Joint 3
            (-np.pi, np.pi),       # Joint 4
            (-np.pi, np.pi),       # Joint 5
            (-np.pi, np.pi)        # Joint 6
        ]

    def sample_random_config(self):
        """Sample a random configuration within joint limits"""
        return np.array([np.random.uniform(low, high) for low, high in self.joint_limits])

    def distance(self, q1, q2):
        """Compute Euclidean distance in joint space"""
        return np.linalg.norm(q1 - q2)

    def interpolate(self, q1, q2):
        """Linear interpolation between two joint configs, returning intermediate points"""
        dist = self.distance(q1, q2)
        steps = int(dist / self.step_size)
        if steps < 1:
            return [q2]
        return [q1 + (q2 - q1) * float(i) / steps for i in range(1, steps + 1)]

    def get_nearest_node(self, q_rand):
        """Find the closest node to q_rand in the current tree"""
        return min(self.nodes, key=lambda node: self.distance(node.config, q_rand))

    def extend(self, q_rand):
        """Extend the tree toward the sampled configuration, checking for collisions"""
        nearest = self.get_nearest_node(q_rand)
        for q_new in self.interpolate(nearest.config, q_rand):
            if is_in_collision(q_new):
                return None
        new_node = Node(q_new, nearest)
        self.nodes.append(new_node)
        return new_node

    def plan(self):
        """Main RRT planning loop"""
        for _ in range(self.max_iter):
            q_rand = self.sample_random_config()
            node = self.extend(q_rand)
            if node and self.distance(node.config, self.goal.config) < self.goal_thresh:
                goal_node = Node(self.goal.config, node)
                self.nodes.append(goal_node)
                return self.extract_path(goal_node)
        return None

    def extract_path(self, node):
        """Backtrack from goal to start to extract the planned path"""
        path = []
        while node:
            path.append(node.config)
            node = node.parent
        return path[::-1]
