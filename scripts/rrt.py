import numpy as np
from collision import is_in_collision

class Node:
    def __init__(self, q, parent=None, cost=0.0):
        self.q = q  # Joint angles
        self.parent = parent
        self.cost = cost  # Cost from start to this node

class RRTStar:
    def __init__(self, start, goal, joint_limits, step_size=0.1, max_iter=1000, radius=0.5):
        self.start = Node(np.array(start))
        self.goal = np.array(goal)
        self.joint_limits = joint_limits
        self.step_size = step_size
        self.max_iter = max_iter
        self.radius = radius  # Radius for rewiring
        self.nodes = [self.start]

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
        return q_new

    def get_near_nodes(self, q_new):
        return [node for node in self.nodes if np.linalg.norm(node.q - q_new) <= self.radius]

    def cost(self, from_node, to_q):
        return from_node.cost + np.linalg.norm(from_node.q - to_q)

    def plan(self):
        for i in range(self.max_iter):
            q_rand = self.sample()
            nearest_node = self.nearest(q_rand)
            q_new = self.steer(nearest_node.q, q_rand)

            if is_in_collision(q_new):
                continue

            # Find near nodes within radius
            near_nodes = self.get_near_nodes(q_new)

            # Choose parent with lowest cost
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

            # Rewire: try to connect nearby nodes to new_node if it lowers their cost
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