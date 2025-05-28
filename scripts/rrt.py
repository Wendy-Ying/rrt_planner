import numpy as np
from collision import is_in_collision

class RRTPlanner:
    def __init__(self, robot, joint_limits, step_size=0.5, max_iter=500, goal_sample_rate=0.3, n_steps=10):
        self.robot = robot
        self.joint_limits = np.array(joint_limits)
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.n_steps = n_steps

    def sample_q(self, end_q):
        if np.random.rand() < self.goal_sample_rate:
            return end_q
        lows, highs = self.joint_limits[:, 0], self.joint_limits[:, 1]
        return np.random.uniform(lows, highs)

    def ee_dist(self, q1, q2):
        p1 = self.robot.forward_kinematics(q1)
        p2 = self.robot.forward_kinematics(q2)
        return np.linalg.norm(p1 - p2)

    def nearest_index(self, tree, q_rand):
        dists = [self.ee_dist(node['q'], q_rand) for node in tree]
        return int(np.argmin(dists))

    def is_within_limits(self, q):
        return np.all((q >= self.joint_limits[:, 0]) & (q <= self.joint_limits[:, 1]))

    def collision_check_line(self, q1, q2):
        for alpha in np.linspace(0, 1, self.n_steps):
            q_interp = q1 + alpha * (q2 - q1)
            if not self.is_within_limits(q_interp):
                return True
            if is_in_collision(q_interp):
                return True
        return False

    def steer(self, q_near, q_rand):
        direction = q_rand - q_near
        length = np.linalg.norm(direction)
        if length == 0:
            return None
        if length <= self.step_size:
            if self.collision_check_line(q_near, q_rand):
                return None
            return q_rand
        
        step_vector = direction / length * self.step_size
        q_new = q_near + step_vector
        if not self.is_within_limits(q_new) or is_in_collision(q_new):
            return None
        return q_new

    def shortcut_path(self, path):
        if len(path) <= 2:
            return path
        
        path = path.copy()
        max_trials = 100
        trial = 0

        while trial < max_trials:
            if len(path) <= 2:
                break
            i = np.random.randint(0, len(path) - 2)
            j = np.random.randint(i + 2, len(path))
            if not self.collision_check_line(path[i], path[j]):
                path = path[:i+1] + path[j:]
            trial += 1
        return path

    def plan(self, start_q, end_q):
        start_q = np.array(start_q)
        end_q = np.array(end_q)

        tree = [{'q': start_q, 'parent': None}]

        for _ in range(self.max_iter):
            q_rand = self.sample_q(end_q)
            idx_near = self.nearest_index(tree, q_rand)
            q_near = tree[idx_near]['q']
            
            q_new = self.steer(q_near, q_rand)
            if q_new is None:
                continue

            tree.append({'q': q_new, 'parent': idx_near})

            if self.ee_dist(q_new, end_q) < self.step_size:
                if self.collision_check_line(q_new, end_q):
                    continue
                tree.append({'q': end_q, 'parent': len(tree) - 1})
                path = []
                idx = len(tree) - 1
                while idx is not None:
                    path.append(tree[idx]['q'])
                    idx = tree[idx]['parent']
                path.reverse()

                path = self.shortcut_path(path)

                return path

        print("Failed to find a path.")
        return None
