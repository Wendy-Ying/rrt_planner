import numpy as np

class RRTPlanner:
    def __init__(self, robot, joint_limits, collison_checker, obstacle, step_size=0.1, max_iter=1000, goal_sample_rate=0.4, n_steps=5):
        self.robot = robot
        self.joint_limits = np.array(joint_limits)
        self.collion_checker = collison_checker
        self.boxes_3d = obstacle
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.n_steps = n_steps

    def sample_q(self, end_q, start_q):
        if np.random.rand() < self.goal_sample_rate:
            return end_q
        bias = np.random.rand() * (end_q - start_q)
        candidate = start_q + bias + 0.1 * np.random.randn(len(start_q))
        return np.clip(candidate, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def ee_dist(self, q1, q2):
        p1 = self.robot.forward_kinematics(q1)
        p2 = self.robot.forward_kinematics(q2)
        return np.linalg.norm(p1 - p2)

    def nearest_index(self, tree, q_rand):
        dists = [self.ee_dist(node['q'], q_rand) for node in tree]
        return int(np.argmin(dists))

    def is_within_limits(self, q):
        def normalize_angle(angle):
            return angle % (2 * np.pi)

        def is_angle_within_limit(angle, low, high):
            angle = normalize_angle(angle)
            low = normalize_angle(low)
            high = normalize_angle(high)
            if low <= high:
                return low <= angle <= high
            else:
                return angle >= low or angle <= high

        for i in range(len(q)):
            if not is_angle_within_limit(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1]):
                return False
        return True

    def collision_check_line(self, q1, q2):
        for alpha in np.linspace(0, 1, self.n_steps):
            q_interp = q1 + alpha * (q2 - q1)
            # if not self.is_within_limits(q_interp):
            #     print(f"Joint limits exceeded at {q_interp}")
            #     return True
            if self.obstacle_collision_check(q_interp):
                # print(f"Collision detected with obstacle")
                return True
            if self.cartesian_collision_check(q_interp):
                # print(f"Collision detected with cartesian")
                return True
        return False
    
    def cartesian_collision_check(self, q):
        joint_positions = self.robot.get_joint_positions(q)
        if np.any(joint_positions[:, 2] < -0.2):
            return True
        return False

    def obstacle_collision_check(self, q, num_interpolation_points=10):
        if len(self.boxes_3d) == 0:
            return False

        # 获取7个关节的位置，每个位置是(x, y, z)
        joint_positions = self.robot.get_joint_positions(q)  # shape: (7, 3)

        # 对每一段连杆进行插值（共6段）
        for i in range(len(joint_positions) - 1):
            start = joint_positions[i]
            end = joint_positions[i + 1]

            # 插值点，包括起点和终点
            for alpha in np.linspace(0, 1, num_interpolation_points):
                point = (1 - alpha) * start + alpha * end  # shape: (3,)

                # 检查这个插值点是否落在任意一个障碍物中
                for x_min, y_min, z_min, x_max, y_max, z_max in self.boxes_3d:
                    if (x_min <= point[0] <= x_max and
                        y_min <= point[1] <= y_max and
                        z_min <= point[2] <= z_max):
                        return True  # 碰撞
        return False  # 无碰撞


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
        if not self.is_within_limits(q_new) or self.collion_checker.self_collision(q_new):
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

        # Handle 3D input (xyz position)
        if start_q.shape[0] == 3:
            ik_start = self.robot.inverse_kinematics(start_q)
            print(f"IK start: {ik_start}")
            if ik_start is None:
                raise ValueError("IK failed for start position")
            start_q = np.array(ik_start)

        if end_q.shape[0] == 3:
            ik_end = self.robot.inverse_kinematics(end_q)
            print(f"IK end: {ik_end}")
            if ik_end is None:
                raise ValueError("IK failed for end position")
            end_q = np.array(ik_end)

        tree = [{'q': start_q, 'parent': None}]

        for _ in range(self.max_iter):
            q_rand = self.sample_q(end_q, start_q)
            idx_near = self.nearest_index(tree, q_rand)
            q_near = tree[idx_near]['q']
            
            q_new = self.steer(q_near, q_rand)
            if q_new is None:
                continue

            tree.append({'q': q_new, 'parent': idx_near})

            if self.ee_dist(q_new, end_q) < self.step_size * 10:
                if self.collision_check_line(q_new, end_q):
                    # print("Collision detected at end point.")
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
            
        for i in reversed(range(len(tree))):
            q_node = tree[i]['q']
            if not self.collision_check_line(q_node, end_q):
                tree.append({'q': end_q, 'parent': i})
                path = []
                idx = len(tree) - 1
                while idx is not None:
                    path.append(tree[idx]['q'])
                    idx = tree[idx]['parent']
                path.reverse()
                path = self.shortcut_path(path)
                return path
            else:
                # print(f"Collision detected at node {i}.")
                continue
            
        print("Failed to find a path.")
        return None
