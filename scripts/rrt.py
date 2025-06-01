import numpy as np

class RRTPlanner:
    def __init__(self, robot, joint_limits, collison_checker, obstacle, step_size=0.05, max_iter=1000, goal_sample_rate=0.2, n_steps=3):
        self.robot = robot
        self.joint_limits = np.array(joint_limits)
        self.collion_checker = collison_checker
        self.boxes_3d = np.array(obstacle)
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.n_steps = n_steps

    def sample_q(self, end_q, start_q):
        if np.random.rand() < self.goal_sample_rate:
            return end_q
        
        bias = np.random.rand() * (end_q - start_q)

        noise_std = 0.1 * np.ones(start_q.size)
        active_joints = [0, 1, 2, 4]
        for j in active_joints:
            noise_std[j] = 0.3

        noise = np.random.randn(start_q.size) * noise_std

        candidate = start_q + bias + noise + np.random.randn(start_q.size)
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

        q = np.array(q)
        for i in range(q.size):
            if not is_angle_within_limit(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1]):
                return False
        return True

    def collision_check_line(self, q1, q2):
        for alpha in np.linspace(0, 1, self.n_steps):
            q_interp = q1 + alpha * (q2 - q1)
            if self.obstacle_collision_check(q_interp):
                return True
            if self.cartesian_collision_check(q_interp):
                return True
        return False
    
    def cartesian_collision_check(self, q):
        joint_positions = self.robot.get_joint_positions(q)
        if np.any(joint_positions[:, 2] < -0.2):
            return True
        return False

    def obstacle_collision_check(self, q, num_interpolation_points=10):
        if self.boxes_3d.size == 0:
            return False

        joint_positions = self.robot.get_joint_positions(q)

        for i in range(joint_positions.shape[0] - 1):
            start = joint_positions[i]
            end = joint_positions[i + 1]

            for alpha in np.linspace(0, 1, num_interpolation_points):
                point = (1 - alpha) * start + alpha * end

                x_min, y_min, z_min, x_max, y_max, z_max = self.boxes_3d
                radius = 0
                x_min -= radius
                y_min -= radius
                z_min -= radius
                x_max += radius
                y_max += radius
                z_max += radius
                if (x_min <= point[0] <= x_max and
                    y_min <= point[1] <= y_max and
                    z_min <= point[2] <= z_max):
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
        if not self.is_within_limits(q_new) or self.collion_checker.self_collision(q_new):
            return None
        return q_new

    def calculate_3d_tangents(self, point, box):
        """Calculate tangent points from a point to a 3D box"""
        x_min, y_min, z_min, x_max, y_max, z_max = box
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_min, y_max, z_min], [x_max, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_min, y_max, z_max], [x_max, y_max, z_max]
        ])
        
        tangent_points = []
        
        # For each edge of the box
        edges = [
            # Vertical edges
            ([0, 4], [x_min, y_min]), ([1, 5], [x_max, y_min]),
            ([2, 6], [x_min, y_max]), ([3, 7], [x_max, y_max]),
            # Horizontal edges at z_min
            ([0, 1], z_min), ([0, 2], z_min),
            ([1, 3], z_min), ([2, 3], z_min),
            # Horizontal edges at z_max
            ([4, 5], z_max), ([4, 6], z_max),
            ([5, 7], z_max), ([6, 7], z_max)
        ]
        
        for edge in edges:
            if len(edge[1]) == 2:  # Vertical edge
                # Project point onto the vertical line
                x, y = edge[1]
                z = np.clip(point[2], z_min, z_max)
                tangent_points.append(np.array([x, y, z]))
            else:  # Horizontal edge
                z = edge[1]
                # Get the two corner points
                c1, c2 = corners[edge[0][0]], corners[edge[0][1]]
                # Project point onto the line segment
                v = c2 - c1
                t = np.clip(np.dot(point - c1, v) / np.dot(v, v), 0, 1)
                proj = c1 + t * v
                tangent_points.append(proj)
        
        return tangent_points

    def check_cartesian_line(self, start_pos, end_pos, num_points=20):
        """Check if a line in Cartesian space collides with the obstacle"""
        x_min, y_min, z_min, x_max, y_max, z_max = self.boxes_3d
        
        # Check multiple points along the line
        for t in np.linspace(0, 1, num_points):
            point = start_pos + t * (end_pos - start_pos)
            
            # Check if point is inside obstacle
            if (x_min <= point[0] <= x_max and
                y_min <= point[1] <= y_max and
                z_min <= point[2] <= z_max):
                return True  # Collision detected
                
        return False  # No collision

    def rrt_plan(self, start_q, end_q):
        """Basic RRT path planning implementation"""
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
                    continue
                tree.append({'q': end_q, 'parent': len(tree) - 1})
                path = []
                idx = len(tree) - 1
                while idx is not None:
                    path.append(tree[idx]['q'])
                    idx = tree[idx]['parent']
                path.reverse()
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
                return path
            
        print("Failed to find a path.")
        return None

    def plan(self, start_q, end_q):
        """Plan a path considering direct path and obstacle avoidance"""
        # Convert inputs to numpy arrays
        start_q = np.array(start_q)
        end_q = np.array(end_q)
        
        # Convert to Cartesian space if needed
        if start_q.size == 3:  # 3D position input
            start_cart = start_q
            start_q = self.robot.inverse_kinematics(start_q)
            print(f"IK start: {start_q}")
            if start_q is None:
                raise ValueError("IK failed for start position")
            start_q = np.array(start_q)
        else:  # Joint angles input
            start_cart = self.robot.forward_kinematics(start_q)
            
        if end_q.size == 3:  # 3D position input
            end_cart = end_q
            end_q = self.robot.inverse_kinematics(end_q)
            print(f"IK end: {end_q}")
            if end_q is None:
                raise ValueError("IK failed for end position")
            end_q = np.array(end_q)
        else:  # Joint angles input
            end_cart = self.robot.forward_kinematics(end_q)
            
        # Check if direct path is blocked by obstacle
        if self.check_cartesian_line(start_cart, end_cart):
            print("Direct path blocked, trying intermediate points...")
        else:
            print("Direct path possible, attempting direct planning...")
            direct_path = self.rrt_plan(start_q, end_q)
            if direct_path is not None:
                return direct_path
            
        # Direct path not possible, try intermediate points
        tangent_points = self.calculate_3d_tangents(start_cart, self.boxes_3d)
        line_vector = end_cart - start_cart
        min_dist = float('inf')
        best_tangent = None
        
        for point in tangent_points:
            point_vector = point - start_cart
            dist = np.linalg.norm(np.cross(point_vector, line_vector)) / np.linalg.norm(line_vector)
            if dist < min_dist:
                min_dist = dist
                best_tangent = point
        
        if best_tangent is not None:
            tangent_q = self.robot.inverse_kinematics(best_tangent)
            if tangent_q is not None:
                print("Planning through tangent point...")
                path1 = self.rrt_plan(start_q, tangent_q)
                if path1 is not None:
                    path2 = self.rrt_plan(tangent_q, end_q)
                    if path2 is not None:
                        return path1 + path2[1:]
        
        # If all else fails, try direct RRT planning again
        print("Attempting direct RRT planning...")
        return self.rrt_plan(start_q, end_q)
