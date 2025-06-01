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
        p1 = np.array(self.robot.forward_kinematics(q1))
        p2 = np.array(self.robot.forward_kinematics(q2))
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

        # Ensure inputs are numpy arrays
        q = np.array(q)
        joint_positions = np.array(self.robot.get_joint_positions(q))
        # Convert box boundaries to native Python floats
        x_min = float(self.boxes_3d[0])
        y_min = float(self.boxes_3d[1])
        z_min = float(self.boxes_3d[2])
        x_max = float(self.boxes_3d[3])
        y_max = float(self.boxes_3d[4])
        z_max = float(self.boxes_3d[5])

        for i in range(joint_positions.shape[0] - 1):
            start = np.array(joint_positions[i])
            end = np.array(joint_positions[i + 1])

            for alpha in np.linspace(0, 1, num_interpolation_points):
                point = np.array((1 - alpha) * start + alpha * end)
                # Box boundaries with margin
                margin = 0
                x_check_min = x_min - margin
                y_check_min = y_min - margin
                z_check_min = z_min - margin
                x_check_max = x_max + margin
                y_check_max = y_max + margin
                z_check_max = z_max + margin
                if (x_check_min <= point[0] <= x_check_max and
                    y_check_min <= point[1] <= y_check_max and
                    z_check_min <= point[2] <= z_check_max):
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
        # Ensure point is numpy array
        point = np.array(point)
        """Calculate tangent points from a point to a 3D box"""
        # Convert to native Python floats to avoid numpy scalar issues
        x_min = float(box[0])
        y_min = float(box[1])
        z_min = float(box[2])
        x_max = float(box[3])
        y_max = float(box[4])
        z_max = float(box[5])
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_min, y_max, z_min], [x_max, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_min, y_max, z_max], [x_max, y_max, z_max]
        ])
        
        tangent_points = []
        
        # For each edge of the box
        edges = [
            # Vertical edges (marked with 'v')
            ([0, 4], ('v', x_min, y_min)), ([1, 5], ('v', x_max, y_min)),
            ([2, 6], ('v', x_min, y_max)), ([3, 7], ('v', x_max, y_max)),
            # Horizontal edges at z_min (marked with 'h')
            ([0, 1], ('h', z_min)), ([0, 2], ('h', z_min)),
            ([1, 3], ('h', z_min)), ([2, 3], ('h', z_min)),
            # Horizontal edges at z_max
            ([4, 5], ('h', z_max)), ([4, 6], ('h', z_max)),
            ([5, 7], ('h', z_max)), ([6, 7], ('h', z_max))
        ]
        
        for edge in edges:
            edge_type = edge[1][0]  # Get the type ('v' or 'h')
            if edge_type == 'v':    # Vertical edge
                # Project point onto the vertical line
                x, y = edge[1][1], edge[1][2]  # Get coordinates from tuple
                z = np.clip(point[2], z_min, z_max)
                tangent_points.append(np.array([x, y, z]))
            else:  # Horizontal edge
                z = edge[1][1]  # Get z coordinate from tuple
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
        # Ensure inputs are numpy arrays
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        # Convert box boundaries to native Python floats
        x_min = float(self.boxes_3d[0])
        y_min = float(self.boxes_3d[1])
        z_min = float(self.boxes_3d[2])
        x_max = float(self.boxes_3d[3])
        y_max = float(self.boxes_3d[4])
        z_max = float(self.boxes_3d[5])
        
        # Check multiple points along the line
        for t in np.linspace(0, 1, num_points):
            point = start_pos + t * (end_pos - start_pos)
            point = np.array(point)  # Ensure point is array
            
            # Check if point is inside obstacle
            if (x_min <= point[0] <= x_max and
                y_min <= point[1] <= y_max and
                z_min <= point[2] <= z_max):
                return True  # Collision detected
                
        return False  # No collision

    def rrt_plan(self, start_q, end_q):
        """Basic RRT path planning implementation"""
        tree = [{'q': start_q, 'parent': None}]
        print("\nStarting RRT planning...")
        print(f"Start config: {start_q}")
        print(f"Goal config: {end_q}")
        print(f"Step size: {self.step_size}")

        for i in range(self.max_iter):
            if i % 100 == 0:  # Print progress every 100 iterations
                print(f"RRT iteration {i}/{self.max_iter}, tree size: {len(tree)}")
            q_rand = self.sample_q(end_q, start_q)
            idx_near = self.nearest_index(tree, q_rand)
            q_near = tree[idx_near]['q']
            
            q_new = self.steer(q_near, q_rand)
            if q_new is None:
                continue

            tree.append({'q': q_new, 'parent': idx_near})

            if self.ee_dist(q_new, end_q) < self.step_size * 10:
                if self.collision_check_line(q_new, end_q):
                    print("Found path but collision check failed")
                    continue
                print("Found valid path to goal!")
                tree.append({'q': end_q, 'parent': len(tree) - 1})
                path = []
                idx = len(tree) - 1
                while idx is not None:
                    path.append(tree[idx]['q'])
                    idx = tree[idx]['parent']
                path.reverse()
                print(f"Path length: {len(path)}")
                return path
            
        for i in reversed(range(len(tree))):
            q_node = tree[i]['q']
            if not self.collision_check_line(q_node, end_q):
                print("Found direct connection to goal!")
                tree.append({'q': end_q, 'parent': i})
                path = []
                idx = len(tree) - 1
                while idx is not None:
                    path.append(tree[idx]['q'])
                    idx = tree[idx]['parent']
                path.reverse()
                print(f"Path length: {len(path)}")
                return path
            
        print("RRT failed to find a path after maximum iterations")
        return None

    def direct_joint_path(self, start_q, end_q, num_points=20):
        """Generate a straight-line path between two joint configurations"""
        alphas = np.linspace(0, 1, num_points)
        path = []
        for alpha in alphas:
            q = start_q + alpha * (end_q - start_q)
            if not self.is_within_limits(q) or self.collision_check_line(start_q, q):
                return None
            path.append(q)
        return path

    def get_best_tangent_ik(self, current_cart, end_cart, tangent_points):
        """Find best tangent point with valid IK solution"""
        line_vector = end_cart - current_cart
        min_dist = float('inf')
        best_tangent = None
        best_tangent_q = None
        
        for point in tangent_points:
            point_vector = point - current_cart
            dist = np.linalg.norm(np.cross(point_vector, line_vector)) / np.linalg.norm(line_vector)
            
            # Try to get IK solution
            tangent_q = self.robot.inverse_kinematics(point)
            if tangent_q is not None and dist < min_dist:
                min_dist = dist
                best_tangent = point
                best_tangent_q = tangent_q
                
        return best_tangent, best_tangent_q

    def iterative_plan(self, current_q, end_q, current_cart=None, end_cart=None, depth=0, max_depth=5):
        """Iteratively plan path through tangent points"""
        if depth >= max_depth:
            print(f"Max recursion depth {max_depth} reached")
            return None
            
        # Get cartesian positions if not provided
        if current_cart is None:
            current_cart = self.robot.forward_kinematics(current_q)
        if end_cart is None:
            end_cart = self.robot.forward_kinematics(end_q)
            
        # First try direct path if possible
        if not self.check_cartesian_line(current_cart, end_cart):
            print("Direct path possible!")
            direct_path = self.direct_joint_path(current_q, end_q)
            if direct_path is not None:
                return direct_path
                
        # Calculate tangent points
        print("\nFinding tangent points...")
        tangent_points = self.calculate_3d_tangents(current_cart, self.boxes_3d)
        best_tangent, best_tangent_q = self.get_best_tangent_ik(current_cart, end_cart, tangent_points)
        
        if best_tangent_q is None:
            print("No valid tangent point found, trying direct RRT")
            return self.rrt_plan(current_q, end_q)
            
        print(f"Best tangent point found at {best_tangent}")
        print("Using direct path to tangent first")
        path1 = self.direct_joint_path(current_q, best_tangent_q)
        
        # If direct path fails, try RRT
        if path1 is None:
            print("Direct path failed, trying RRT to tangent")
            path1 = self.rrt_plan(current_q, best_tangent_q)
            if path1 is None:
                print("Failed to reach tangent point")
                return None
                
        # Recursively plan from tangent to goal
        print("Planning from tangent to goal...")
        path2 = self.iterative_plan(best_tangent_q, end_q, best_tangent, end_cart, depth + 1)
        if path2 is None:
            return None
            
        return path1 + path2[1:]
        
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
            start_cart = np.array(self.robot.forward_kinematics(start_q))
            
        if end_q.size == 3:  # 3D position input
            end_cart = np.array(end_q)
            end_q = self.robot.inverse_kinematics(end_q)
            print(f"IK end: {end_q}")
            if end_q is None:
                raise ValueError("IK failed for end position")
            end_q = np.array(end_q)
        else:  # Joint angles input
            end_cart = np.array(self.robot.forward_kinematics(end_q))

        # Ensure all positions are numpy arrays
        start_cart = np.array(start_cart)
        end_cart = np.array(end_cart)
        start_q = np.array(start_q)
        end_q = np.array(end_q)
        
        print("\nStarting iterative planning...")
        path = self.iterative_plan(start_q, end_q, start_cart, end_cart)
        
        if path is not None:
            print("Successfully found path!")
            return path
            
        print("Failed to find any valid path")
        return None
