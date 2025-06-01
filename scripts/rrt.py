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

    def get_obstacle_distance(self, point):
        """Calculate minimum distance from a point to obstacle box"""
        # Convert to native Python floats
        x_min = float(self.boxes_3d[0])
        y_min = float(self.boxes_3d[1])
        z_min = float(self.boxes_3d[2])
        x_max = float(self.boxes_3d[3])
        y_max = float(self.boxes_3d[4])
        z_max = float(self.boxes_3d[5])

        # Calculate distance components
        dx = max(x_min - point[0], 0, point[0] - x_max)
        dy = max(y_min - point[1], 0, point[1] - y_max)
        dz = max(z_min - point[2], 0, point[2] - z_max)
        
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def get_position_bias(self, pos):
        """Calculate position-based bias for sampling"""
        dist = self.get_obstacle_distance(pos)
        return np.exp(-dist / 0.2)  # Distance decay factor

    def generate_safe_sample(self):
        """Generate configuration biased towards safe regions with lower thresholds"""
        best_dist = 0
        best_q = None
        min_acceptable_dist = 0.05  # 5cm minimum acceptable distance
        
        for _ in range(20):  # Try more attempts for better samples
            q = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
            pos = self.robot.forward_kinematics(q)
            dist = self.get_obstacle_distance(pos)
            
            # Return immediately if distance is acceptable
            if dist > min_acceptable_dist:
                print(f"Found safe sample with distance {dist:.3f}m")
                return q
                
            if dist > best_dist:
                best_dist = dist
                best_q = q
                
        if best_q is not None:
            print(f"Using best available sample with distance {best_dist:.3f}m")
            
        return best_q if best_q is not None else np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])

    def generate_biased_sample(self, end_q, start_q):
        """Generate sample biased towards goal and away from obstacles"""
        bias = np.random.rand() * (end_q - start_q)
        noise_std = 0.1 * np.ones(start_q.size)
        active_joints = [0, 1, 2, 4]
        for j in active_joints:
            noise_std[j] = 0.3

        # Add position-based bias
        pos = self.robot.forward_kinematics(start_q)
        pos_bias = self.get_position_bias(pos)
        noise = np.random.randn(start_q.size) * noise_std * (1 + pos_bias)
        
        candidate = start_q + bias + noise
        return np.clip(candidate, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def sample_q(self, end_q, start_q, iteration):
        """Enhanced adaptive sampling strategy"""
        # Non-linear exploration weight growth
        explore_weight = min(0.9, (iteration / self.max_iter) ** 0.5 * 0.7)
        goal_weight = max(0.1, 0.3 - iteration / self.max_iter * 0.2)
        
        r = np.random.rand()
        if r < goal_weight:
            return end_q
        elif r < goal_weight + explore_weight:
            return self.generate_safe_sample()
        else:
            return self.generate_biased_sample(end_q, start_q)

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

    def steer(self, q_near, q_rand, iteration):
        """Enhanced adaptive steering with multi-direction exploration"""
        direction = q_rand - q_near
        length = np.linalg.norm(direction)
        if length == 0:
            return None
            
        # Get current position and distance to obstacle
        pos = self.robot.forward_kinematics(q_near)
        dist_to_obstacle = self.get_obstacle_distance(pos)
        end_pos = self.robot.forward_kinematics(q_rand)
        end_dist = self.get_obstacle_distance(end_pos)
        
        print(f"Steering: current dist = {dist_to_obstacle:.3f}m, target dist = {end_dist:.3f}m")
        
        # Calculate adaptive factors with smaller thresholds
        iteration_factor = min(1.5, (1.0 + iteration/self.max_iter)**0.5)  # Reduced maximum
        distance_factor = min(1.5, max(0.5, (dist_to_obstacle + end_dist) / 0.1))  # Changed from 0.2 to 0.1
        goal_bias = max(0.5, 1.0 - iteration/self.max_iter)
        
        # Combine factors for final step size
        adaptive_step = self.step_size * iteration_factor * distance_factor * goal_bias
        print(f"Step size: base={self.step_size}, adaptive={adaptive_step}")
        
        if length <= adaptive_step:
            if self.collision_check_line(q_near, q_rand):
                return None
            return q_rand
        
        # Try multiple directions if near obstacle
        if dist_to_obstacle < 0.01:  # Reduced from 0.1 to 0.05
            print("Near obstacle, trying multiple directions")
            best_q = None
            best_dist = -float('inf')
            
            # Try original direction plus random perturbations
            for attempt in range(5):  # Increased attempts
                if attempt == 0:
                    step_vector = direction / length * adaptive_step
                else:
                    # Add random perturbation to direction
                    perturbed_direction = direction + np.random.randn(direction.size) * 0.1
                    step_vector = perturbed_direction / np.linalg.norm(perturbed_direction) * adaptive_step
                
                q_try = q_near + step_vector
                
                if not self.is_within_limits(q_try) or self.collion_checker.self_collision(q_try):
                    continue
                    
                pos_try = self.robot.forward_kinematics(q_try)
                dist_try = self.get_obstacle_distance(pos_try)
                
                if dist_try > best_dist and not self.obstacle_collision_check(q_try):
                    best_dist = dist_try
                    best_q = q_try
                    print(f"Found better direction, distance: {best_dist:.3f}m")
            
            return best_q
        else:
            # Standard steering for safer regions
            step_vector = direction / length * adaptive_step
            q_new = q_near + step_vector
            
            if not self.is_within_limits(q_new) or self.collion_checker.self_collision(q_new):
                return None
                
            if self.obstacle_collision_check(q_new):
                return None
                
        return q_new

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

    def rrt_plan(self, start_q, end_q):
        """Adaptive RRT path planning implementation"""
        # Store tree as instance variable
        self.tree = [{'q': start_q, 'parent': None}]
        print("\nStarting adaptive RRT planning...")
        print(f"Start config: {start_q}")
        print(f"Goal config: {end_q}")
        print(f"Base step size: {self.step_size}")

        for i in range(self.max_iter):
            if i % 100 == 0:  # Print progress every 100 iterations
                print(f"RRT iteration {i}/{self.max_iter}, tree size: {len(self.tree)}")
                # Print additional statistics
                near_pos = self.robot.forward_kinematics(self.tree[-1]['q'])
                dist = self.get_obstacle_distance(near_pos)
                print(f"Current distance to obstacle: {dist:.3f}m")

            # Adaptive sampling based on iteration count
            q_rand = self.sample_q(end_q, start_q, i)
            idx_near = self.nearest_index(self.tree, q_rand)
            q_near = self.tree[idx_near]['q']
            
            # Adaptive steering
            q_new = self.steer(q_near, q_rand, i)
            if q_new is None:
                continue

            self.tree.append({'q': q_new, 'parent': idx_near})

            # Adaptive goal check threshold based on iteration
            goal_threshold = self.step_size * (10 - min(5, i/self.max_iter * 5))  # Decrease threshold over time
            if self.ee_dist(q_new, end_q) < goal_threshold:
                if self.collision_check_line(q_new, end_q):
                    print("Found path but collision check failed")
                    continue
                print("Found valid path to goal!")
                self.tree.append({'q': end_q, 'parent': len(self.tree) - 1})
                path = []
                idx = len(self.tree) - 1
                while idx is not None:
                    path.append(self.tree[idx]['q'])
                    idx = self.tree[idx]['parent']
                path.reverse()
                print(f"Path length: {len(path)}")
                return path
            
        # Try direct connections with gradually increasing thresholds
        for threshold_multiplier in [1.0, 1.5, 2.0]:  # Try increasingly relaxed thresholds
            print(f"\nTrying direct connections with threshold {threshold_multiplier}x...")
            for i in reversed(range(len(self.tree))):
                q_node = self.tree[i]['q']
                if self.ee_dist(q_node, end_q) < self.step_size * 10 * threshold_multiplier:
                    if not self.collision_check_line(q_node, end_q):
                        print("Found direct connection to goal!")
                        self.tree.append({'q': end_q, 'parent': i})
                        path = []
                        idx = len(self.tree) - 1
                        while idx is not None:
                            path.append(self.tree[idx]['q'])
                            idx = self.tree[idx]['parent']
                        path.reverse()
                        print(f"Path length: {len(path)}")
                        return path
        
        print("RRT failed to find a path after maximum iterations")
        return None

    def find_safest_node(self, tree):
        """Find the node in the tree that is furthest from obstacles"""
        max_dist = -float('inf')
        best_node = None
        best_pos = None
        
        for node in tree:
            pos = self.robot.forward_kinematics(node['q'])
            dist = self.get_obstacle_distance(pos)
            if dist > max_dist:
                max_dist = dist
                best_node = node
                best_pos = pos
                
        return best_node, best_pos, max_dist

    def plan(self, start_q, end_q, max_attempts=5):
        """Plan a path using iterative RRT from safe points"""
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
        
        # Initialize for iterative planning
        current_q = start_q
        current_cart = start_cart
        attempts = 0
        tree = None
        
        while attempts < max_attempts:
            print(f"\nPlanning attempt {attempts + 1}/{max_attempts}")
            
            # Check if direct path is possible
            if not self.check_cartesian_line(current_cart, end_cart):
                print("Direct path possible!")
                direct_path = self.direct_joint_path(current_q, end_q)
                if direct_path is not None:
                    return direct_path
            
            # Try RRT from current position
            print("\nAttempting RRT planning...")
            path = self.rrt_plan(current_q, end_q)
            if path is not None:
                return path
            
            # If RRT failed, find safest point in the tree to restart from
            if hasattr(self, 'tree'):  # Use the instance's tree from last RRT attempt
                best_node, best_pos, max_dist = self.find_safest_node(self.tree)
                if best_node is not None and max_dist > 0.1:  # Only use points with reasonable clearance
                    print(f"\nMoving to safer position (distance: {max_dist:.3f}m)")
                    current_q = best_node['q']
                    current_cart = best_pos
                else:
                    print("No safe nodes found")
                    break
            
            attempts += 1
            
        print("Failed to find path after maximum attempts")
        return None
