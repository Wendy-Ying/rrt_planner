import numpy as np

class RRTPlanner:
    def __init__(self, robot, joint_limits, collison_checker, obstacle, step_size=0.05, max_iter=1000, goal_sample_rate=0.1, n_steps=5):
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

        # Get current ee positions
        start_ee = np.array(self.robot.forward_kinematics(start_q))
        end_ee = np.array(self.robot.forward_kinematics(end_q))
        
        # Calculate box boundaries with margin for avoidance
        x_min = float(self.boxes_3d[0])
        y_min = float(self.boxes_3d[1])
        z_min = float(self.boxes_3d[2])
        x_max = float(self.boxes_3d[3])
        y_max = float(self.boxes_3d[4])
        z_max = float(self.boxes_3d[5])
        
        # If path would cross the obstacle, try to sample around it
        if (min(start_ee[0], end_ee[0]) < x_max and max(start_ee[0], end_ee[0]) > x_min and
            min(start_ee[1], end_ee[1]) < y_max and max(start_ee[1], end_ee[1]) > y_min):
            # Sample with bias to avoid obstacle
            if start_ee[1] < y_min:  # If start is below obstacle, bias downward
                bias_y = -0.2
            else:  # If start is above obstacle, bias upward
                bias_y = 0.2
            
            # Apply bias to random sample
            for _ in range(10):  # Try up to 10 times
                bias = np.random.rand() * (end_q - start_q)
                noise_std = 0.5 * np.ones(start_q.size)
                active_joints = [0, 1, 2, 4]
                for j in active_joints:
                    noise_std[j] = 0.3
                noise = np.random.randn(start_q.size) * noise_std
                candidate = start_q + bias + noise
                
                # Check if this configuration avoids the obstacle
                ee_pos = self.robot.forward_kinematics(candidate)
                if not (x_min <= ee_pos[0] <= x_max and
                       y_min <= ee_pos[1] <= y_max):
                    return np.clip(candidate, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        # Default sampling if no obstacle avoidance needed
        bias = np.random.rand() * (end_q - start_q)
        noise_std = 0.1 * np.ones(start_q.size)
        active_joints = [0, 1, 2, 4]
        for j in active_joints:
            noise_std[j] = 0.3
        noise = np.random.randn(start_q.size) * noise_std
        candidate = start_q + bias + noise
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
        ee_position = np.array(self.robot.forward_kinematics(q))
        
        # Convert box boundaries to native Python floats
        x_min = float(self.boxes_3d[0])
        y_min = float(self.boxes_3d[1])
        z_min = float(self.boxes_3d[2])
        x_max = float(self.boxes_3d[3])
        y_max = float(self.boxes_3d[4])
        z_max = float(self.boxes_3d[5])
        # print(f"x_min={x_min:.3f}, x_max={x_max:.3f}, ")
        # print(f"y_min={y_min:.3f}, y_max={y_max:.3f}, ")
        # print(f"z_min={z_min:.3f}, z_max={z_max:.3f}, ")

        # Check 
        if (x_min <= ee_position[0] <= x_max and
            y_min <= ee_position[1] <= y_max and
            z_min <= ee_position[2] <= z_max):
            print(f"\nEnd-effector collision detected!")
            print(f"EE Position: x={ee_position[0]:.3f}, y={ee_position[1]:.3f}, z={ee_position[2]:.3f}")
            return True

        # Joint collision check code (commented out for now)

        joint_positions = np.array(self.robot.get_joint_positions(q))
        for i in range(joint_positions.shape[0] - 1):
            start = np.array(joint_positions[i])
            end = np.array(joint_positions[i + 1])

            for alpha in np.linspace(0, 1, num_interpolation_points):
                point = np.array((1 - alpha) * start + alpha * end)
                if (x_min <= point[0] <= x_max and
                    y_min <= point[1] <= y_max and
                    z_min <= point[2] <= z_max):
                    print(f"Joint link collision detected at position: {point}")
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
                print(f"Cartesian path collision at point: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
                return True  # Collision detected
                
        return False  # No collision

    def rrt_plan(self, start_q, end_q):
        """Basic RRT path planning implementation"""
        tree = [{'q': start_q, 'parent': None}]
        failed_nodes = []  # Keep track of nodes that led to invalid paths
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
            
            # Skip sampling near failed nodes
            if any(np.linalg.norm(q_near - failed_node) < self.step_size for failed_node in failed_nodes):
                continue
            
            q_new = self.steer(q_near, q_rand)
            if q_new is None:
                continue

            tree.append({'q': q_new, 'parent': idx_near})
            tree_idx = len(tree) - 1  # Index of newly added node

            if self.ee_dist(q_new, end_q) < self.step_size * 10:
                if self.collision_check_line(q_new, end_q):
                    # Remove invalid node and add to failed nodes
                    failed_node = tree.pop()['q']
                    failed_nodes.append(failed_node)
                    print("Found path but collision check failed - backtracking")
                    continue

                # Construct potential path
                tree.append({'q': end_q, 'parent': tree_idx})
                path = []
                idx = len(tree) - 1
                valid_nodes = set()  # Keep track of valid nodes

                # Build the path and validate
                while idx is not None:
                    path.append(tree[idx]['q'])
                    valid_nodes.add(idx)  # Mark this node as part of a valid path
                    idx = tree[idx]['parent']
                path.reverse()

                # Validate entire path with higher resolution
                print("\nValidating path with high-resolution collision checking...")
                valid_path = True
                invalid_idx = None

                for i in range(len(path) - 1):
                    start_q = path[i]
                    end_q = path[i + 1]
                    
                    # Check multiple points along segment with higher resolution
                    for t in np.linspace(0, 1, 10):
                        interp_q = start_q + t * (end_q - start_q)
                        pos = self.robot.forward_kinematics(interp_q)
                        
                        if self.obstacle_collision_check(interp_q, num_interpolation_points=20):
                            print(f"Collision detected during high-resolution check at point {i}")
                            valid_path = False
                            invalid_idx = i
                            break
                    
                    if not valid_path:
                        break

                if not valid_path:
                    # Remove invalid nodes from the tree
                    for i in range(len(tree) - 1, -1, -1):
                        if i not in valid_nodes:
                            failed_node = tree.pop(i)['q']
                            failed_nodes.append(failed_node)
                    print("Path validation failed - invalid nodes removed")
                    continue

                print("\nFound valid path to goal!")
                print(f"Path length: {len(path)}")
                print("\nFinal path waypoints:")
                for i, q in enumerate(path):
                    pos = self.robot.forward_kinematics(q)
                    print(f"Waypoint {i} position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
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
        if start_q is not None:
            start_q = np.array(start_q)
        if end_q is not None:
            end_q = np.array(end_q)
            
        # Check if direct path is possible
        if self.check_cartesian_line(start_cart, end_cart):
            print("Direct path blocked, using RRT planning...")
        else:
            print("Direct path possible, attempting direct planning...")
            direct_path = self.rrt_plan(start_q, end_q)
            if direct_path is not None:
                return direct_path
        
        # If direct path failed or is blocked, just use RRT planning
        print("Using RRT planning...")
        return self.rrt_plan(start_q, end_q)
