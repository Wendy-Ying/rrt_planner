import numpy as np

class Node:
    def __init__(self, q, parent=None, cost=0.0):
        self.q = q  # Joint angles
        self.parent = parent
        self.cost = cost  # Cost from start to this node

class RRTStar:
    def __init__(self, joint_limits, dh_table, collision_checker,
                 step_size=0.1, max_iter=1000, radius=0.5, use_cartesian_cost=True):
        self.joint_limits = joint_limits  # list of (low, high) for each joint
        self.dh_table = dh_table  # (n_joints x 4) -> [theta, d, a, alpha]
        self.collision_checker = collision_checker
        self.step_size = step_size
        self.max_iter = max_iter
        self.radius = radius
        self.use_cartesian_cost = use_cartesian_cost
        
        # 添加关节权重，大关节移动代价高
        self.joint_weights = np.array([
            2.0,  # Joint 1 (base) - 最大权重因为移动影响最大
            1.5,  # Joint 2 (shoulder) - 较大权重
            1.5,  # Joint 3 (elbow) - 较大权重
            1.0,  # Joint 4 - 中等权重
            0.8,  # Joint 5 - 较小权重
            0.5   # Joint 6 (wrist) - 最小权重
        ])
        
        # 笛卡尔空间和关节空间的混合比例
        self.cartesian_weight = 0.3  # 30%笛卡尔空间，70%关节空间

    def sample(self):
        rand = np.random.rand()
        if rand < 0.2:  # 20% 目标偏置
            return self.goal
        elif rand < 0.4:  # 20% 在目标附近采样
            # 在目标附近添加高斯噪声
            noise = np.random.normal(0, 0.2, len(self.goal))  # 标准差0.2弧度
            sample = self.goal + noise
            # 确保在关节限位内
            return self.wrap_angles(sample)
        else:  # 60% 完全随机采样
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

    def joint_space_cost(self, q1, q2):
        """计算加权的关节空间代价"""
        joint_diff = np.abs(q2 - q1)
        # 将角度差归一化到[-pi, pi]范围
        joint_diff = np.where(joint_diff > np.pi, 2*np.pi - joint_diff, joint_diff)
        return np.sum(self.joint_weights * joint_diff)

    def cost(self, from_node, to_q):
        """计算混合代价"""
        # 计算关节空间代价
        joint_cost = self.joint_space_cost(from_node.q, to_q)
        
        if self.use_cartesian_cost:
            # 计算笛卡尔空间代价
            cart_cost = self.cartesian_distance(from_node.q, to_q)
            # 归一化并混合两种代价
            return from_node.cost + ((1 - self.cartesian_weight) * joint_cost + 
                                   self.cartesian_weight * cart_cost)
        else:
            return from_node.cost + joint_cost

    def interpolate(self, q1, q2, steps=10):
        """在两个构型之间进行线性插值"""
        path = []
        for i in range(steps + 1):
            t = i / steps
            q = q1 + t * (q2 - q1)
            path.append(self.wrap_angles(q))
        return path

    def simplify_path(self, path):
        """简化路径，移除不必要的中间点"""
        if len(path) < 3:
            return path

        simplified = [path[0]]
        i = 0
        while i < len(path) - 1:
            # 尝试跳过中间点直接连接
            next_idx = i + 2
            while next_idx < len(path):
                # 检查直接路径是否可行
                direct_path = self.interpolate(path[i], path[next_idx])
                if all(not self.collision_checker.self_collision(q) for q in direct_path):
                    i = next_idx - 1  # 下一次从这个点开始
                    break
                next_idx -= 1
            simplified.append(path[i + 1])
            i += 1
        
        if simplified[-1] != path[-1]:
            simplified.append(path[-1])
        return simplified

    def plan(self, start, goal):
        """
        Main RRT* planning loop with path optimization
        Args:
            start: start configuration
            goal: goal configuration
        """
        self.start = Node(start)
        self.goal = np.array(goal)
        self.nodes = [self.start]

        # 首先检查是否存在直接路径
        print("Checking for direct path...")
        direct_path = self.interpolate(start, goal)
        if all(not self.collision_checker.self_collision(q) for q in direct_path):
            print("Found direct path!")
            return direct_path
        for i in range(self.max_iter):
            q_rand = self.sample()
            nearest_node = self.nearest(q_rand)
            q_new = self.steer(nearest_node.q, q_rand)

            if self.collision_checker.self_collision(q_new):
                continue

            near_nodes = self.get_near_nodes(q_new)

            # Choose best parent
            min_cost = self.cost(nearest_node, q_new)
            best_parent = nearest_node
            for node in near_nodes:
                if not self.collision_checker.self_collision(q_new):
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
                if c < node.cost and not self.collision_checker.self_collision(node.q):
                    node.parent = new_node
                    node.cost = c

            if np.linalg.norm(q_new - self.goal) < self.step_size:
                print(f"Goal reached in {i} iterations.")
                path = self.extract_path(new_node)
                
                # 简化和优化路径
                print("Simplifying path...")
                simplified_path = self.simplify_path(path)
                if len(simplified_path) < len(path):
                    print(f"Path simplified from {len(path)} to {len(simplified_path)} points")
                    path = simplified_path

                # Path execution (can be commented out to disable)
                # if execute:
                #     try:
                #         import pid_angle_control
                #         import utilities
                #         from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

                #         print("Executing planned path...")
                #         args = utilities.parseConnectionArguments()
                #         with utilities.DeviceConnection.createTcpConnection(args) as router:
                #             base = BaseClient(router)
                #             success = pid_angle_control.execute_path(base, path)
                #             if not success:
                #                 print("Path execution failed")
                #             else:
                #                 print("Path execution completed successfully")
                #     except Exception as e:
                #         print(f"Error executing path: {str(e)}")

                return path

        print("Failed to find a path.")
        return None

    def extract_path(self, node):
        path = []
        while node:
            path.append(node.q)
            node = node.parent
        return path[::-1]
