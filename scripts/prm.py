import numpy as np
from scipy.spatial import KDTree
import heapq

class PRMPlanner:
    def __init__(self, robot, joint_limits, boxes_3d, num_samples=200, k=10):
        """
        robot: 机器人模型，提供get_joint_positions和is_within_limits接口
        joint_limits: [(low1, high1), (low2, high2), ...]
        boxes_3d: 障碍物列表，每个为(x_min, y_min, z_min, x_max, y_max, z_max)
        num_samples: PRM采样点数量
        k: 每个节点连接k个最近邻
        """
        self.robot = robot
        self.joint_limits = joint_limits
        self.boxes_3d = boxes_3d
        self.num_samples = num_samples
        self.k = k
        self.dim = len(joint_limits)

        self.samples = []
        self.graph = {}  # 邻接表 {index: [(neighbor_index, cost), ...]}

    def sample_random_state(self):
        q = np.array([np.random.uniform(low, high) for (low, high) in self.joint_limits])
        return q

    def is_state_valid(self, q):
        # 判断关节限制
        # if not self.robot.is_within_limits(q):
        #     return False
        # 碰撞检测
        if self.cartesian_collision_check(q):
            return False
        if self.obstacle_collision_check(q):
            return False
        return True

    def cartesian_collision_check(self, q):
        joint_positions = self.robot.get_joint_positions(q)
        if np.any(joint_positions[:, 2] < -0.2):
            return True
        return False

    def obstacle_collision_check(self, q, num_interpolation_points=10):
        if len(self.boxes_3d) == 0:
            return False

        joint_positions = self.robot.get_joint_positions(q)

        for i in range(len(joint_positions) - 1):
            start = joint_positions[i]
            end = joint_positions[i + 1]

            for alpha in np.linspace(0, 1, num_interpolation_points):
                point = (1 - alpha) * start + alpha * end

                x_min, y_min, z_min, x_max, y_max, z_max = self.boxes_3d
                if (x_min <= point[0] <= x_max and
                    y_min <= point[1] <= y_max and
                    z_min <= point[2] <= z_max):
                    return True
        return False

    def edge_valid(self, q1, q2, num_interpolation_points=10):
        # 检查两点连线是否碰撞（线性插值）
        for alpha in np.linspace(0, 1, num_interpolation_points):
            q = (1 - alpha) * q1 + alpha * q2
            if not self.is_state_valid(q):
                return False
        return True

    def build_roadmap(self, start_q, goal_q):
        # 采样有效节点
        self.samples = []
        while len(self.samples) < self.num_samples:
            q = self.sample_random_state()
            if self.is_state_valid(q):
                self.samples.append(q)

        # 加入起点终点
        self.samples.append(start_q)
        self.samples.append(goal_q)

        # 建立kd树加速邻居查询
        tree = KDTree(self.samples)

        self.graph = {i: [] for i in range(len(self.samples))}

        # 连接k近邻
        for i, q in enumerate(self.samples):
            distances, indices = tree.query(q, k=self.k + 1)  # 包含自身
            for dist, idx in zip(distances[1:], indices[1:]):  # 跳过自身
                if self.edge_valid(q, self.samples[idx]):
                    self.graph[i].append((idx, dist))
                    self.graph[idx].append((i, dist))  # 无向图

    def dijkstra(self, start_idx, goal_idx):
        queue = [(0, start_idx, [])]  # (cost, node, path)
        visited = set()
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node in visited:
                continue
            path = path + [node]
            if node == goal_idx:
                return path
            visited.add(node)
            for neighbor, edge_cost in self.graph[node]:
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + edge_cost, neighbor, path))
        return None

    def plan(self, start_q, goal_q):
        self.build_roadmap(start_q, goal_q)
        start_idx = len(self.samples) - 2
        goal_idx = len(self.samples) - 1
        path_indices = self.dijkstra(start_idx, goal_idx)
        if path_indices is None:
            print("No path found")
            return None
        else:
            path = [self.samples[i] for i in path_indices]
            return path


# -------------------示例用法--------------------

# 假设你已有：
# robot对象，带get_joint_positions和is_within_limits接口
# boxes_3d为障碍物列表
# joint_limits为关节角度范围列表
# start_q和goal_q为起止关节角度numpy数组

# planner = PRMPlanner(robot, joint_limits, boxes_3d)
# path = planner.plan(start_q, goal_q)
# if path is not None:
#     for q in path:
#         print(q)
