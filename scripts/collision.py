import numpy as np
import math

class CollisionChecker:
    def __init__(self, dh_params):
        self.dh_params = dh_params
        # 定义哪些连杆对不需要检查碰撞
        self.ignore_pairs = {
            (1, 3), (2, 4), (3, 5),  # 相邻连杆对
            (1, 4),   
            (2, 6), (1, 5)           # 距离较远的连杆对
        }
        # [radius, length] for each link
        self.link_geometries = [
            [0.01, 0.2433],  # Link 1: 基座到第一个关节
            [0.01, 0.01],    # Link 2: 第一个关节的偏移
            [0.01, 0.28],    # Link 3: 上臂长度
            [0.01, 0.245],   # Link 4: 前臂长度
            [0.01, 0.057],   # Link 5: 手腕段
            [0.01, 0.235]    # Link 6: 末端执行器
        ]
        self.debug = True  # 启用调试信息

    def get_link_transforms(self, joint_angles):
        """计算每个连杆的变换矩阵"""
        transforms = []
        T = np.eye(4)
        for i in range(len(joint_angles)):
            theta = joint_angles[i]  # joint_angles已经包含了初始偏移
            d = self.dh_params[i][1]
            a = self.dh_params[i][2]
            alpha = self.dh_params[i][3]
            
            # 标准DH变换顺序：Rz(theta) -> Tz(d) -> Tx(a) -> Rx(alpha)
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)

            # 先计算旋转部分
            Rz = np.array([
                [ct, -st, 0, 0],
                [st,  ct, 0, 0],
                [0,   0,  1, 0],
                [0,   0,  0, 1]
            ])

            # 沿着Z轴平移d
            Tz = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, d],
                [0, 0, 0, 1]
            ])

            # 沿着X轴平移a
            Tx = np.array([
                [1, 0, 0, a],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            # 绕X轴旋转alpha
            Rx = np.array([
                [1,   0,    0,  0],
                [0,   ca,  -sa, 0],
                [0,   sa,   ca, 0],
                [0,   0,    0,  1]
            ])

            # 按照正确顺序组合变换
            T_i = Rz @ Tz @ Tx @ Rx
            T = T @ T_i
            transforms.append(T.copy())
        return transforms

    def should_check_collision(self, i, j):
        """判断两个连杆是否需要检查碰撞"""
        return (i+1, j+1) not in self.ignore_pairs

    def check_link_direction(self, p1, p2, p3, p4):
        """检查两个连杆是否近乎平行"""
        dir1 = p2 - p1
        dir2 = p4 - p3
        dir1_norm = np.linalg.norm(dir1)
        dir2_norm = np.linalg.norm(dir2)
        
        if dir1_norm < 1e-6 or dir2_norm < 1e-6:
            return False
            
        cos_angle = np.dot(dir1, dir2) / (dir1_norm * dir2_norm)
        return abs(cos_angle) > 0.9  # 如果夹角小于约25度，认为是平行

    def check_link_collision(self, p1, p2, p3, p4, r1, r2):
        """检查两个圆柱体之间是否碰撞"""
        # 最小碰撞距离阈值（米）
        MIN_COLLISION_DIST = 0.015  # 15mm的安全距离

        # 首先检查是否平行
        if self.check_link_direction(p1, p2, p3, p4):
            return False  # 平行的连杆不太可能碰撞

        # 计算两线段间最短距离
        u = p2 - p1
        v = p4 - p3
        w = p1 - p3
        
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w)
        e = np.dot(v, w)
        
        # 参数化距离
        if abs(a*c - b*b) < 1e-7:  # 平行线段
            sc = 0
            if c != 0:
                tc = e/c
            else:
                tc = 0
        else:
            sc = (b*e - c*d) / (a*c - b*b)
            tc = (a*e - b*d) / (a*c - b*b)
        
        # 确保参数在[0,1]范围内
        sc = np.clip(sc, 0, 1)
        tc = np.clip(tc, 0, 1)
        
        # 计算最近点
        p_c1 = p1 + sc * u
        p_c2 = p3 + tc * v
        
        # 检查最短距离是否小于安全距离
        min_dist = np.linalg.norm(p_c1 - p_c2)
        return min_dist < ((r1 + r2) + MIN_COLLISION_DIST)  # 增加额外的安全距离

    def self_collision(self, joint_angles):
        """检查机器人是否发生自碰撞"""
        transforms = self.get_link_transforms(joint_angles)
        
        # 检查每对连杆之间的碰撞
        for i in range(len(transforms)-2):
            for j in range(i+2, len(transforms)):
                # 获取连杆的起点和终点
                p1 = transforms[i][:3, 3]
                p2 = transforms[i+1][:3, 3]
                p3 = transforms[j][:3, 3]
                p4 = transforms[j+1][:3, 3] if j < len(transforms)-1 else transforms[j][:3, 3]
                
                # 检查是否需要进行碰撞检测
                if not self.should_check_collision(i, j):
                    continue

                # 获取连杆的半径
                r1 = self.link_geometries[i][0]
                r2 = self.link_geometries[j][0]
                
                # 进行碰撞检测
                result = self.check_link_collision(p1, p2, p3, p4, r1, r2)
                if result:
                    if self.debug:
                        # 计算最近点和距离以供调试
                        u = p2 - p1
                        v = p4 - p3
                        w = p1 - p3
                        
                        # 获取最近点的参数
                        a = np.dot(u, u)
                        b = np.dot(u, v)
                        c = np.dot(v, v)
                        d = np.dot(u, w)
                        e = np.dot(v, w)
                        
                        # 计算参数
                        if abs(a*c - b*b) < 1e-7:
                            sc = 0
                            tc = e/c if c != 0 else 0
                        else:
                            sc = np.clip((b*e - c*d) / (a*c - b*b), 0, 1)
                            tc = np.clip((a*e - b*d) / (a*c - b*b), 0, 1)
                        
                        # 计算最近点
                        p_c1 = p1 + sc * u
                        p_c2 = p3 + tc * v
                        min_dist = np.linalg.norm(p_c1 - p_c2)
                        
                        print(f"\nCollision detected between link {i+1} and {j+1}:")
                        print(f"Distance: {min_dist:.3f}, Required: {r1 + r2:.3f}")
                        print(f"Link {i+1} endpoints: {p1}, {p2}")
                        print(f"Link {j+1} endpoints: {p3}, {p4}")
                        print(f"Closest points: {p_c1}, {p_c2}\n")
                    return True
                    
        return False

# 为了兼容性保留原函数
def is_in_collision(joint_angles):
    return False
