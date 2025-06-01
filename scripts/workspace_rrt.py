import numpy as np

class WorkspaceRRT:
    def __init__(self, obstacle, step_size=0.05, clearance=0.15):
        # Store obstacle boundaries
        self.x_min = float(obstacle[0])
        self.y_min = float(obstacle[1])
        self.z_min = float(obstacle[2])
        self.x_max = float(obstacle[3])
        self.y_max = float(obstacle[4])
        self.z_max = float(obstacle[5])
        self.step_size = step_size
        self.clearance = clearance

    def get_intermediate_target(self, start_pos, end_pos):
        """Generate intermediate target to guide end-effector around obstacle"""
        # Calculate obstacle center
        obstacle_center_x = (self.x_min + self.x_max) / 2
        obstacle_center_y = (self.y_min + self.y_max) / 2

        # Determine if we should go above or below the obstacle
        if start_pos[1] < obstacle_center_y:
            # Go below the obstacle
            y_target = self.y_min - self.clearance
        else:
            # Go above the obstacle
            y_target = self.y_max + self.clearance

        # Create waypoint at x position of obstacle but offset in y
        intermediate_target = np.array([
            obstacle_center_x,  # x at obstacle center
            y_target,          # y with clearance above/below
            (start_pos[2] + end_pos[2]) / 2  # z halfway between start and end
        ])

        return intermediate_target

    def plan(self, start_pos, end_pos):
        """Plan a path in workspace that avoids obstacles"""
        path = []
        path.append(start_pos)

        # Check if direct path intersects with obstacle
        if self._check_line_obstacle_intersection(start_pos, end_pos):
            # Generate intermediate target to avoid obstacle
            via_point = self.get_intermediate_target(start_pos, end_pos)
            
            # Add intermediate point
            path.append(via_point)

        path.append(end_pos)
        return path

    def _check_line_obstacle_intersection(self, start, end):
        """Check if line segment intersects with obstacle"""
        # Check multiple points along the line
        for t in np.linspace(0, 1, 20):
            point = start + t * (end - start)
            if (self.x_min <= point[0] <= self.x_max and
                self.y_min <= point[1] <= self.y_max and
                self.z_min <= point[2] <= self.z_max):
                return True
        return False
