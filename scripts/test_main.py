#!/usr/bin/env python3
import rospy
import numpy as np
import time

from moveit import init, add_obstacle, set_goal, go_home
from kinematics import NLinkArm
from optimize import BSplineOptimizer

import pid_angle_control
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

def execute_task_segment(group, base, target_pos, optimizer):
    """
    Simplified version of task segment execution without interruption handling
    """
    # Plan new path
    print(f"Planning path to {target_pos}")
    path = set_goal(group, target_pos[0], target_pos[1], target_pos[2])
    if path is None:
        print("Failed to plan path")
        return False
        
    # Optimize path
    smooth_path = optimizer.optimize(path)
    
    # Execute path
    success = pid_angle_control.execute_path(base, smooth_path)
    
    if success:
        print("Successfully reached target")
        return True
    else:
        print("Execution failed")
        return False

def main():
    # Define positions as numpy arrays (manually set)
    obj = np.array([0.166, 0.148, 0], dtype=float)      # Object position
    obstacle = np.array([0.367, 0.11, 0], dtype=float)  # Obstacle position
    goal = np.array([0.552, 0.131, 0], dtype=float)     # Goal position

    # Robot parameters (identical to main.py)
    joint_limits = np.array([(205, 150), (205, 150), (210, 150), 
                            (210, 145), (215, 140), (210, 150)]) / 180 * np.pi

    dh_params = [
        [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
        [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
        [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
        [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
    ]

    robot = NLinkArm(dh_params, joint_limits)

    print(f"Object position: {obj}")
    print(f"Goal position: {goal}")
    print(f"Obstacle position: {obstacle}")

    # Initialize MoveIt
    group, scene, _ = init()

    # Set up static obstacle
    scene.remove_world_object()
    rospy.sleep(1.0)
    add_obstacle(scene, obstacle[0], obstacle[1]+0.02, 0.01, 0.12, 0.12, 0.2)

    # Initialize optimizer
    optimizer = BSplineOptimizer(robot, degree=3, num_points=20)

    # Setup connection
    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            
            # Initial gripper position
            pid_angle_control.send_gripper_command(base, 0.2)

            # Move to object
            success = execute_task_segment(group, base, [obj[0], obj[1]+0.02, 0.03], optimizer)
            if success:
                # Grasp object
                pid_angle_control.send_gripper_command(base, 0.8)
                print("Successfully picked up object")

                # Move to goal
                success = execute_task_segment(group, base, [goal[0], goal[1], 0.02], optimizer)
                if success:
                    # Release object
                    pid_angle_control.send_gripper_command(base, 0.3)
                    print("Successfully placed object at goal")

                    # Return home
                    path = go_home(group, mode="C")
                    if path is not None:
                        smooth_path = optimizer.optimize(path)
                        success = pid_angle_control.execute_path(base, smooth_path)
                        if success:
                            print("Successfully returned home")
                        else:
                            print("Failed to return home")
                    else:
                        print("Failed to plan path home")
                else:
                    print("Failed to reach goal position")
            else:
                print("Failed to reach object")
    
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
