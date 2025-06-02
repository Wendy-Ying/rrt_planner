#!/usr/bin/env python3
import rospy
import numpy as np
import math
from perception import init_realsense, detect
from moveit import init, add_obstacle, set_goal, go_home
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

import pid_angle_control
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

def main():
    pipeline, align = init_realsense()
    
    joint_limits = np.array([(205, 150), (205, 150), (210, 150), (210, 145), (215, 140), (210, 150)]) / 180 * np.pi

    dh_params = [
        [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
        [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
        [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
        [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
    ]

    robot = NLinkArm(dh_params, joint_limits)

    # Get positions and ensure they are numpy arrays
    obj, goal, obstacle = detect(pipeline, align)

    print(f"Detected object: {obj}")
    print(f"Goal position: {goal}")
    print(f"Obstacle position: {obstacle}")

    group, scene, _ = init()

    scene.remove_world_object()
    rospy.sleep(1.0)
    add_obstacle(scene, obstacle[0], obstacle[1], 0.07, 0.12, 0.12, 0.14)

    optimizer = BSplineOptimizer(robot, degree=3, num_points=5)

    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            pid_angle_control.send_gripper_command(base, 0)

        path = set_goal(group, obj[0], obj[1]+0.01, 0.01)
        if path is not None:
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, path)
                pid_angle_control.send_gripper_command(base, 0.8)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")
            plot_cartesian_trajectory(path, robot)

        path = set_goal(group, obj[0], obj[1]+0.01, 0.1)
        if path is not None:
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, path)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")
            plot_cartesian_trajectory(path, robot)

        path = set_goal(group, goal[0], goal[1], 0.015)
        if path is not None:
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, path)
                pid_angle_control.send_gripper_command(base, 0.1)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")
            plot_cartesian_trajectory(path, robot)

        path = go_home(group, mode="C")
        if path is not None:
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, path)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")
            plot_cartesian_trajectory(path, robot)
    
    except KeyboardInterrupt:
        print("Script interrupted by user")
        pipeline.stop()
    except Exception as e:
        print(f"An error occurred: {e}")
        pipeline.stop()
    finally:
        pipeline.stop()
    

if __name__ == "__main__":
    main()
