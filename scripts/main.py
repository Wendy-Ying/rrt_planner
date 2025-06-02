#!/usr/bin/env python3
import rospy
import numpy as np
from datetime import datetime
import time
import threading

from perception import init_realsense, detect, process_frame
from moveit import init, add_obstacle, set_goal, go_home
from kinematics import NLinkArm
from optimize import BSplineOptimizer
from visualize import plot_cartesian_trajectory

import pid_angle_control
import utilities
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

obstacle = None
lock = threading.Lock()
obstacle_updated_event = threading.Event()
stop_flag = threading.Event()

def renew_listener(pipeline, align, prev, threshold=0.05, stable_duration=1.0):
    global obstacle
    prev_obstacle = prev
    candidate_obstacle = None
    candidate_start_time = None

    while not stop_flag.is_set():
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        new_obstacle = process_frame(color_image, depth_frame, mode='replan')

        if new_obstacle is None:
            time.sleep(0.05)
            continue

        diff = np.linalg.norm(np.array(new_obstacle) - np.array(prev_obstacle))

        if diff > threshold:
            # New potential value, check if it's the same as last candidate
            if candidate_obstacle is None or np.linalg.norm(np.array(new_obstacle) - np.array(candidate_obstacle)) > threshold:
                candidate_obstacle = new_obstacle
                candidate_start_time = time.time()
            else:
                # Candidate is stable, check if it's stable long enough
                if time.time() - candidate_start_time >= stable_duration:
                    with lock:
                        obstacle = candidate_obstacle
                        obstacle_updated_event.set()
                    prev_obstacle = candidate_obstacle
                    on_obstacle_changed(obstacle)
                    candidate_obstacle = None
                    candidate_start_time = None
        else:
            candidate_obstacle = None
            candidate_start_time = None

        time.sleep(0.05)

def on_obstacle_changed(obstacle):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [Trigger] Responding immediately to obstacle change!")

def main():
    global obstacle
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

    obj, goal, obstacle = detect(pipeline, align)

    print(f"Detected object: {obj}")
    print(f"Goal position: {goal}")
    print(f"Obstacle position: {obstacle}")

    group, scene, _ = init()

    scene.remove_world_object()
    rospy.sleep(1.0)
    add_obstacle(scene, obstacle[0], obstacle[1]+0.02, 0.01, 0.12, 0.12, 0.2)

    optimizer = BSplineOptimizer(robot, degree=3, num_points=20)

    t = threading.Thread(target=renew_listener, args=(pipeline, align, obstacle))
    t.start()

    args = utilities.parseConnectionArguments()

    try:
        smooth_path1, smooth_path2, smooth_path3 = None, None, None
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            pid_angle_control.send_gripper_command(base, 0.2)

        path = set_goal(group, obj[0], obj[1]+0.02, 0.03)
        if path is not None:
            smooth_path1 = optimizer.optimize(path)
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, smooth_path1)
                pid_angle_control.send_gripper_command(base, 0.8)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")

        path = set_goal(group, goal[0], goal[1], 0.02)
        if path is not None:
            smooth_path2 = optimizer.optimize(path)
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, smooth_path2)
                pid_angle_control.send_gripper_command(base, 0.3)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")

        path = go_home(group, mode="C")
        if path is not None:
            smooth_path3 = optimizer.optimize(path)
            with utilities.DeviceConnection.createTcpConnection(args) as router:
                base = BaseClient(router)
                success = pid_angle_control.execute_path(base, smooth_path3)
                if not success:
                    print("Path execution failed")
                else:
                    print("Path execution completed successfully")

        if smooth_path1 is not None and smooth_path2 is not None and smooth_path3 is not None:
            combined_path = np.vstack((smooth_path1, smooth_path2, smooth_path3))
            plot_cartesian_trajectory(combined_path, robot)
    
    except KeyboardInterrupt:
        print("Script interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    stop_flag.set()
    t.join()
    pipeline.stop()
    

if __name__ == "__main__":
    main()
