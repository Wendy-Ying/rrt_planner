#!/usr/bin/env python3
import numpy as np
import time
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
import pid_angle_control
import utilities

def main():
    # Define two paths (angles in degrees, will be converted to radians)
    # Path 1: From initial position to object position
    path1 = np.array([
        # [j1, j2, j3, j4, j5, j6]
        [357, 21, 150, 272, 320, 273],
        [322, 287, 81, 353, 267, 295],          
        [351, 344, 119, 274, 325, 265],
        [357, 21, 150, 272, 320, 273],
    ]) * np.pi / 180.0  # Convert to radians

    # Path 2: From object position to goal position
    path2 = np.array([
        [313, 280, 53, 349, 259, 314],        
        [351, 344, 119, 274, 325, 265],        
        [351, 344, 119, 274, 325, 265],
        [357, 21, 150, 272, 320, 273]         
    ]) * np.pi / 180.0  # Convert to radians

    # Setup connection
    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)

            # Execute first path with open gripper
            # pid_angle_control.send_gripper_command(base, 0.9) 
            print("Executing path to object...")
            success = pid_angle_control.execute_path(base, path1)
            if not success:
                print("Failed to reach object")
                return

            print("\nObject reached, closing gripper...")
            pid_angle_control.send_gripper_command(base, 0.3)  
            time.sleep(3.0)  # Wait for gripper to close

            # Execute second path with closed gripper
            print("\nExecuting path to goal with closed gripper...")
            success = pid_angle_control.execute_path(base, path2)
            if not success:
                print("Failed to reach goal")
                return

            print("\nGoal reached, opening gripper...")
            pid_angle_control.send_gripper_command(base, 0.9)  # Open gripper
            time.sleep(1.0)  # Wait for gripper to open

            print("\nMotion completed successfully")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
