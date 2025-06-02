#!/usr/bin/env python3
import numpy as np
import time
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
import pid_angle_control
import utilities

def generate_path(start, end, num_points=20):
    """Generate a smooth path from start to end position"""
    path = []
    for i in range(len(start)):
        joint_angles = np.linspace(start[i], end[i], num_points)
        path.append(joint_angles)
    return np.array(path).T  # Convert to shape (num_points, num_joints)

def execute_test_path(base, path, path_name=""):
    """Execute a test path and measure performance"""
    print(f"\nExecuting {path_name}...")
    start_time = time.time()
    
    # Convert to radians
    path_rad = path * np.pi / 180.0
    
    # Execute path
    success = pid_angle_control.execute_path(base, path_rad)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Path execution {'succeeded' if success else 'failed'}")
    print(f"Duration: {duration:.2f} seconds")
    return success, duration

def main():
    # Path 1: Initial position to grasp position
    path1_start = np.array([357, 21, 150, 272, 320, 273])  # Initial pose
    path1_end = np.array([320, 45, 130, 260, 300, 280])    # Grasp pose
    path1_waypoints = generate_path(path1_start, path1_end, 20)
    path1 = np.vstack([path1_waypoints, path1_end])  # Add endpoint

    # Path 2: Grasp position to place position
    path2_start = path1_end  # Start from previous endpoint
    path2_end = np.array([280, 60, 110, 240, 280, 290])    # Place pose
    path2_waypoints = generate_path(path2_start, path2_end, 20)
    path2 = np.vstack([path2_waypoints, path2_end])  # Add endpoint

    print("Test paths generated:")
    print(f"Path 1: {len(path1)} points, {path1[0]} → {path1[-1]}")
    print(f"Path 2: {len(path2)} points, {path2[0]} → {path2[-1]}")

    # Setup connection
    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            
            # Execute first path
            success1, time1 = execute_test_path(base, path1, "Path 1 (Initial → Grasp)")
            if not success1:
                print("Failed to execute path 1")
                return
                
            # Pause to let robot stabilize
            print("\nPausing for stability...")
            time.sleep(2.0)
            
            # Execute second path
            success2, time2 = execute_test_path(base, path2, "Path 2 (Grasp → Place)")
            if not success2:
                print("Failed to execute path 2")
                return
            
            # Print overall results
            print("\nTest completed!")
            print(f"Total execution time: {time1 + time2:.2f} seconds")
            print("All paths executed successfully")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
