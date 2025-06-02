#!/usr/bin/env python3
import numpy as np
import time
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kinematics import NLinkArm
import pid_angle_control
import utilities

def generate_cartesian_path(start, end, num_points=20):
    """Generate a straight line path in Cartesian space"""
    path = []
    for i in range(3):  # x, y, z coordinates
        coordinates = np.linspace(start[i], end[i], num_points)
        path.append(coordinates)
    return np.array(path).T  # Convert to shape (num_points, 3)

def cartesian_to_joint_path(robot, cartesian_path):
    """Convert Cartesian path to joint angles using inverse kinematics"""
    joint_path = []
    last_angles = None  # Store last solution for continuity
    
    for point in cartesian_path:
        # If no previous solution, try multiple initial guesses
        if last_angles is None:
            initial_guesses = [
                np.array([0, 0, 0, 0, 0, 0]),
                np.array([np.pi/4, 0, 0, 0, 0, 0]),
                np.array([-np.pi/4, 0, 0, 0, 0, 0])
            ]
            
            for guess in initial_guesses:
                solution = robot.inverse_kinematics(point, guess)
                if solution is not None:
                    joint_angles = solution
                    last_angles = solution
                    break
        else:
            # Use last solution as initial guess
            solution = robot.inverse_kinematics(point, last_angles)
            if solution is not None:
                joint_angles = solution
                last_angles = solution
            else:
                print(f"Failed to find IK solution for point {point}")
                return None
        
        if last_angles is None:
            print(f"Failed to find initial IK solution for point {point}")
            return None
            
        joint_path.append(joint_angles)
    
    return np.array(joint_path)

def execute_test_path(base, path, path_name=""):
    """Execute a test path and measure performance"""
    print(f"\nExecuting {path_name}...")
    start_time = time.time()
    
    # Execute path (path is already in radians)
    success = pid_angle_control.execute_path(base, path)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Path execution {'succeeded' if success else 'failed'}")
    print(f"Duration: {duration:.2f} seconds")
    return success, duration

def main():
    # Define robot parameters
    joint_limits = np.array([(205, -150), (205, -150), (210, -150), 
                            (210, -145), (215, -140), (210, -150)]) / 180 * np.pi
    dh_params = np.array([
        [0.0,        0.0 / 1000,   243.3 / 1000,  0.0],
        [np.pi/2,    0.0 / 1000,    30.0 / 1000,  np.pi/2],
        [np.pi,    280.0 / 1000,    20.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,   245.0 / 1000,  np.pi/2],
        [np.pi/2,    0.0 / 1000,    57.0 / 1000,  0.0],
        [-np.pi/2,   0.0 / 1000,   235.0 / 1000, -np.pi/2]
    ])

    # Initialize robot
    robot = NLinkArm(dh_params, joint_limits)

    # Define key positions
    obj = np.array([0.166, 0.148, 0], dtype=float)      # Object position
    obstacle = np.array([0.367, 0.11, 0], dtype=float)  # Obstacle position
    goal = np.array([0.552, 0.131, 0], dtype=float)     # Goal position
    
    # Add z-offset for practical motion
    z_offset = 0.05
    obj = obj + np.array([0, 0, z_offset])
    obstacle = obstacle + np.array([0, 0, z_offset])
    goal = goal + np.array([0, 0, z_offset])

    # Generate Cartesian paths
    # Path 1: From obstacle point (higher) to object
    path1_cartesian = generate_cartesian_path(obstacle, obj)
    
    # Path 2: From object to goal
    path2_cartesian = generate_cartesian_path(obj, goal)

    # Convert to joint paths
    print("Converting Cartesian paths to joint angles...")
    path1 = cartesian_to_joint_path(robot, path1_cartesian)
    if path1 is None:
        print("Failed to generate Path 1")
        return
        
    path2 = cartesian_to_joint_path(robot, path2_cartesian)
    if path2 is None:
        print("Failed to generate Path 2")
        return

    print("\nPaths generated successfully")
    print(f"Path 1: {len(path1)} points")
    print(f"Path 2: {len(path2)} points")

    # Setup connection
    args = utilities.parseConnectionArguments()

    try:
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            
            # Execute first path
            success1, time1 = execute_test_path(base, path1, "Path 1 (Obstacle → Object)")
            if not success1:
                print("Failed to execute path 1")
                return
                
            # Pause to let robot stabilize
            print("\nPausing for stability...")
            time.sleep(2.0)
            
            # Execute second path
            success2, time2 = execute_test_path(base, path2, "Path 2 (Object → Goal)")
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
