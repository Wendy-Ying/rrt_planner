#!/usr/bin/env python3
import numpy as np
import time
import sys
import os
import argparse
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

class PIDController:
    def __init__(self, Kp, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_sum = 0
        self.error_last = 0

    def control(self, ref, fdb):
        error = ref - fdb
        self.error_sum = self.error_sum + error
        error_diff = error - self.error_last
        self.error_last = error
        control = self.Kp * error + self.Ki * self.error_sum + self.Kd * error_diff
        return control

def warp_to_range(angle, min_value=-180, max_value=180):
    """Normalize angle to specified range"""
    while angle > max_value:
        angle = angle - 360
    while angle < min_value:
        angle = angle + 360
    return angle

def send_gripper_command(base, value):
    """
    Control the gripper
    base: robot control interface
    value: gripper position (0.0 to 1.0, where 0.0 is closed and 1.0 is open)
    """
    print("Performing gripper test in position...")
    gripper_command = Base_pb2.GripperCommand()
    finger = gripper_command.gripper.finger.add()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = max(0.0, min(1.0, value))  # Ensure value is between 0 and 1
    base.SendGripperCommand(gripper_command)

def check_joint_limits(angles):
    """Check if joint angles are within limits"""
    limits = np.array([365, 153, 149, 149, 144, 148])  # Maximum angle limits
    for i, angle in enumerate(angles):
        if abs(angle) > limits[i]:
            print(f"Joint {i+1} angle {angle} exceeds limit {limits[i]}")
            return False
    return True

def move_to_angles(base, target_angles, gripper_value=0.0):
    """
    Control all joints to target angles
    base: robot control interface
    target_angles: list of 6 target angles (degrees)
    gripper_value: gripper position (0.0 to 1.0)
    """
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    if len(target_angles) != 6:
        print("Error: Must provide exactly 6 joint angles")
        return False

    if not check_joint_limits(target_angles):
        return False

    # Initialize PID controllers for each joint with specific gains
    pids = [
        PIDController(Kp=2.0, Ki=0.0, Kd=0.1),  # Joint 1
        PIDController(Kp=1.0, Ki=0.0, Kd=0.1),  # Joint 2
        PIDController(Kp=1.0, Ki=0.0, Kd=0.1),  # Joint 3
        PIDController(Kp=1.5, Ki=0.0, Kd=0.1),  # Joint 4
        PIDController(Kp=1.0, Ki=0.0, Kd=0.1),  # Joint 5
        PIDController(Kp=4.0, Ki=0.0, Kd=0.1)   # Joint 6
    ]
    
    # Normalize target angles
    target_angles = [warp_to_range(angle) for angle in target_angles]
    start_time = time.time()
    
    try:
        # Control gripper
        send_gripper_command(base, gripper_value)
        
        # Joint control loop
        while True:
            # Get current joint angles
            feedback = base.GetMeasuredJointAngles()
            current_angles = [warp_to_range(joint.value) for joint in feedback.joint_angles]
            
            # Calculate speeds for all joints
            speeds = []
            errors = []
            for i in range(6):
                speed = pids[i].control(target_angles[i], current_angles[i])
                speeds.append(speed)
                errors.append(abs(target_angles[i] - current_angles[i]))
            
            # Send velocity commands
            joint_speeds = Base_pb2.JointSpeeds()
            for i in range(6):
                joint_speed = joint_speeds.joint_speeds.add()
                joint_speed.joint_identifier = i
                joint_speed.value = speeds[i]
                joint_speed.duration = 0
            
            base.SendJointSpeedsCommand(joint_speeds)
            
            # Print status
            status = "Joints: " + ", ".join([f"{i+1}:{angle:.1f}" for i, angle in enumerate(current_angles)])
            print(f"{status} | Errors: {max(errors):.2f}", flush=True)
            
            # Check if all joints reached target (error less than 1 degree)
            if all(error < 1.0 for error in errors):
                print("Target position reached")
                return True
                
            # Timeout protection (10 seconds)
            if time.time() - start_time > 10:
                print("Timeout reached")
                return False
                
    except KeyboardInterrupt:
        print("Control interrupted by user")
        return False
    finally:
        # Stop all joint motion
        joint_speeds = Base_pb2.JointSpeeds()
        for i in range(6):
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i
            joint_speed.value = 0
            joint_speed.duration = 0
        base.SendJointSpeedsCommand(joint_speeds)

def execute_path(base, path, gripper_value=0.0):
    """
    Execute a sequence of joint configurations
    base: robot control interface
    path: list of joint angle configurations (list of 6-element arrays)
    gripper_value: gripper position (0.0 to 1.0)
    """
    success = True
    for target_angles in path:
        # Convert from radians to degrees
        angles_deg = np.array(target_angles) * 180.0 / np.pi
        success &= move_to_angles(base, angles_deg, gripper_value)
        if not success:
            print("Failed to reach configuration:", angles_deg)
            return False
    return True

def main():
    """
    Main function: setup connection and execute angle control
    Usage: 
    1. Single position control:
       rosrun final_project pid_angle_control.py --ip IP -u USERNAME -p PASSWORD j1 j2 j3 j4 j5 j6 [gripper]
       where j1-j6 are joint angles in degrees
       gripper is gripper position (0.0-1.0, optional)
    
    2. Path execution (internal use):
       This mode is used when called from other scripts like RRT
    """
    # Check if being called from command line or another script
    if len(sys.argv) > 1:  # Command line mode
        parser = argparse.ArgumentParser(description='Control Kinova arm joint angles with PID')
        # Connection arguments
        parser.add_argument("--ip", type=str, help="IP address of destination", default="192.168.1.10")
        parser.add_argument("-u", "--username", type=str, help="username to login", default="admin")
        parser.add_argument("-p", "--password", type=str, help="password to login", default="admin")
        # Joint angles and gripper arguments
        parser.add_argument("joint_angles", type=float, nargs=6, help="6 joint angles in degrees")
        parser.add_argument("gripper", type=float, nargs='?', default=0.0, 
                           help="gripper position (0.0-1.0)")

    try:
        args = parser.parse_args()
        
        # Set up connection
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import utilities
        
        with utilities.DeviceConnection.createTcpConnection(args) as router:
            base = BaseClient(router)
            success = move_to_angles(base, args.joint_angles, args.gripper)
            return 0 if success else 1

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
