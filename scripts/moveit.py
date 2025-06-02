#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, JointConstraint, PositionConstraint
from shape_msgs.msg import SolidPrimitive

def init():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("gen3_lite_moveit_script", anonymous=True)
    rospy.sleep(1.0)

    group = MoveGroupCommander("arm", robot_description="my_gen3_lite/robot_description", ns="/my_gen3_lite")
    scene = PlanningSceneInterface(ns="/my_gen3_lite")
    robot = moveit_commander.RobotCommander(robot_description="my_gen3_lite/robot_description")

    group.set_planning_time(10)
    group.allow_replanning(True)
    group.set_num_planning_attempts(500)
    group.set_planner_id("RRTstarkConfigDefault")
    group.set_max_velocity_scaling_factor(0.7)
    group.set_max_acceleration_scaling_factor(0.7)

    limit_joint(group, "joint_1", target_position=0, tolerance_above=1, tolerance_below=0.2)
    limit_joint(group, "joint_2", target_position=0, tolerance_above=0.3, tolerance_below=1.5)
    limit_joint(group, "joint_3", target_position=1.75, tolerance_above=1.2, tolerance_below=1.6)

    set_cartesian_bounds(group)

    return group, scene, robot

def add_obstacle(scene, x, y, z, size_x, size_y, size_z):
    box_pose = PoseStamped()
    box_pose.header.frame_id = "base_link"
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.x = x
    box_pose.pose.position.y = y
    box_pose.pose.position.z = z
    box_name = "box"
    scene.add_box(box_name, box_pose, size=(size_x, size_y, size_z))

    rospy.sleep(1.0)

def set_goal(group, x, y, z):
    pose_target = group.get_current_pose().pose
    pose_target.position.x = x
    pose_target.position.y = y
    pose_target.position.z = z
    group.set_pose_target(pose_target)
    while not rospy.is_shutdown():
        success, trajectory, _, _ = group.plan()
        if success:
            print("Trajectory planned successfully")
            joint_trajectory = []
            for point in trajectory.joint_trajectory.points:
                joint_trajectory.append(list(point.positions))
            return np.array(joint_trajectory)
        else:
            print("Failed to plan")
            return None

def go_home(group, mode="A"):
    goal_init = np.array([0, 21, 150, 272, 320, 273]) / 180 * np.pi
    goal_final = np.array([0, 343, 75, 0, 300, 0]) / 180 * np.pi
    while not rospy.is_shutdown():
        if mode == "A":
            group.set_joint_value_target(goal_init)
            success, trajectory, _, _ = group.plan()
            if success:
                print("Trajectory planned successfully")
                joint_trajectory = []
                for point in trajectory.joint_trajectory.points:
                    joint_trajectory.append(list(point.positions))
                return np.array(joint_trajectory)
            else:
                print("Failed to plan")
                return None
        if mode == "B":
            group.set_joint_value_target(goal_final)
            success, trajectory, _, _ = group.plan()
            if success:
                print("Trajectory planned successfully")
                joint_trajectory = []
                for point in trajectory.joint_trajectory.points:
                    joint_trajectory.append(list(point.positions))
                return np.array(joint_trajectory)
            else:
                print("Failed to plan")
                return None
        if mode == "C":
            group.set_named_target("home")
            success, trajectory, _, _ = group.plan()
            if success:
                print("Trajectory planned successfully")
                joint_trajectory = []
                for point in trajectory.joint_trajectory.points:
                    joint_trajectory.append(list(point.positions))
                return np.array(joint_trajectory)
            else:
                print("Failed to plan")
                return None

def limit_joint(group, joint_name, target_position, tolerance_above=0.1, tolerance_below=0.1, weight=1.0):
    constraint = JointConstraint()
    constraint.joint_name = joint_name
    constraint.position = target_position
    constraint.tolerance_above = tolerance_above
    constraint.tolerance_below = tolerance_below
    constraint.weight = weight

    constraints = Constraints()
    constraints.joint_constraints.append(constraint)
    
    group.set_path_constraints(constraints)

def set_cartesian_bounds(group):
    constraints = Constraints()
    
    position_constraint = PositionConstraint()
    position_constraint.header.frame_id = "base_link"
    position_constraint.link_name = group.get_end_effector_link()
    
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [1.0, 0.8, 1.0]
    
    box_pose = PoseStamped()
    box_pose.header.frame_id = "base_link"
    box_pose.pose.position.x = 0.25
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = 0.3
    box_pose.pose.orientation.w = 1.0
    
    position_constraint.constraint_region.primitives.append(box)
    position_constraint.constraint_region.primitive_poses.append(box_pose.pose)
    position_constraint.weight = 1.0
    
    constraints.position_constraints.append(position_constraint)
    
    group.set_path_constraints(constraints)

def main():
    group, scene, robot = init()

    scene.remove_world_object()
    rospy.sleep(1.0)
    add_obstacle(scene, 0.35, -0.05, 0.07, 0.12, 0.12, 0.2)

    pose_target = group.get_current_pose().pose
    print(f"Current pose: x={pose_target.position.x}, y={pose_target.position.y}, z={pose_target.position.z}")
    
    set_goal(group, 0.2, 0.25, 0.01)
    set_goal(group, 0.5, -0.03, 0.01)
    
    group.set_named_target("home")
    group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

if __name__ == "__main__":
    main()
