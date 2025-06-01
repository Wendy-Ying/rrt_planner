#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, JointConstraint

def init():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("gen3_lite_moveit_script", anonymous=True)
    rospy.sleep(1.0)

    group = MoveGroupCommander("arm", robot_description="my_gen3_lite/robot_description", ns="/my_gen3_lite")
    scene = PlanningSceneInterface(ns="/my_gen3_lite")
    robot = moveit_commander.RobotCommander(robot_description="my_gen3_lite/robot_description")

    group.set_planning_time(10)
    group.allow_replanning(True)
    group.set_num_planning_attempts(1000)
    group.set_planner_id("RRTConnectkConfigDefault")
    group.set_max_velocity_scaling_factor(0.7)
    group.set_max_acceleration_scaling_factor(0.7)

    limit_joint(group, "joint_0", target_position=0, tolerance_above=1.8, tolerance_below=-0.2)

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
    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    if success:
        print(f"Goal reached: x={x}, y={y}, z={z}")

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


def main():
    group, scene, robot = init()

    scene.remove_world_object()
    rospy.sleep(1.0)
    add_obstacle(scene, 0.35, -0.05, 0.07, 0.12, 0.12, 0.2)

    pose_target = group.get_current_pose().pose
    print(f"Current pose: x={pose_target.position.x}, y={pose_target.position.y}, z={pose_target.position.z}")
    
    set_goal(group, 0.2, 0.25, 0.03)
    set_goal(group, 0.5, -0.03, 0.03)
    
    group.set_named_target("home")
    group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

if __name__ == "__main__":
    main()
