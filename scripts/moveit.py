#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from moveit_commander import MoveGroupCommander, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("gen3_lite_moveit_script", anonymous=True)

    rospy.sleep(2.0)

    group = MoveGroupCommander("arm")
    scene = PlanningSceneInterface()
    robot = moveit_commander.RobotCommander()

    group.set_planning_time(10)
    group.allow_replanning(True)
    group.set_num_planning_attempts(5)
    group.set_planner_id("RRTConnectkConfigDefault")

    # 清除场景中旧的障碍物
    scene.remove_world_object("box")
    rospy.sleep(1.0)

    # 添加一个障碍物：一个立方体（比如放在机器人前面）
    box_pose = PoseStamped()
    box_pose.header.frame_id = robot.get_planning_frame()  # 通常是 "base_link"
    box_pose.pose.position.x = 0.5
    box_pose.pose.position.y = 0.0
    box_pose.pose.position.z = 0.2
    box_pose.pose.orientation.w = 1.0

    scene.add_box("box", box_pose, size=(0.3, 0.3, 0.4))
    rospy.sleep(1.0)

    # 获取当前末端位姿，并设置新目标
    pose_target = group.get_current_pose().pose
    pose_target.position.x += 0.1
    group.set_pose_target(pose_target)

    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    if success:
        rospy.loginfo("Motion executed successfully while avoiding obstacles.")
    else:
        rospy.logwarn("Motion failed due to collision or planning error.")

if __name__ == "__main__":
    main()
