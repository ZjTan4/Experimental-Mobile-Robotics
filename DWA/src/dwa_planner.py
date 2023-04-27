import rospy
import cv2
import os
import numpy as np
import sys

from geometry_msgs.msg import PoseStamped, Twist, Pose, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import tf

class State:
    def  __init__(self, x, y, yaw, velocity, yawrate) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity
        self.yawrate = yawrate

class Window:
    def __init__(self, min_velocity, max_velocity, min_yawrate, max_yawrate) -> None:
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_yawrate = min_yawrate
        self.max_yawrate = max_yawrate

class DWA_Planner:
    def __init__(self) -> None:
        # Args
        self.robot_frame = "base_link"
        self.velocity_resolution = 0.1
        self.yawrate_resolution = 0.1
        self.predict_time = 5.0
        self.frequency = 20
        self.dt = 1 / self.frequency
        self.dist_lower_bound = 0.05
        # DW Args
        self.max_velocity = 0.5
        self.min_velocity = 0.0
        self.max_acceleration = 0.5
        self.max_yawrate = 1.5
        self.max_d_yawrate = 5.0
        # DWA Args
        self.to_goal_gain = 5.0
        self.velocity_gain = 0.3
        self.obstacle_gain = 0.3
        self.goal_threshold = 0.3
        self.turn_direction_threshold = 0.1

        # Utils
        self.listener = tf.TransformListener()
        self.norm = lambda state1, state2: np.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2 + (state1.yaw - state2.yaw)**2)

        # Subscriber
        self.local_goal_sub = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.local_goal_cb)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_cb)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_cb)
        self.target_velocity_sub = rospy.Subscriber("/target_velocity", Twist, self.target_velocity_cb)
        # Subscriber Messages
        self.local_goal = None
        self.scan = None
        self.current_velocity = None
        self.target_velocity = 0.2
        # Publisher
        self.velocity_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.candidate_trajectories_pub = rospy.Publisher("candidate_trajectories", MarkerArray, queue_size=1)
        self.selected_trajectory_pub = rospy.Publisher("selected_trajectory", Marker, queue_size=1)

    
    def local_goal_cb(self, msg):
        self.local_goal = msg
        # print(self.local_goal)

    def transformPose(self, robot_frame, time, pose_in, fixed_frame):
        # Transform pose_in to fixed_frame
        self.listener.waitForTransform(fixed_frame, pose_in.header.frame_id, time, rospy.Duration(4.0))
        pose_fixed_frame = self.listener.transformPose(fixed_frame, pose_in)
        # Transform fixed_frame pose to robot_frame (target_frame)
        self.listener.waitForTransform(robot_frame, fixed_frame, time, rospy.Duration(4.0))
        pose_fixed_frame.header.stamp = time
        pose_out = self.listener.transformPose(robot_frame, pose_fixed_frame)
        # Result Pose
        return pose_out

    def scan_cb(self, msg):
        self.scan = msg
        # print(self.scan)

    def odom_cb(self, msg):
        self.current_velocity = msg.twist.twist
        # print(self.current_velocity)

    def target_velocity_cb(self, msg):
        self.target_velocity = msg.linear.x
        # print(self.target_velocity)

    def dwa_planning(self, dynamic_window, goal, obs_list):
        min_cost = 1e6
        trajectories = []
        best_traj = None

        for v in np.arange(dynamic_window.min_velocity, dynamic_window.max_velocity + self.velocity_resolution, self.velocity_resolution):
            for y in np.arange(dynamic_window.min_yawrate, dynamic_window.max_yawrate + self.yawrate_resolution, self.yawrate_resolution):
                traj = []
                state = State(0, 0, 0, self.current_velocity.linear.x, self.current_velocity.angular.z)
                for _ in np.arange(0, self.predict_time, self.dt):
                    state = self.motion(state, v, y)
                    traj.append(state)
                trajectories.append(traj)

                to_goal_cost = self.calc_to_goal_cost(traj, goal)
                velocity_cost = self.calc_speed_cost(traj, self.target_velocity)
                obstacle_cost = self.calc_obstacle_cost(traj, obs_list)
                final_cost = self.to_goal_gain * to_goal_cost + self.velocity_gain * velocity_cost + self.obstacle_gain * obstacle_cost
                if final_cost <= min_cost:
                    min_cost = final_cost
                    best_traj = traj
        self.visualize_trajectories(trajectories, 0, 1, 0, 1000)
        # No good traj found (every traj's cost is greater than the init min_cost)
        if best_traj is None:
            best_traj = [State(0.0, 0.0, 0.0, self.current_velocity.linear.x, self.current_velocity.angular.z)]
        return best_traj

    def prcoess(self):
        rate = rospy.Rate(self.frequency)
        while not rospy.is_shutdown():
            if self.scan is not None and self.local_goal is not None:
                dynamic_window = self.calc_dynamic_window()

                local_goal = self.transformPose(robot_frame=self.robot_frame, time=rospy.Time(0), pose_in=self.local_goal, fixed_frame=self.local_goal.header.frame_id)
                goal_quaternion = (
                    local_goal.pose.orientation.x, 
                    local_goal.pose.orientation.y, 
                    local_goal.pose.orientation.z, 
                    local_goal.pose.orientation.w, 
                )
                _, _,  yaw = tf.transformations.euler_from_quaternion(goal_quaternion)
                goal = State(local_goal.pose.position.x, local_goal.pose.position.y, yaw, None, None)

                cmd_vel = Twist()
                # print(np.sqrt(goal.x**2 + goal.y**2))
                if np.sqrt(goal.x**2 + goal.y**2) > self.goal_threshold:
                    obs_list = self.scan_to_obs(self.scan)
                    self.scan = None
                    best_traj = self.dwa_planning(dynamic_window, goal, obs_list)

                    cmd_vel.linear.x = best_traj[0].velocity
                    cmd_vel.angular.z = best_traj[0].yawrate

                    self.visualize_trajectory(best_traj, 1, 0, 0)
                else:
                    cmd_vel.linear.x = 0.0
                    cmd_vel.angular.z = min(max(goal.yaw, -self.max_yawrate), self.max_yawrate) if np.fabs(goal.yaw) > self.turn_direction_threshold else 0.0
                self.velocity_pub.publish(cmd_vel)
            rate.sleep()

    def motion(self, state, velocity, yawrate):
        ret_state = State(state.x, state.y, state.yaw, state.velocity, state.yawrate)
        ret_state.yaw += yawrate * self.dt
        ret_state.x += velocity * np.cos(state.yaw) * self.dt
        ret_state.y += velocity * np.sin(state.yaw) * self.dt
        ret_state.velocity = velocity
        ret_state.yawrate = yawrate
        return ret_state

    def scan_to_obs(self, scan):
        obs_list = []
        angle = scan.angle_min
        for r in scan.ranges:
            if r != float("inf"):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                obs_list.append((x, y))
            angle += scan.angle_increment
        return obs_list

    def calc_dynamic_window(self):
        current_velocity = self.current_velocity.linear.x
        current_yawrate = self.current_velocity.angular.z
        return Window(
            min_velocity=max((current_velocity - self.max_acceleration * self.dt), self.min_velocity),
            max_velocity=min((current_velocity + self.max_acceleration * self.dt), self.max_velocity),
            min_yawrate=max((current_yawrate - self.max_d_yawrate * self.dt), -self.max_yawrate), 
            max_yawrate=min((current_yawrate + self.max_d_yawrate * self.dt), self.max_yawrate)
        )

    def calc_to_goal_cost(self, traj, goal):
        last_position = traj[-1]
        return self.norm(last_position, goal)

    def calc_speed_cost(self, traj, target_velocity):
        return np.fabs(target_velocity - np.fabs(traj[-1].velocity))

    def calc_obstacle_cost(self, traj, obs_list):
        min_dist = 1e3
        for state in traj:
            for obs in obs_list:
                obs = State(obs[0], obs[1], state.yaw, None, None)
                dist = self.norm(state, obs)
                # too close
                if dist <= self.dist_lower_bound:
                    cost = 1e6
                    return cost 
                min_dist = min(min_dist, dist)
        cost = 1 / min_dist
        return cost

    def visualize_trajectories(self, trajectories, r, g, b, size):
        v_trajectories = MarkerArray()
        count = 1
        for trajectory in trajectories:
            v_trajectory = self.create_trajectory(trajectory, r, g, b, 0.01, "candidate_trajectories", count)
            v_trajectories.markers.append(v_trajectory)
            count += 1
        while count <= size:
            v_trajectory = Marker()
            # header
            v_trajectory.header.frame_id = self.robot_frame
            v_trajectory.header.stamp = rospy.Time.now()
            v_trajectory.id = count
            v_trajectory.ns = "candidate_trajectories"
            v_trajectory.type = Marker.LINE_STRIP
            v_trajectory.action = Marker.DELETE
            v_trajectory.lifetime = rospy.Duration()
            v_trajectories.markers.append(v_trajectory)
            count += 1
        self.candidate_trajectories_pub.publish(v_trajectories)

    def visualize_trajectory(self, trajectory, r, g, b):
        v_trajectory = self.create_trajectory(trajectory, r, g, b, 0.03, "selected_trajectory")
        self.selected_trajectory_pub.publish(v_trajectory)

    def create_trajectory(self, trajectory, r, g, b, scale, topic, id=None):
        v_trajectory = Marker()
        # header
        v_trajectory.header.frame_id = self.robot_frame
        v_trajectory.header.stamp = rospy.Time.now()
        # color
        v_trajectory.color.r = r
        v_trajectory.color.g = g
        v_trajectory.color.b = b
        v_trajectory.color.a = 0.8
        # marker params 
        v_trajectory.id = id if id is not None else 0
        v_trajectory.ns = topic
        v_trajectory.type = Marker.LINE_STRIP
        v_trajectory.action = Marker.ADD
        v_trajectory.lifetime = rospy.Duration()
        v_trajectory.scale.x = scale
        # components
        pose = Pose()
        pose.orientation.w = 1
        v_trajectory.pose = pose
        for pose in trajectory:
            point = Point()
            point.x = pose.x
            point.y = pose.y
            v_trajectory.points.append(point)
        return v_trajectory
