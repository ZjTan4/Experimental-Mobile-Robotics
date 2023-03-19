#!/usr/bin/env python3
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped, Pose2DStamped
from std_msgs.msg import Header, Float32, String
from duckietown_msgs.srv import ChangePattern

import rosbag
import time 

class OdometryNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(OdometryNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")

        # Get static parameters
        self._radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 0.0318)
        self._baseline = rospy.get_param(f'/{self.veh_name}/kinematics_node/baseline', 0.1)
        # self._baseline = 0.12
        
        self.cache = [None, None]
        self.d = [0, 0]
        self.distance = [0, 0]      # left, right
        self.coordinate = [0, 0, 0] # x, y, theta
        self.theta_cache = 0

        # self.counter = [0, 0]

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks_left = rospy.Subscriber(
            "/{}/left_wheel_encoder_node/tick".format(self.veh_name), 
            WheelEncoderStamped, 
            self.cb_encoder_data, 
            callback_args="left",
        )
        self.sub_encoder_ticks_right = rospy.Subscriber(
            "/{}/right_wheel_encoder_node/tick".format(self.veh_name), 
            WheelEncoderStamped, 
            self.cb_encoder_data, 
            callback_args="right"
        )

        # Publishers
        self.pub_integrated_distance_left = rospy.Publisher("/{}/odometry_node/integrated_distance_left".format(self.veh_name), Float32, queue_size=1)
        self.pub_integrated_distance_right = rospy.Publisher("/{}/odometry_node/integrated_distance_right".format(self.veh_name), Float32, queue_size=1)
        self.pub_executed_commands = rospy.Publisher("/{}/wheels_driver_node/wheels_cmd".format(self.veh_name), WheelsCmdStamped, queue_size=1)

        # proxy
        led_service = '/{}/led_node/led_pattern'.format(self.veh_name)
        rospy.wait_for_service(led_service)
        self.led_pattern = rospy.ServiceProxy(led_service, ChangePattern)

        # self.bag = rosbag.Bag("/data/bags/log.bag", 'w')
        self.log("Initialized")

    def cb_encoder_data(self, msg, wheel):
        """ Update encoder distance information from ticks.
        """
        if rospy.is_shutdown():
            return
        wheel_idx = 0 if wheel == "left" else 1
        
        # self.counter[wheel_idx] = (self.counter[wheel_idx] + 1) % 100

        # if self.counter[wheel_idx] % 30 == 0: # lower the update frequency
        # compute distance
        N_tick = msg.data - self.cache[wheel_idx] if self.cache[wheel_idx] is not None else 0
        self.cache[wheel_idx] = msg.data
        N_total = msg.resolution
        self.d[wheel_idx] = 2 * np.pi * self._radius * N_tick / N_total
        self.distance[wheel_idx] += self.d[wheel_idx]
        # rospy.loginfo("The wheels travel {} | {} meters. ".format(*self.distance))

        # publish distance
        publisher = self.pub_integrated_distance_left if wheel_idx == 0 else self.pub_integrated_distance_right
        distance_msg = Float32()
        distance_msg.data = self.distance[wheel_idx]
        if not rospy.is_shutdown():
            publisher.publish(distance_msg)

        # compute coordinate
        velocity = (self.d[0] + self.d[1]) / 2.0
        theta_delta = (self.d[1] - self.d[0]) / self._baseline
        self.coordinate[0] += velocity * np.cos(self.coordinate[2])             # x
        self.coordinate[1] += velocity * np.sin(self.coordinate[2])             # y
        self.coordinate[2] = (self.coordinate[2] + theta_delta) % (2 * np.pi)   # theta -> [0, 2pi]
        self.theta_cache += theta_delta
        # rospy.loginfo("{} | {} | {} ".format(*self.coordinate))
        # self.write_to_bag(self.robot_frame_to_world_frame(self.coordinate), msg.header.stamp) 

    def move(self, vel_left, vel_right):
        header = Header()
        msg = WheelsCmdStamped()
        msg.header = header
        msg.vel_left = vel_left
        msg.vel_right = vel_right
        self.pub_executed_commands.publish(msg)

    def stop(self, seconds, rate):
        self.change_led_lights("red")
        start_time = time.time()
        while time.time() - start_time < seconds:
            rate.sleep()
            self.move(0, 0)
    
    def forward(self, desired_distance, rate, speed_offset=[0, 0]):
        # forward
        self.clear_distance()
        self.change_led_lights("green")
        while not rospy.is_shutdown() and (self.distance[0] < desired_distance and self.distance[1] < desired_distance):
            # rospy.loginfo("The wheels travel {} | {} meters. ".format(*self.distance))
            self.move(0.4 + speed_offset[0], 0.4 + speed_offset[1])
            rate.sleep()

    def backward(self, desired_distance, rate, speed_offset=[0, 0]):
        # backward
        self.clear_distance()
        self.change_led_lights("yellow")
        while not rospy.is_shutdown() and (self.distance[0] > -desired_distance and self.distance[1] > -desired_distance):
            # rospy.loginfo("The wheels travel {} | {} meters. ".format(*self.distance))
            self.move(-(0.4 + speed_offset[0]), -(0.4 + speed_offset[1]))
            rate.sleep()

    def rotate(self, angle, rate, clockwise):
        # rotate clockwise for 90 degrees
        clockwise = 1 if clockwise else -1
        self.change_led_lights("blue")
        orig_theta = self.theta_cache
        while np.abs(self.theta_cache - orig_theta) < angle:
            self.move(0.45 * clockwise, -0.45 * clockwise)
            rate.sleep()
    
    def clockwiseCircularMovement(self, radius, rate, propotion, speed_ratio=None):
        speed_ratio = (radius - self._baseline / 2) / (radius + self._baseline / 2) if speed_ratio is None else speed_ratio
        self.change_led_lights("purple")
        self.clear_distance()
        left_desired_distance = (2 * radius + self._baseline) * np.pi * propotion
        while not rospy.is_shutdown() and self.distance[0] < left_desired_distance:
            self.move(0.4, 0.4 * speed_ratio)
            rate.sleep()

    def write_to_bag(self, coordinate, stamp):
        # write coordinate to self.bag 
        # header = Header()
        # header.stamp = stamp
        # msg = Pose2DStamped()
        # msg.header = header
        # msg.x = coordinate[0]
        # msg.y = coordinate[1]
        # msg.theta = coordinate[2]

        # rospy.loginfo("{} | {} | {}".format(*coordinate))
        pass
        # x = Float32()
        # x.data = coordinate[0]
        # y = Float32()
        # y.data = coordinate[1]
        # theta = Float32()
        # theta.data = coordinate[2]
        # self.bag.write("x", x)
        # self.bag.write("y", y)
        # self.bag.write("theta", theta)

    def robot_frame_to_world_frame(self, coordinate):
        '''
        Matrix (Robot -> World):
        sin(t)  -cos(t) 0.32
        cos(t)   sin(t) 0.32
        0        0      1
        
        returns a np array
        '''
        homo_coord = np.array([coordinate[0], coordinate[1], 1.0])
        theta = coordinate[2]
        tranfromation_matrix = np.array([
            [np.sin(theta), -np.cos(theta), 0.32], 
            [np.cos(theta),  np.sin(theta), 0.32], 
            [0            ,  0            , 1],
        ])
        world_frame = tranfromation_matrix @ homo_coord
        world_theta = coordinate[2] + np.pi / 2
        world_frame[2] = world_theta
        return world_frame

    def change_led_lights(self, color):
        '''
        Sends msg to service server
        Colors:
            "off": [0,0,0],
            "white": [1,1,1],
            "green": [0,1,0],
            "red": [1,0,0],
            "blue": [0,0,1],
            "yellow": [1,0.8,0],
            "purple": [1,0,1],
            "cyan": [0,1,1],
            "pink": [1,0,0.5],
            "shutdown": shutdown led node
        '''
        msg = String()
        msg.data = color
        self.led_pattern(msg)

    def clear_distance(self):
        self.distance = [0, 0]

    def run(self):        
        # init
        rate = rospy.Rate(10)

        # straight forwards
        # while not rospy.is_shutdown() and (self.distance[0] < 1.25 and self.distance[1] < 1.25):
        #     # rospy.loginfo("The wheels travel {} | {} meters. ".format(*self.distance))
        #     self.move(0.2, 0.2)
        #     rate.sleep()
        
        # straight backwards
        # self.change_led_lights("green")
        # while not rospy.is_shutdown() and (self.distance[0] > 0 or self.distance[1] > 0):
        #     # rospy.loginfo("The wheels travel {} | {} meters. ".format(*self.distance))
        #     self.move(-0.4, -0.4)
        #     rate.sleep()

        # clockwise rotation
        # self.clear_distance()
        # angle = 90
        # desired_distance = np.pi * wheelspan * (angle / 360)

        # drive clockwise in circle (not very successful, it now follows a eclipse track somehow)
        # circle_radius = 0.16
        # speed = 0.2
        # deduction_rate = 0.5
        # while not rospy.is_shutdown():
        #     # and self.distance[0] < 2 * np.pi * circle_radius:
        #     self.move(speed, speed * (circle_radius - wheelspan) / circle_radius * deduction_rate)
        #     rate.sleep()

        # START (State 2 & 3)
        self.stop(seconds=3, rate=rate)
        angle_offset1 = -0.45
        self.rotate(angle=np.pi / 2 + angle_offset1, rate=rate, clockwise=True)

        self.stop(seconds=3, rate=rate)
        distance_offset1 = -0.15
        self.forward(desired_distance=1.3 + distance_offset1, rate=rate, speed_offset=[0.0, 0.0])

        self.stop(seconds=3, rate=rate)
        angle_offset2 = -0.015
        self.rotate(angle=np.pi / 2 + angle_offset2, rate=rate, clockwise=False)

        self.stop(seconds=3, rate=rate)
        distance_offset2 = -0.17
        self.forward(desired_distance=1.3 + distance_offset2, rate=rate, speed_offset=[0.0, 0.0])

        self.stop(seconds=3, rate=rate)
        angle_offset3 = -0.1
        self.rotate(angle=np.pi / 2 + angle_offset3, rate=rate, clockwise=False)

        self.stop(seconds=3, rate=rate)
        distance_offset3 = -0.15
        self.forward(desired_distance=1.3 + distance_offset3, rate=rate, speed_offset=[0.0, 0.0])

        self.stop(seconds=3, rate=rate)
        angle_offset4 = -0.01
        self.rotate(angle=np.pi / 2 + angle_offset4, rate=rate, clockwise=True)

        self.stop(seconds=3, rate=rate)
        distance_offset4 = -0.15
        self.backward(desired_distance=1.3 + distance_offset4, rate=rate, speed_offset=[0.0, 0.0])
        # END (State 2 & 3)

        # State 4
        self.stop(seconds=3, rate=rate)
        self.forward(desired_distance=0.5, rate=rate, speed_offset=[0, 0])
        self.clockwiseCircularMovement(radius=0.3, rate=rate, propotion=1.2, speed_ratio=0.4)
        self.forward(desired_distance=0.5, rate=rate, speed_offset=[0, 0])


        # stop
        self.stop(seconds=3, rate=rate)

        # self.bag.close()
        rate.sleep()
        print("Done everything")
        self.change_led_lights("shutdown")
        rospy.signal_shutdown("Done everything")


if __name__ == '__main__':
    node = OdometryNode(node_name='my_odometry_node')
    # Keep it spinning to keep the node alive
    node.run()
    rospy.spin()
    rospy.loginfo("wheel_encoder_node is up and running...")