#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from lane_controller import LaneController
from sensor_msgs.msg import CompressedImage
import numpy as np

class LaneFollowingNode(DTROS):
    def __init__(self, node_name) -> None:
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")
        self.lane_offset = 135
        self.height_ratio = 0.8
        # Subscriber
        self.sub_image_compressed = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_compressed_image)
        self.compressed_image_cache = None
        # Publisher
        # self.pub_executed_commands = rospy.Publisher("/{}/wheels_driver_node/wheels_cmd".format(self.veh_name), WheelsCmdStamped, queue_size=1)
        self.pub_car_commands = rospy.Publisher(f"/{self.veh_name}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        self.pub_augmented_image = rospy.Publisher(f'/{self.veh_name}/lane_following_node/image/compressed', CompressedImage, queue_size=1)
        # Assistant module
        self.bridge = CvBridge()
        self.controller = LaneController()
    
    def cb_compressed_image(self, compressed_image):
        self.compressed_image_cache = compressed_image
    
    def move(self, v, omega):
        new_cmd = Twist2DStamped()
        new_cmd.header.stamp = rospy.Time.now()
        new_cmd.header.frame_id = "~/car_cmd"
        new_cmd.v = v
        new_cmd.omega = omega
        self.pub_car_commands.publish(new_cmd)
    
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.compressed_image_cache is not None:
                image = self.bridge.compressed_imgmsg_to_cv2(self.compressed_image_cache)
                height, width, _ = image.shape
                # Filter colour
                imhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                #lower = np.array([20, 150, 150], dtype="uint8") 
                #upper = np.array([55, 255, 255], dtype="uint8")
                lower = np.array([13, 70, 170], dtype="uint8") 
                upper = np.array([32, 255, 255], dtype="uint8")
                mask = cv2.inRange(imhsv, lower, upper)
                mask[int(height*self.height_ratio):height, 0:width] = 0
                mask[0:int(height*(1 - self.height_ratio)), 0:width] = 0
                # Find contours -> [np.array of shape (n, 1, 2)]
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
                # Comopute target point
                current_point = [width/2 + self.lane_offset, height/2]
                if len(contours) > 0:
                    # if detect contours, find the one with the largest area
                    contour = max(contours, key=cv2.contourArea) 
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
                else:
                    contour = np.array(current_point).reshape((1, 1, 2))
                target_point = contour.mean(axis=0).squeeze()
                # Call controller to get the next action
                v, omega = self.controller.getNextCommand(target_point, current_point)
                # Send Command
                self.move(v, omega)
                # Publish Image
                self.pub_augmented_image.publish(self.bridge.cv2_to_compressed_imgmsg(image))
            rate.sleep()

if __name__ == "__main__":
    node = LaneFollowingNode(node_name="lane_following_node")
    node.run()
    rospy.spin()
