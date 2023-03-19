#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, Image
from duckietown.dtros import DTROS, NodeType

import cv2
from cv_bridge import CvBridge


class MyCameraSubscriberNode(DTROS):
    def __init__(self, node_name) -> None:
        super(MyCameraSubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.sub = rospy.Subscriber('/{}/camera_node/image/compressed'.format(os.environ["VEHICLE_NAME"]), CompressedImage, self.callback)
        self.pub = rospy.Publisher('/{}/my_camera_subscriber_node/image/compressed'.format(os.environ["VEHICLE_NAME"]), CompressedImage, queue_size=10)
        
    
    def callback(self, data):
        bridge = CvBridge()
        cv_img = bridge.compressed_imgmsg_to_cv2(data, desired_encoding="passthrough")
        # rospy.loginfo("Camera sent an image of size {}".format(cv_img.shape))
        self.pub.publish(data)
        

if __name__ == "__main__":
    node = MyCameraSubscriberNode(node_name="my_subscriber_node")
    rospy.spin()
