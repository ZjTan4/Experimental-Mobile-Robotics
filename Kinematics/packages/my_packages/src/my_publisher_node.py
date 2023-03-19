#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType

class MyPublisherNode(DTROS):
    def __init__(self, node_name) -> None:
        super(MyPublisherNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.pub = rospy.Publisher('~chatter', String, queue_size=10)
    
    def run(self):
        rate = rospy.Rate(1)
        # while not rospy.is_shutdown():
        #     message = "How are your, world? from {}".format(os.environ["VEHICLE_NAME"])
        #     rospy.loginfo("Publishing Message: {}".format(message))
        #     self.pub.publish(message)
        #     rate.sleep()
        rate.sleep()
        message = "How are your, world? from {}".format(os.environ["VEHICLE_NAME"])
        rospy.loginfo("Publishing Message: {}".format(message))
        self.pub.publish(message)

if __name__ == "__main__":
    node = MyPublisherNode(node_name="my_publisher_node")
    node.run()
    rospy.spin()
