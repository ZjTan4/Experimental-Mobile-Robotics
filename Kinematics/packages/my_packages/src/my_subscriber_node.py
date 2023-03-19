#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from duckietown.dtros import DTROS, NodeType

class MySubscriberNode(DTROS):
    def __init__(self, node_name) -> None:
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.sub = rospy.Subscriber('~chatter', String, self.callback)
    
    def callback(self, data):
        rospy.loginfo("I heard {}".format(data.data))

if __name__ == "__main__":
    node = MySubscriberNode(node_name="my_subscriber_node")
    rospy.spin()
