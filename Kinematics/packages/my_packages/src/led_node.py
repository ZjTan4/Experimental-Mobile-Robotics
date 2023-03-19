#!/usr/bin/env python3

import os
import rospy
from std_msgs.msg import String
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern, ChangePatternResponse
from duckietown_msgs.msg import LEDPattern
from std_msgs.msg import ColorRGBA
from duckietown.dtros import DTROS, NodeType, TopicType

# Reference: https://github.com/duckietown/dt-core/blob/6d8e99a5849737f86cab72b04fd2b449528226be/packages/led_emitter/src/led_emitter_node.py#L254
class LEDNode(DTROS):
    def __init__(self, node_name) -> None:
        '''
        +------------------+------------------------------------------+
        | Index            | Position (rel. to direction of movement) |
        +==================+==========================================+
        | 0                | Front left                               |
        +------------------+------------------------------------------+
        | 1                | Rear left                                |
        +------------------+------------------------------------------+
        | 2                | Top / Front middle                       |
        +------------------+------------------------------------------+
        | 3                | Rear right                               |
        +------------------+------------------------------------------+
        | 4                | Front right                              |
        +------------------+------------------------------------------+
        '''
        super(LEDNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")
        service_name = "/{}/led_emitter_node/set_custom_pattern".format(self.veh_name)
        rospy.wait_for_service(service_name)
        self.setCustomPattern = rospy.ServiceProxy(service_name, SetCustomLEDPattern)
        
        # service_name = "/{}/led_emitter_node/set_pattern".format(self.veh_name)
        # rospy.wait_for_service(service_name)
        # self.setPattern = rospy.ServiceProxy(service_name, ChangePattern)

        # set up publisher
        # an attempt to skip the led_emitter and directly control the led driver
        # self.pub_leds = rospy.Publisher("~led_pattern", LEDPattern, queue_size=1, dt_topic_type=TopicType.DRIVER)
        # self.colors = { 
        #     "off": [0,0,0],
        #     "white": [1,1,1],
        #     "green": [0,1,0],
        #     "red": [1,0,0],
        #     "blue": [0,0,1],
        #     "yellow": [1,0.8,0],
        #     "purple": [1,0,1],
        #     "cyan": [0,1,1],
        #     "pink": [1,0,0.5],
        #     "shutdown": shutdown this node
        # }

        # set up server
        self.shutdown = False
        self.server = rospy.Service('/{}/led_node/led_pattern'.format(self.veh_name), ChangePattern, self.handle_change_led_msg)

    def handle_change_led_msg(self, msg):
        if msg.pattern_name.data == "shutdown":
            self.shutdown = True
            return ChangePatternResponse()

        new_msg = LEDPattern()

        new_msg.color_list = [msg.pattern_name.data] * 5
        new_msg.color_mask = [1,1,1,1,1]
        new_msg.frequency = 0.0
        new_msg.frequency_mask = [0,0,0,0,0]
        self.setCustomPattern(new_msg)

        # color = self.colors[msg.pattern_name.data]
        # for i in range(5):
        #     rgba = ColorRGBA()
        #     rgba.r = color[0]
        #     rgba.g = color[1]
        #     rgba.b = color[2]
        #     rgba.a = 1.0
        #     new_msg.rgb_vals.append(rgba)
        # self.pub_leds.publish(new_msg)

        # self.setPattern(msg)

        return ChangePatternResponse()
    
    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()
            if self.shutdown:
                rospy.signal_shutdown("Led out")

if __name__ == "__main__":
    node = LEDNode(node_name="led_node")
    node.run()
    rospy.spin()
