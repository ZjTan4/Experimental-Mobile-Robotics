#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np

from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Int16
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from turbojpeg import TurboJPEG

ROAD_MASK = [(13, 70, 170), (32, 255, 255)]
STOP_MASK = [(0, 50, 50), (10, 255, 255)]
DEBUG = True

class LaneFollowingNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowingNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = str(os.environ['VEHICLE_NAME'])
        self.process_frequency = 3

        # Subscribers
        self.sub_image = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed",
                                          CompressedImage,
                                          self.cb_image,
                                          queue_size=1,
                                          buff_size="20MB")
        self.sub_prediction = rospy.Subscriber(f"/{self.veh}/detection_node/detection",
                                          Int16,
                                          self.cb_detection,
                                          queue_size=1,
                                          buff_size="20MB")

        # Publishers
        self.pub_cmd = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=10)
        self.pub_image_test = rospy.Publisher(f"/{self.veh}/{self.node_name}/output/image/mask/compressed",
                                              CompressedImage,
                                              queue_size=10)
        
        # Util varaibles
        self.jpeg = TurboJPEG()
        self.img = None
        self.last_stop_time = rospy.Time.now()
        self.prediction = set()

        # PID Variables
        self.P = 0.05 # 0.023
        self.D = -0.0025 # -0.0035
        self.lane_following_error = None

        self.offset = 185

        self.omega_bound = 6
        self.last_follow_error = 0
        self.last_PID_time_follow = rospy.get_time()

        # Velocity
        self.velocity = 0.33 # 0.13
        self.omega = 0

        # Shutdown hook
        rospy.on_shutdown(self.hook)

        # Finish
        self.loginfo("Initialized")

    def cb_image(self, msg):
        img = self.jpeg.decode(msg.data)
        self.img = img
        return

    def cb_detection(self, msg):
        prediction = msg.data
        if not prediction in self.prediction:
            self.prediction.add(prediction)
            print(self.prediction)
        return

    def PID(self, error):
        if error is None:
            self.omega = 0
        else:
            # P Term
            P = -error * self.P
            # D Term
            d_error = (error - self.last_follow_error) / \
                (rospy.get_time() - self.last_PID_time_follow)
            self.last_follow_error = error
            self.last_PID_time_follow = rospy.get_time()

            D = d_error * self.D
            self.omega = min(
                max((P + D), -self.omega_bound), self.omega_bound)

    def lane_detection(self):
        crop = self.img[350:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.lane_following_error = cx - \
                    int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(
                        crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
                    rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
                    self.pub_image_test.publish(rect_img_msg)
            except:
                self.lane_following_error = None

        else:
            self.lane_following_error = None

    def lane_following(self):
        self.PID(self.lane_following_error)

    def hook(self):
        print("SHUTTING DOWN")
        self.stop()

    def stop(self, rate=None):
        twist = Twist2DStamped(v=0, omega=0)
        self.pub_cmd.publish(twist)
        if rate:
            rate.sleep()

    def move(self, rate=None):
        twist = Twist2DStamped(v=self.velocity, omega=self.omega)
        self.pub_cmd.publish(twist)
        if rate:
            rate.sleep()

    def detect_intersection(self):
        # stop line
        crop = self.img[400:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
        detection = cv2.bitwise_and(crop, crop, mask=mask)
        if np.sum(detection) > 100000:
            return True
        return False

    def run(self):
        rate = rospy.Rate(self.process_frequency)
        not_turn_yet = True
        not_straight_yet = True
        while not rospy.is_shutdown():
            # start percetion
            if self.img is None:
                # print("None")
                continue

            if self.detect_intersection() and rospy.Time.now() - self.last_stop_time > rospy.Duration.from_sec(4.0):
                # Reset PID
                self.last_follow_error = 0
                self.lane_following_error = 0
                self.last_stop_time = rospy.Time.now()
                # Check prediction
                if 6 <= len(self.prediction) < 8 and not_turn_yet:
                    # Left Turn
                    turn_duration = rospy.Duration.from_sec(1.5)
                    start_time = rospy.Time.now()
                    not_turn_yet = False
                    while rospy.Time.now() - start_time < turn_duration:
                        self.omega = 2.5
                        self.move()
                        rate.sleep()
                elif 8 <= len(self.prediction) < 10 and not_straight_yet:
                    # Go Straight
                    turn_duration = rospy.Duration.from_sec(3.0)
                    start_time = rospy.Time.now()
                    not_straight_yet = False
                    while rospy.Time.now() - start_time < turn_duration:
                        self.omega = 0
                        self.move()
                        rate.sleep()
                else:
                    # Do Nothing
                    pass
            # Lane following
            self.lane_detection()
            self.lane_following()
            self.move()
            # Next
            rate.sleep()

if __name__ == "__main__":
    node = LaneFollowingNode("lane_following_node")
    node.run()
