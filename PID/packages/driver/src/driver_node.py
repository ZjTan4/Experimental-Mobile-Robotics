#!/usr/bin/env python3

import rospy
import copy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import os
import numpy as np
from duckietown_msgs.msg import Twist2DStamped, VehicleCorners, LEDPattern
from duckietown_msgs.srv import SetCustomLEDPattern

ROAD_MASK = [(13, 70, 170), (32, 255, 255)]
STOP_MASK = [(0, 50, 50), (10, 255, 255)]
DEBUG = False
ENGLISH = False


class DriverNode(DTROS):

    def __init__(self, node_name):
        super(DriverNode, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = str(os.environ['VEHICLE_NAME'])
        self.process_frequency = 5

        # Publishers
        self.pub_image = rospy.Publisher(f"/{self.veh}/output/image/mask/compressed",
                                         CompressedImage,
                                         queue_size=10)
        self.pub_cmd = rospy.Publisher(f"/{self.veh}/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=10)
        # Subscribers
        self.sub_image = rospy.Subscriber(f"/{self.veh}/camera_node/image/compressed",
                                          CompressedImage,
                                          self.cb_image,
                                          queue_size=1,
                                          buff_size="20MB")
        self.sub_distance = rospy.Subscriber(f"/{self.veh}/duckiebot_distance_node/distance",
                                             Float32,
                                             self.cb_distance,
                                             queue_size=1)
        self.sub_centers = rospy.Subscriber(f"/{self.veh}/duckiebot_detection_node/centers",
                                            VehicleCorners,
                                            self.cb_centers,
                                            queue_size=1)
        # Service
        # service_name = f"/{self.veh}/led_emitter_node/set_custom_pattern"
        # rospy.wait_for_service(service_name)
        # self.sev_led_color = rospy.ServiceProxy(
        #     service_name, SetCustomLEDPattern)

        # Util varaibles
        self.jpeg = TurboJPEG()
        self.centers = VehicleCorners()
        self.img = np.empty((480, 640, 3), dtype=np.uint8)
        self.distance = float('inf')
        self.counter = 0
        self.last_stop_time = rospy.Time.now()

        # PID Variables
        self.P = 0.02 # 0.049
        self.D = -0.0035 # -0.004
        self.lane_following_error = None
        self.taling_error = None
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 185

        self.omega_bound = 4
        self.last_trail_error = 0
        self.last_follow_error = 0
        self.last_PID_time_trail = rospy.get_time()
        self.last_PID_time_follow = rospy.get_time()

        # Distance
        self.safe_distance = 0.25
        # Velocity
        self.velocity = 0.13

        # Messages
        self.omega = 0

        # Shutdown hook
        rospy.on_shutdown(self.hook)

        # Finish
        self.loginfo("Initialized")

    def change_led_color(self, color: str):
        msg = LEDPattern()
        msg.color_list = [color] * 5
        msg.color_mask = [1, 1, 1, 1, 1]
        msg.frequency = 0.0
        msg.frequency_mask = [0, 0, 0, 0, 0]
        print(color)
        # self.sev_led_color(msg)

    def cb_distance(self, msg: Float32):
        self.distance = msg.data
        return

    def cb_centers(self, msg):
        if self.centers.detection.data and not msg.detection.data:
            if self.counter <= 10:
                self.counter += 1
                return 
            else:
                self.distance = float('inf')
        self.counter = 0
        self.centers = msg
        return

    def cb_image(self, msg):
        img = self.jpeg.decode(msg.data)
        self.img = img
        return

    def PID(self, error, trail=False):
        if error is None:
            self.omega = 0
        else:
            # P Term
            P = -error * self.P
            # D Term
            if trail:
                d_error = (error - self.last_trail_error) / \
                    (rospy.get_time() - self.last_PID_time_trail)
                self.last_trail_error = error
                self.last_PID_time_trail = rospy.get_time()
            else:
                d_error = (error - self.last_follow_error) / \
                    (rospy.get_time() - self.last_PID_time_follow)
                self.last_follow_error = error
                self.last_PID_time_follow = rospy.get_time()
                
            D = d_error * self.D
            self.omega = min(
                max((P + D), -self.omega_bound), self.omega_bound)
            # if DEBUG:
            #     print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            #         error, P, D, self.twist.omega, self.twist.v))

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
            except:
                self.lane_following_error = None
                self.last_follow_error = 0
        else:
            self.lane_following_error = None
            self.last_follow_error = 0

    def tail_detection(self):
        if self.centers.detection.data:
            points = np.zeros((self.centers.H * self.centers.W, 2))
            for i in range(len(points)):
                points[i] = np.array(
                    [self.centers.corners[i].x, self.centers.corners[i].y])
            mean_center = points.mean(axis=0)
            bound = 100
            self.taling_error = min(max((mean_center[0] - 320), -bound), bound)
            # self.taling_error = (mean_center[0] - 320) - abs(self.last_trail_error)
            # self.taling_error = mean_center[0] - abs(self.last_trail_error)
        else:
            self.taling_error = None
            # self.last_trail_error = 0
            
    def lane_following(self):
        print(self.lane_following_error, 'lane_following_error')
        print(self.last_follow_error, 'last_follow_error')
        self.PID(self.lane_following_error, trail=False)

    def tailing(self):
        print(self.taling_error, 'taling_error')
        print(self.last_trail_error, 'last_trail_error')
        self.PID(self.taling_error, trail=True)

    def t_intersection(self, rate):
        self.stop(rate)
        self.change_led_color("red")
        rospy.sleep(0.5)
        self.move(rate)
        self.change_led_color("switchedoff")
        self.last_stop_time = rospy.Time.now()

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

    def perception(self):
        # stop line
        crop = self.img[400:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
        detection = cv2.bitwise_and(crop, crop, mask=mask)
        if np.sum(detection) > 100000:
            return 1
        # tailing
        if self.centers.detection.data:
            return 2
        # lane following
        return 3

    def run(self):
        rate = rospy.Rate(self.process_frequency)
        latest_perception = None
        while not rospy.is_shutdown():
            if self.centers.detection.data and self.distance < self.safe_distance:
                # Too close, solid stop
                self.stop(rate)
                continue

            # start percetion
            perception = self.perception()
            if latest_perception != perception:
                print("CHANGING THE PERCEPTION: ", perception)
                self.last_follow_error = 0
                self.last_trail_error = 0
                
                self.lane_following_error = 0
                self.taling_error = 0

                latest_perception = perception

            if perception == 1 and rospy.Time.now() - self.last_stop_time > rospy.Duration.from_sec(4.0):
                # stop line
                self.t_intersection(rate)
            elif perception == 2 and self.distance >= self.safe_distance:
                # tailing
                self.tail_detection()
                self.tailing()
            elif perception == 3 and self.distance >= self.safe_distance:
                # lane following
                self.lane_detection()
                self.lane_following()

            self.move()
            rate.sleep()


if __name__ == "__main__":
    node = DriverNode("driver_node")
    node.run()
