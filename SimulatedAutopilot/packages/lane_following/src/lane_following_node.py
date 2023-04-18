#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
import sys

from duckietown_msgs.msg import Twist2DStamped, AprilTagDetectionArray
from std_msgs.msg import Int32
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
        self.sub_detection = rospy.Subscriber(f"/{self.veh}/detection_node/detection",
                                               AprilTagDetectionArray,
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
        self.pub_shutdown_command = rospy.Publisher(f"/{self.veh}/{self.node_name}/shutdown",
                                                    Int32,
                                                    queue_size=10)

        # Util varaibles
        self.jpeg = TurboJPEG()
        self.img = None

        # PID Variables
        self.P = 0.053  # 0.023
        self.D = -0.0025  # -0.0035
        self.lane_following_error = None

        self.offset = 185

        self.omega_bound = 7
        self.last_PID_error = 0
        self.last_PID_time_follow = rospy.get_time()

        # Velocity
        self.velocity = 0.33  # 0.13
        self.omega = 0

        self.tags = None
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 10
        params.minDistBetweenBlobs = 2
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(params)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

        # Finish
        self.loginfo("Initialized")

    def cb_image(self, msg):
        img = self.jpeg.decode(msg.data)
        self.img = img
        return

    def cb_detection(self, msg):
        self.tags = msg
        return

    def get_tag(self, id=None):
        if self.tags is None:
            return None
        if id is None:
            return self.tags.detections[0]
        else:
            for detection in self.tags.detections:
                if detection.tag_id == id:
                    return detection
        return None

    def PID(self, error):
        if error is None:
            self.omega = 0
        else:
            # P Term
            P = -error * self.P
            # D Term
            d_error = (error - self.last_PID_error) / \
                (rospy.get_time() - self.last_PID_time_follow)
            self.last_PID_error = error
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
                    rect_img_msg = CompressedImage(
                        format="jpeg", data=self.jpeg.encode(crop))
                    self.pub_image_test.publish(rect_img_msg)
            except:
                self.lane_following_error = None

        else:
            self.lane_following_error = None

    def lane_following(self):
        self.PID(self.lane_following_error)

    def hook(self):
        self.pub_shutdown_command.publish(Int32(data=1))
        print("SHUTTING DOWN")
        self.stop()

    def stop(self):
        twist = Twist2DStamped(v=0, omega=0)
        self.pub_cmd.publish(twist)

    def move(self):
        twist = Twist2DStamped(v=self.velocity, omega=self.omega)
        self.pub_cmd.publish(twist)

    def detect_intersection(self):
        # stop line
        crop = self.img[400:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
        detection = cv2.bitwise_and(crop, crop, mask=mask)
        if np.sum(detection) > 150000:
            return True
        return False

    def detect_duck(self):
        crop = self.img[350:-1, 140:500, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, STOP_MASK[0], STOP_MASK[1])
        detection = cv2.bitwise_and(crop, crop, mask=mask)
        if np.sum(detection) > 10000:
            return True
        return False
    
    def detect_park_border(self):
        crop = self.img[450:-1, :, :]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        detection = cv2.bitwise_and(crop, crop, mask=mask)
        if np.sum(detection) > 100000:
            return True
        return False

    def detect_tail(self):
        (detection, _) = cv2.findCirclesGrid(
            self.img,
            patternSize=(7, 3),
            flags=cv2.CALIB_CB_SYMMETRIC_GRID,
            blobDetector=self.simple_blob_detector,
        )
        if detection > 0:
            return True
        return False
            
    def route(self):
        if self.tags is None:
            return
        self.omega = 0
        self.move()
        rospy.sleep(1.0)
        if self.get_tag(id=56) is not None: # For room CSC229
            # Straight
            self.omega = 0
            self.move()
            rospy.sleep(2.0)
            self.tags = None
        elif self.get_tag(id=48) is not None: # For room CSC229
            # Right
            self.omega = -5
            self.move()
            rospy.sleep(1.0)
            self.tags = None
        elif self.get_tag(id=50) is not None: # For room CSC229
            # Left
            self.omega = 5
            self.move()
            rospy.sleep(1.0)
            self.tags = None

    def avoid_duck(self):
        self.stop()
        rospy.sleep(1.0)
        while self.detect_duck():
            self.stop()
            rospy.sleep(1.0)

    def park(self):
        stall = rospy.get_param(f'/{self.veh}/lane_following_node/stall', 1)
        traffic_light_distance = 0.15 if stall in [1, 3] else 0.45
        rotate_direction = -1 if stall in [1, 2] else 1
        traffic_light_id = 227
        stall_ids = {
            1 : 228, 
            2 : 75, 
            3 : 207, 
            4 : 226
        }
        stall_id = stall_ids[stall]
        rate = rospy.Rate(8)
        norm = lambda v: np.sqrt(v.x**2 + v.y**2 + v.z**2)
        if DEBUG:
            print(f"Parking to Stall {stall}")        
        # go straight to the traffic light
        while (self.get_tag(id=traffic_light_id) is not None) and (norm(self.get_tag(id=traffic_light_id).transform.translation) > traffic_light_distance):
            self.PID(self.get_tag(id=traffic_light_id).center[0] - int(self.img.shape[1] / 2))
            self.velocity = 0.29
            self.move()
            rate.sleep()

        # rotate to find the tag
        while not (self.get_tag(id=stall_id) is not None and np.abs(self.get_tag(id=stall_id).center[0] - int(self.img.shape[1] / 2)) < 15):
            self.velocity = 0.0
            self.omega = 7 * rotate_direction if self.get_tag(id=stall_id) is None else 5 * -np.sign(self.get_tag(id=stall_id).center[0] - int(self.img.shape[1] / 2))
            self.move()
            rate.sleep()
            self.stop()
            rate.sleep()

        # start parking
        while not self.detect_park_border():
            # park
            self.velocity = -0.33
            if self.get_tag(id=stall_id) is not None and self.get_tag(id=stall_id).transform.translation.z <= 0.9:
                self.PID(self.get_tag(id=stall_id).center[0] - int(self.img.shape[1] / 2))
            else:
                self.PID(None)
            self.move()
            rate.sleep()
            self.stop()
            rate.sleep()

        self.stop()
        rospy.sleep(3.0)
        rospy.signal_shutdown("FINISH PARKING")
        
    def run(self):
        rate = rospy.Rate(self.process_frequency)
        tail_detected = False
        stopped = False
        tail_detected_time = rospy.Time.now()
        while not rospy.is_shutdown():
            # start percetion
            if self.img is None:
                continue
            
            # Stage 3
            if self.tags is not None and self.get_tag().tag_id == 227 and self.get_tag(id=227).transform.translation.z <= 0.6 and self.get_tag(id=38) is None:
                self.park()
            
            # Stage 2 
            elif not tail_detected and self.detect_tail():
            # detect duckiebot tail
                self.stop()
                rospy.sleep(1.0)
                tail_detected = True
                tail_detected_time = rospy.Time.now()
            elif self.tags is not None and self.get_tag().tag_id == 163 and self.detect_duck(): 
            # avoid little duck
                self.avoid_duck()
                self.tags = None
                stopped = False
            elif self.tags is not None and self.get_tag().tag_id == 163 and self.get_tag().transform.translation.z <= 0.3 and not stopped: 
            # stop 
                self.stop()
                rospy.sleep(1.0)
                stopped = True
            
            # Stage 1
            elif self.detect_intersection() and self.tags is not None:
                # Reset PID
                self.last_PID_error = 0
                self.lane_following_error = None
                # Stop
                self.stop()
                rospy.sleep(1.0)
                # Route
                self.route()
            
            # avoid duckiebot
            if tail_detected and rospy.Time.now() - tail_detected_time < rospy.Duration.from_sec(5.5):
                self.offset = -120
                stopped = False
            else:
                tail_detected = False
                self.offset = 185

            # Lane following
            self.lane_detection()
            self.lane_following()
            self.move()
            # Next
            rate.sleep()


if __name__ == "__main__":
    node = LaneFollowingNode("lane_following_node")
    node.run()
