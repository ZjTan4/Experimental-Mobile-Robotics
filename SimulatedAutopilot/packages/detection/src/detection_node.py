#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np

from std_msgs.msg import Int32
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from geometry_msgs.msg import Transform, Vector3, Quaternion
from sensor_msgs.msg import CompressedImage
from turbojpeg import TurboJPEG
from dt_apriltags import Detector
from cv_bridge import CvBridge
from augmenter import Augmenter
from utils import *
from tf import transformations as tr

DEBUG = True

class DetectionNode(DTROS):

    def __init__(self, node_name):
        super(DetectionNode, self).__init__(
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
        
        self.sub_shutdown_command = rospy.Subscriber(f"/{self.veh}/lane_following_node/shutdown",
                                          Int32,
                                          self.cb_shutdown,
                                          queue_size=1,
                                          buff_size="20MB")
        

        # Publishers
        self.pub_image_test = rospy.Publisher(f"/{self.veh}/{self.node_name}/output/image/mask/compressed",
                                              CompressedImage,
                                              queue_size=10)
        self.pub_detection = rospy.Publisher(f"/{self.veh}/{self.node_name}/detection",
                                             AprilTagDetectionArray,
                                             queue_size=10)

        # Util varaibles
        self.jpeg = TurboJPEG()
        self.img = None
        self.last_stop_time = rospy.Time.now()

        # Calibration and apriltag things
        self.intrinsic = load_intrinsic(self.veh)
        self.extrinsic = load_homography(self.veh)
        self.augmenter = Augmenter(self.intrinsic, self.extrinsic)
        self.apriltag_detector = Detector(nthreads=4)
        K = np.array(self.intrinsic["K"]).reshape((3, 3))
        self.cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        # Detection Variable
        self.bridge = CvBridge()
        
        # Finish
        self.loginfo("Initialized")

    def cb_shutdown(self, msg):
        if msg.data == 1:
            rospy.signal_shutdown("shutting down detection node")

    def cb_image(self, msg):
        img = self.jpeg.decode(msg.data)
        self.img = img
        return

    def detect_apriltag(self):
        # undistorts raw images
        image = self.augmenter.process_image(self.img)
        tag_id = None

        # gray-scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect
        detections = self.apriltag_detector.detect(
            gray_image, estimate_tag_pose=True, camera_params=self.cam_params, tag_size=0.06)

        if len(detections) > 0:            
            tags_msg = AprilTagDetectionArray()
            for detection in detections:
                translation = detection.pose_t.T[0]
                rotation = matrix_to_quaternion(detection.pose_R)
                detection_msg = AprilTagDetection(
                    transform=Transform(
                        translation=Vector3(
                            x=translation[0], y=translation[1], z=translation[2]),
                        rotation=Quaternion(
                            x=rotation[0], y=rotation[1], z=rotation[2], w=rotation[3])
                    ),
                    tag_id=detection.tag_id,
                    tag_family=str(detection.tag_family),
                    hamming=detection.hamming,
                    decision_margin=detection.decision_margin,
                    homography=detection.homography.flatten().astype(np.float32).tolist(),
                    center=detection.center.tolist(),
                    corners=detection.corners.flatten().tolist(),
                    pose_error=detection.pose_err,
                )
                tags_msg.detections.append(detection_msg)
            # publish it
            self.pub_detection.publish(tags_msg)

    def run(self):
        rate = rospy.Rate(self.process_frequency)
        while not rospy.is_shutdown():
            # start percetion
            if self.img is None:
                continue
            self.detect_apriltag()
            rate.sleep()

def matrix_to_quaternion(r):
    T = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
                 (0, 0, 0, 1)), dtype=np.float64)
    T[0:3, 0:3] = r
    return tr.quaternion_from_matrix(T)

if __name__ == "__main__":
    node = DetectionNode("detection_node")
    node.run()
