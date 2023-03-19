#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from dt_apriltags import Detector
from duckietown_msgs.srv import SetCustomLEDPattern
# , AprilTagDetectionArray
from duckietown_msgs.msg import LEDPattern, AprilTagDetection
from geometry_msgs.msg import Transform, Vector3, Quaternion
from sensor_msgs.msg import CompressedImage
from augmenter import Augmenter
import yaml
import numpy as np
from os.path import isfile

from tf2_ros import TransformBroadcaster
from tf import transformations as tr


class AugmentedRealityApriltagNode(DTROS):
    def __init__(self, node_name) -> None:
        super(AugmentedRealityApriltagNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")
        # Load Intrinsic/Extrinsic parameters
        self.intrinsic = self.load_intrinsic()
        self.extrinsic = self.load_homography()
        K = np.array(self.intrinsic["K"]).reshape((3, 3))
        self.cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        # Subscriber
        self.sub_image_compressed = rospy.Subscriber(
            f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_compressed_image)
        self.image_cache = None
        # Publisher
        self.pub_augmented_image = rospy.Publisher(
            f'/{self.veh_name}/augmented_reality_apriltag_node/image/compressed', CompressedImage, queue_size=1)
        self.pub_detection = rospy.Publisher(
            f'/{self.veh_name}/augmented_reality_apriltag_node/detection', AprilTagDetection, queue_size=1)
        #self.pub_detections = rospy.Publisher(f'/{self.veh_name}/augmented_reality_apriltag_node/detections', AprilTagDetectionArray, queue_size=1)
        # Service
        service_name = f"/{self.veh_name}/led_emitter_node/set_custom_pattern"
        rospy.wait_for_service(service_name)
        self.sev_led_color = rospy.ServiceProxy(
            service_name, SetCustomLEDPattern)
        # Tf broadcasters
        self.tf_detection = TransformBroadcaster()
        # Utils
        self.augmenter = Augmenter(self.intrinsic, self.extrinsic)
        self.apriltag_detector = Detector(nthreads=4)
        self.bridge = CvBridge()
        # Init done
        rospy.loginfo("Initialized")

    def cb_compressed_image(self, compressed_image):
        # cache the image since we want to reduce the frequency
        self.image_cache = compressed_image

    def handle_detection(self, image, detections):
        '''
            This function should handle the detections:
                1. draw the bounding box
                2. chnage the led light accordingly
                3. (Probably) ignore the distant tags and only keep the closest?
            LED color: 
                a. Red : Stop Sign
                b. Blue : T-Intersection
                c. Green : UofA Tag
                d. White : No Detections
        '''
        colors = {
            # STOP SIGN
            162: {"color": "red", "bgr": (0, 0, 255)},
            169: {"color": "red", "bgr": (0, 0, 255)},
            # T-INTERSECTION
            58: {"color": "blue", "bgr": (255, 0, 0)},
            62: {"color": "blue", "bgr": (255, 0, 0)},
            133: {"color": "blue", "bgr": (255, 0, 0)},
            153: {"color": "blue", "bgr": (255, 0, 0)},
            # UofA TAG
            93: {"color": "green", "bgr": (0, 255, 0)},
            94: {"color": "green", "bgr": (0, 255, 0)},
            200: {"color": "green", "bgr": (0, 255, 0)},
            201: {"color": "green", "bgr": (0, 255, 0)},
        }
        # white if nothing detected
        if len(detections) == 0:
            self.change_led_color("white")
        # handle each tag (or one tag?)
        # for detection in detections:
        if len(detections) > 0:
            detection = detections[0]
            # change LED color
            tag_id = detection.tag_id
            color = colors.get(
                tag_id, {"color": "white", "bgr": (255, 255, 255)})
            self.change_led_color(color["color"])
            # draw the bounding box
            for idx in range(len(detection.corners)):
                cv2.line(image, tuple(detection.corners[idx - 1, :].astype(int)), tuple(
                    detection.corners[idx, :].astype(int)), color["bgr"])
            cv2.putText(
                image,
                str(detection.tag_id),
                org=(detection.corners[0, 0].astype(int) + 10, detection.corners[0, 1].astype(int) + 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255),
            )
            # terminate here if the tag is not in duckietown
            if tag_id not in colors:
                return image
            # publish it
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
            self.pub_detection.publish(detection_msg)
        return image

    def change_led_color(self, color: str):
        msg = LEDPattern()
        msg.color_list = [color] * 5
        msg.color_mask = [1, 1, 1, 1, 1]
        msg.frequency = 0.0
        msg.frequency_mask = [0, 0, 0, 0, 0]
        self.sev_led_color(msg)

    def read_yaml(self, path):
        with open(path, 'r') as f:
            content = f.read()
            data = yaml.load(content, yaml.SafeLoader)
        return data

    def load_intrinsic(self):
        path = f"/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml"
        # validate path
        if not isfile(path):
            print(f"Intrinsic calibration for {self.veh_name} does not exist.")
            exit(3)
        # read calibration file
        data = self.read_yaml(path)
        # load data
        intrinsics = {}
        intrinsics["W"] = data["image_width"]
        intrinsics["H"] = data["image_height"]
        intrinsics["K"] = np.array(data["camera_matrix"]["data"]).reshape(3, 3)
        intrinsics["D"] = np.array(
            data["distortion_coefficients"]["data"]).reshape(1, 5)
        intrinsics["R"] = np.array(
            data["rectification_matrix"]["data"]).reshape(3, 3)
        intrinsics["P"] = np.array(
            data["projection_matrix"]["data"]).reshape((3, 4))
        intrinsics["distortion_model"] = data["distortion_model"]
        return intrinsics

    def load_homography(self):
        path = f"/data/config/calibrations/camera_extrinsic/{self.veh_name}.yaml"
        # validate path
        if not isfile(path):
            print(f"Intrinsic calibration for {self.veh_name} does not exist.")
            exit(2)
        # read calibration file
        data = self.read_yaml(path)
        return np.array(data["homography"]).reshape(3, 3)

    def run(self):
        rate = rospy.Rate(8)
        while not rospy.is_shutdown():
            # if nothing comes in, do nothing
            if self.image_cache is None:
                continue
            # load image from compressed msg
            image = self.bridge.compressed_imgmsg_to_cv2(self.image_cache)
            # undistorts raw images
            image = self.augmenter.process_image(image)
            # gray-scale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect
            detections = self.apriltag_detector.detect(
                gray_image, estimate_tag_pose=True, camera_params=self.cam_params, tag_size=0.06)
            # augment and change led color
            image = self.handle_detection(image, detections)
            # publish
            image_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.pub_augmented_image.publish(image_msg)
            # next
            rate.sleep()


def matrix_to_quaternion(r):
    T = np.array(((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0),
                 (0, 0, 0, 1)), dtype=np.float64)
    T[0:3, 0:3] = r
    return tr.quaternion_from_matrix(T)


if __name__ == "__main__":
    node = AugmentedRealityApriltagNode(
        node_name="augmented_reality_apriltag_node")
    node.run()
    rospy.spin()
