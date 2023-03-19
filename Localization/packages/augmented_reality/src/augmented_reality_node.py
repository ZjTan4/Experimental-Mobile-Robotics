#!/usr/bin/env python3

import rospy
import rospkg
import cv2
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from augmenter import Augmenter
import yaml
import sys
import numpy as np
from os.path import isfile

class AugmentedRealityNode(DTROS):
    def __init__(self, node_name) -> None:
        super(AugmentedRealityNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        # load Arguments
        self.map_file_base_name = sys.argv[1]
        self.veh_name = sys.argv[2]
        # Load Intrinsic/Extrinsic parameters
        self.intrinsic = self.load_intrinsic()
        self.extrinsic = self.load_homography()
        # Load Map file
        self.map_data = self.load_map()
        # Subscriber
        self.sub_image_compressed = rospy.Subscriber(f"/{self.veh_name}/camera_node/image/compressed", CompressedImage, self.cb_compressed_image)
        # Publisher
        self.pub_augmented_image = rospy.Publisher(f'/{self.veh_name}/augmented_reality_node/{self.map_file_base_name}/image/compressed', CompressedImage, queue_size=1)
        # Augmenter
        self.augmenter = Augmenter(self.intrinsic, self.extrinsic, self.map_data)
        self.bridge = CvBridge()
        # Init done
        rospy.loginfo("Initialized")

    def cb_compressed_image(self, compressed_image):
        # load image from compressed msg
        image = self.bridge.compressed_imgmsg_to_cv2(compressed_image)
        # undistorts raw images
        image = self.augmenter.process_image(image)
        # draw the augmented segments
        image = self.augmenter.render_segments(image)
        # publish 
        image_msg = self.bridge.cv2_to_compressed_imgmsg(image)
        self.pub_augmented_image.publish(image_msg)
    
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
        intrinsics["D"] = np.array(data["distortion_coefficients"]["data"]).reshape(1, 5)
        intrinsics["R"] = np.array(data["rectification_matrix"]["data"]).reshape(3, 3)
        intrinsics["P"] = np.array(data["projection_matrix"]["data"]).reshape((3, 4))
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

    def load_map(self):
        path = rospkg.RosPack().get_path("augmented_reality") + f'/src/maps/{self.map_file_base_name}.yaml'
        return self.read_yaml(path)

if __name__ == "__main__":
    node = AugmentedRealityNode(node_name="augmented_reality_node")
    rospy.spin()