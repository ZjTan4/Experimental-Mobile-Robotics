#!/usr/bin/env python3

import rospy
import cv2
import os
import numpy as np
import rospkg
import torch
import yaml

from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Int16
from sensor_msgs.msg import CompressedImage
from turbojpeg import TurboJPEG
from dt_apriltags import Detector
from cv_bridge import CvBridge
from cnn import MnistCNN
from augmenter import Augmenter

DEBUG = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

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

        # Publishers
        self.pub_image_test = rospy.Publisher(f"/{self.veh}/{self.node_name}/output/image/mask/compressed",
                                              CompressedImage,
                                              queue_size=10)
        self.pub_bounding_box = rospy.Publisher(f"/{self.veh}/{self.node_name}/bounding_box/image/mask/compressed",
                                             CompressedImage,
                                             queue_size=10)
        self.pub_detection = rospy.Publisher(f"/{self.veh}/{self.node_name}/detection",
                                             Int16,
                                             queue_size=10)

        self.rospack = rospkg.RosPack()
        self.path = self.rospack.get_path("detection")

        # Util varaibles
        self.jpeg = TurboJPEG()
        self.img = None
        self.last_stop_time = rospy.Time.now()

        # Calibration and apriltag things
        self.intrinsic = self.load_intrinsic()
        self.extrinsic = self.load_homography()
        self.augmenter = Augmenter(self.intrinsic, self.extrinsic)
        self.apriltag_detector = Detector(nthreads=4)
        K = np.array(self.intrinsic["K"]).reshape((3, 3))
        self.cam_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        # Detection Variable
        self.bridge = CvBridge()
        self.model = self.load_detection_model()
        self.model.eval()
        self.detected_numbers = set()
        self.last_detected_number = None
        self.last_detection_time = rospy.get_time()

        # Finish
        self.loginfo("Initialized")

    def cb_image(self, msg):
        img = self.jpeg.decode(msg.data)
        self.img = img
        return

    def load_detection_model(self):
        model = MnistCNN(in_channels=1, out_channels=16,
                         kernel_size=3, num_classes=10)
        model_path = os.path.join(self.path, 'model/best_model.pt')
        model.load_state_dict(torch.load(model_path))

        return model

    def crop_image_with_changed_coordinates(self, image, coordinates, change=50):
        c1, c2, c3, c4 = coordinates
        offset = c1[1] - c4[1]

        c1 += [-change, change - offset]
        c2 += [change, change - offset]
        c3 += [change, -change - offset]
        c4 += [-change, -change - offset]

        cropped_image = image[int(c3[1])-10:int(c1[1])-10,
                              int(c1[0]):int(c3[0]), :]
        if DEBUG:
            for idx in range(len(coordinates)):
                cv2.line(image, tuple(coordinates[idx - 1, :].astype(int)), tuple(
                    coordinates[idx, :].astype(int)), (0, 255, 0))
            image_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.pub_bounding_box.publish(image_msg)

        return cropped_image

    def handle_detection(self, image, detection):
        '''
            This function should handle the detections:
                1. draw the bounding box
                2. (Probably) ignore the distant tags and only keep the closest?
        '''
        # draw the bounding box
        cropped_image = self.crop_image_with_changed_coordinates(
            image, detection.corners, 0)
        cropped_image = cv2.resize(cropped_image, dsize=(
            28, 28), interpolation=cv2.INTER_CUBIC)
        cropped_image = cv2.fastNlMeansDenoising(cropped_image)

        gray_cropped_image = cv2.cvtColor(
            cropped_image, cv2.COLOR_BGR2GRAY)
        _, bw_image = cv2.threshold(
            gray_cropped_image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bw_image = cv2.subtract(255, bw_image)

        if DEBUG:
            image_msg = self.bridge.cv2_to_compressed_imgmsg(
                bw_image)
            self.pub_image_test.publish(image_msg)

        return bw_image

    def detect_apriltag(self):
        # undistorts raw images
        image = self.augmenter.process_image(self.img)

        # gray-scale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect
        detections = self.apriltag_detector.detect(
            gray_image, estimate_tag_pose=True, camera_params=self.cam_params, tag_size=0.06)

        if len(detections) > 0:
            cropped_image = self.handle_detection(image, detections[0])

            cropped_image = torch.Tensor(
                cropped_image).unsqueeze(0).unsqueeze(0)
            prediction = torch.argmax(self.model(cropped_image)).item()

            # DUMMY
            # TODO: Use Real model
            tag_id = detections[0].tag_id
            tag_id_number_pair = {200:0, 62:1, 93:2, 58:3, 162:4, 201:5, 143:6, 153:7, 94:8, 169:9}
            prediction = tag_id_number_pair[tag_id]

            self.detected_numbers.add(prediction)
            return prediction

    def run(self):
        rate = rospy.Rate(self.process_frequency)
        while not rospy.is_shutdown():
            # start percetion
            if self.img is None:
                # print("None")
                continue
            if self.last_detected_number is None or (rospy.Time.now() - self.last_detection_time > rospy.Duration.from_sec(3.0)):
                prediction = self.detect_apriltag()
                
                if prediction is not None and self.last_detected_number != prediction:
                    self.pub_detection.publish(Int16(data=prediction))
                    self.last_detected_number = prediction
                    self.last_detection_time = rospy.Time.now()

            rate.sleep()

    def read_yaml(self, path):
        with open(path, 'r') as f:
            content = f.read()
            data = yaml.load(content, yaml.SafeLoader)
        return data

    def load_intrinsic(self):
        path = f"data/config/calibrations/camera_intrinsic/{self.veh}.yaml"
        path = os.path.join(self.path, path)

        # validate path
        if not os.path.isfile(path):
            print(f"Intrinsic calibration for {self.veh} does not exist.")
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
        path = f"data/config/calibrations/camera_extrinsic/{self.veh}.yaml"
        path = os.path.join(self.path, path)

        # validate path
        if not os.path.isfile(path):
            print(f"Intrinsic calibration for {self.veh} does not exist.")
            exit(2)
        # read calibration file
        data = self.read_yaml(path)
        return np.array(data["homography"]).reshape(3, 3)


if __name__ == "__main__":
    node = DetectionNode("detection_node")
    node.run()
