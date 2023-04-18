import os
import yaml
import numpy as np

def read_yaml(path):
        with open(path, 'r') as f:
            content = f.read()
            data = yaml.load(content, yaml.SafeLoader)
        return data

def load_intrinsic(veh_name):
    path = f"/data/config/calibrations/camera_intrinsic/{veh_name}.yaml"

    # validate path
    if not os.path.isfile(path):
        print(path)
        print(f"Intrinsic calibration for {veh_name} does not exist.")
        exit(3)
    # read calibration file
    data = read_yaml(path)
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

def load_homography(veh_name):
    path = f"/data/config/calibrations/camera_extrinsic/{veh_name}.yaml"

    # validate path
    if not os.path.isfile(path):
        print(path)
        print(f"Extrinsic calibration for {veh_name} does not exist.")
        exit(2)
    # read calibration file
    data = read_yaml(path)
    return np.array(data["homography"]).reshape(3, 3)
