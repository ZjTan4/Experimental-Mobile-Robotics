import rospy
import cv2
import os
import numpy as np
import sys

class State:
    def  __init__(self, x, y, yaw, velocity, yawrate) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.velocity = velocity
        self.yawrate = yawrate

class Window:
    def __init__(self, min_velocity, max_velocity, min_yawrate, max_yawrate) -> None:
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_yawrate = min_yawrate
        self.max_yawrate = max_yawrate

class DWA_Planner:
    def __init__(self) -> None:
        pass

    def prcoess(self):
        pass
