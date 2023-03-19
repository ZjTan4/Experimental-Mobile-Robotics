#!/usr/bin/env python3
import rospy
import numpy as np

class LaneController:
    def __init__(self) -> None:
        self.last_t = rospy.Time.now().to_sec()
        self.integral_err = 0.0
        self.prev_err = 0.0
        # PID-related parameters
        self.k_p = 0.01
        self.k_i = 0.01
        self.k_d = 0.00
        # other paramters
        self.v = 0.13
        self.bound = 8

    def reset_if_needed(self, current_err):
        if np.sign(current_err) != np.sign(self.prev_err):
            self.integral_err = 0.0
        self.integral_err = max(-self.bound, min(self.integral_err, self.bound))

    def getNextCommand(self, target_point, current_point):
        # compute dt
        current_t = rospy.Time.now().to_sec()
        dt = current_t - self.last_t
        self.last_t = current_t
        # compute error
        try:
            err = current_point[0] - target_point[0]
        except:
            err = 0
        self.integral_err += err * dt
        self.reset_if_needed(err)
        # compute command
        omega = self.k_p * err + self.k_i * self.integral_err + self.k_d * (err / dt)
        omega = omega if abs(omega) < self.bound else np.sign(omega) * self.bound
        return self.v, omega