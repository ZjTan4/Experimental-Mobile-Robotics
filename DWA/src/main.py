#!/usr/bin/env python3
from dwa_planner import DWA_Planner
import rospy

if __name__ == "__main__":
    rospy.init_node("dwa_planner")
    planner = DWA_Planner()
    planner.prcoess()
