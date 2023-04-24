#!/usr/bin/env python3
from duckietown.dtros import DTROS, NodeType
from dwa_planner import DWA_Planner

class DWA_Node(DTROS):

    def __init__(self, node_name):
        super(DWA_Node, self).__init__(
            node_name=node_name, node_type=NodeType.GENERIC)
        self.planner = DWA_Planner()

    def run(self):
        self.planner.run()

if __name__ == "__main__":
    node = DWA_Node("lane_following_node")
    node.run()
