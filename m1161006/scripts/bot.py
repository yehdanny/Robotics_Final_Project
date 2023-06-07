#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy, math
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
import numpy as np

class Bot(object):

    def __init__(self, name, odom_positions, handle_odom_positions=None):
        self.name = name
        self.odom_positions = odom_positions
        self.handle_odom_positions = handle_odom_positions

        # rospy.Subscriber(f'/{self.name}/odom', Odometry, self.process_odom)

    def process_odom(self, odom):
        self.odom_positions[self.name] = odom.pose.pose
        # x = odom.pose.pose.position.x
        # y = odom.pose.pose.position.y
        # if self.name not in self.odom_positions:
        #     self.odom_positions[self.name] = odom.pose.pose
        #     self.odom_positions[self.name] = {
        #         "x": x,
        #         "y": y
        #     }
        
        # self.odom_positions[self.name]["x"] = x
        # self.odom_positions[self.name]["y"] = y

        if self.handle_odom_positions is not None:
            self.handle_odom_positions(self.odom_positions)
