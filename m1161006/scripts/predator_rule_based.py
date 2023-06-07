#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy, math
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
import numpy as np
from bot import Bot

class KirBot(Bot):
    def __init__(self, odom_positions, handle_odom_positions, DEBUG=False):
        super().__init__("kir_bot", odom_positions, handle_odom_positions)

        self.initialized = False

        self.DEBUG = DEBUG

        if self.DEBUG:
            rospy.init_node(self.name)

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber(f'/{self.name}/camera/rgb/image_raw',
                Image, self.image_callback, queue_size=10)

        # subscribe to the robot's scan topic
        rospy.Subscriber(f"/{self.name}/scan", LaserScan, self.process_scan, queue_size=10)

        # set up publisher and Twist to publish to /cmd_vel
        self.cmd_vel_pub = rospy.Publisher(f'/{self.name}/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        self.prey_in_frame = False

        self.initialized = True


    def set_v(self, velocity, angular_velocity):
        """ The current velocity and angular velocity of the robot are set here
        """
        v1 = Vector3(velocity, 0.0, 0.0)
        v2 = Vector3(0.0, 0.0, angular_velocity)
        t = Twist(v1, v2)
        self.cmd_vel_pub.publish(t)


    def process_scan(self, data):

        if self.prey_in_frame:
            # use minimum distance to nearest object to avoid obstacles,
            # walls, or other robots
            min_dist = min(data.ranges)
            min_angl = data.ranges.index(min_dist)

            # initialize angular velocity to 0
            v = 0

            # PID control to avoid obstacles

            if self.DEBUG:
                print(min_angl, v)

            self.twist.linear.x = +0.6
            self.twist.angular.z = v
            self.cmd_vel_pub.publish(self.twist)
        else:
            # doesn't see predator, so no linear velocity
            self.twist.linear.x = 0.0
            #self.cmd_vel_pub.publish(self.twist)
            self.set_v(0,1.2)


    def image_callback(self, data):

        if (not self.initialized):
            return

        self.prey_angle = 0
        self.prey_in_frame = False

        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, w, d = image.shape

        # colors of prey we want to search for
        lower_blue = np.array([235/2, 230, 230])
        upper_blue = np.array([245/2, 255, 255])

        lower_green = np.array([115/2, 230, 230])
        upper_green = np.array([125/2, 255, 255])

        lower_yellow = np.array([55/2, 230, 230])
        upper_yellow = np.array([65/2, 255, 255])
        
        lower_bounds = [lower_blue, lower_green, lower_yellow]
        upper_bounds = [upper_blue, upper_green, upper_yellow]

        best_err_to_approx_angle = None

        self.pixel_percent = 0

        for i in range(len(lower_bounds)):
            lower = lower_bounds[i]
            upper = upper_bounds[i]

            mask = cv2.inRange(hsv, lower, upper)

            M = cv2.moments(mask)
            # if there are any colored pixels found
            if M['m00'] > 0:
                    pixel_percent = (mask>0).mean()
                    self.pixel_percent = min(1, max(math.pow(pixel_percent, .25), self.pixel_percent))
                    # center of the colored pixels in the image
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                    err = (w / 2 - cx)
                    err_to_approx_angle =  err / w * math.pi * 0.5
                    if best_err_to_approx_angle is None or abs(err_to_approx_angle) < abs(best_err_to_approx_angle):
                        best_err_to_approx_angle = err_to_approx_angle
            else:
                continue

        if best_err_to_approx_angle is not None:
            self.prey_angle = best_err_to_approx_angle
            self.prey_in_frame = True


    def run(self):
        if self.DEBUG:
            rospy.spin()


# this is for running the node manually for debugging and testing
if __name__ == '__main__':
    node = KirBot(DEBUG=True)
    node.run()
