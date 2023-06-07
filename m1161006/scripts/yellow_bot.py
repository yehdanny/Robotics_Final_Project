#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy, math
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
from bot import Bot

class YellowBot(Bot):

    def __init__(self, odom_positions, DEBUG=False):
        self.name = "yellow_bot"#rachel
        super().__init__(self.name, odom_positions)

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

        self.sees_predator = False

        self.initialized = True


    def set_v(self, velocity, angular_velocity):
        """ The current velocity and angular velocity of the robot are set here
        """
        v1 = Vector3(velocity, 0.0, 0.0)
        v2 = Vector3(0.0, 0.0, angular_velocity)
        t = Twist(v1, v2)
        self.cmd_vel_pub.publish(t)


    def process_scan(self, data):

        if self.sees_predator:
            # use minimum distance to nearest object to avoid obstacles,
            # walls, or other robots
            min_dist = min(data.ranges)
            min_angl = data.ranges.index(min_dist)

            # initialize angular velocity to 0
            v = 0

            # PID control to avoid obstacles
            if min_angl < 180:
                v = 0.01 * (min_angl - 90)
                if abs(v) > 0.5:
                    v = v * 0.5
            else:
                v = 0.01 * (min_angl - 270)
                if abs(v) > 0.5:
                    v = v * 0.5

            if self.DEBUG:
                print(min_angl, v)

            self.twist.linear.x = -0.2
            self.twist.angular.z = v
            self.cmd_vel_pub.publish(self.twist)
        else:
            # doesn't see predator, so no linear velocity
            self.twist.linear.x = 0.0
            self.cmd_vel_pub.publish(self.twist)


    def image_callback(self, data):

        if (not self.initialized):
            return

        # converts the incoming ROS message to cv2 format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # looking for red color, i.e. predator bot
        mask1 = cv2.inRange(hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(hsv, (175,50,20), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)

        M = cv2.moments(mask)

        if M['m00'] > 0:
            # color/predator has been detected
            self.sees_predator = True
        else:
            # since predator not detected, spin until it is detected
            self.sees_predator = False
            self.twist.angular.z = 0.2
            self.cmd_vel_pub.publish(self.twist)


    def run(self):
        if self.DEBUG:
            rospy.spin()


# this is for running the node manually for debugging and testing
if __name__ == '__main__':
    node = RachelBot(DEBUG=True)
    node.run()
