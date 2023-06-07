#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy, math
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
from bot import Bot
import numpy as np

distance = .2

class BlueBot(Bot):

    def __init__(self, odom_positions, DEBUG=False):
        self.name = "blue_bot"#alec_
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

        # Initialize variables used for movement
        self.found_obstacle = False
        self.approaching_obstacle = True
        self.robot_circling = False
        self.robot_turning = False
        self.obstacle = 9999
        self.angle_searched = 0
        self.initialized = True


    def set_v(self, velocity, angular_velocity):
        """ The current velocity and angular velocity of the robot are set here
        """
        v1 = Vector3(velocity, 0.0, 0.0)
        v2 = Vector3(0.0, 0.0, angular_velocity)
        t = Twist(v1, v2)
        self.cmd_vel_pub.publish(t)


    def process_scan(self, data):
        """Make robot approach and circle obstacle
        """
        if not self.initialized:
            return
        # Find closest surface
        min_angle = numpy.argmin(np.asarray(data.ranges))#.index(min(distances))
        min_distance = data.ranges[min_angle]
        if not self.found_obstacle:
            # If obstacle is still not found
            if self.obstacle==9999:
                # If no obstacle in field of view, turn
                turn = 90*3.14159265/180
                self.angle_searched = self.angle_searched + 30
                self.set_v(0,turn)
                return 
            elif np.abs(self.obstacle)>3:
                # If obstacle in field of view, re-orient robot to face it
                turn = .003*self.obstacle
                self.set_v(0,turn)
                return 
            else: 
                # If robot facing obstacle, go towards it
                self.found_obstacle = True
                self.set_v(.2,0)
                return
        elif self.approaching_obstacle: 
            if not self.robot_turning and not self.robot_circling and data.ranges[0] > distance+.1: 
                # If approaching obstacle, keep moving forward
                self.set_v(.2,0)  
            elif not self.robot_turning and not self.robot_circling and self.obstacle>1:
                if self.obstacle == 9999: 
                    # If no obstacle, turn
                    turn = 90*3.14159265/180
                    self.set_v(0,turn)
                    return 
                else:
                    # Turn toward obstacle
                    turn = .005*self.obstacle
                    self.set_v(0,turn)
                    return 
            elif not self.robot_circling and min_angle<90:
                # Begin turning around obstacle
                self.robot_turning = True
                turn = .01*(min_angle + 90) #3*3.14159265/180
                self.set_v(0, turn)
                return
            elif not self.robot_circling and min_angle>280:
                # Begin turning around obstacle
                self.robot_turning = True
                turn = .01*(min_angle-250)
                self.set_v(0,turn)
            else: 
                self.robot_circling = True
                if min_angle<15 or min_angle>345:
                    # If facing obstacle, turn sideways
                    self.set_v(0.,.5)
                elif data.ranges[266]<data.ranges[270]: #min_angle < 267 and min_angle>90: 
                    # If angle between robot and obstacle is too large, re-orient
                    if data.ranges[270]>distance:
                        # If far away from obstacle, turn more
                        turn = -.3
                        self.set_v(0.03,turn)
                    else:
                        turn = -.15*(270 - min_angle) 
                        self.set_v(0.03,turn)
                elif data.ranges[274]<data.ranges[270]:
                    # If angle between robot and obstacle is too small, re-orient 
                    if data.ranges[270]>distance:
                        # If far away from obstacle, turn less
                        turn = .01*(min_angle - 270) 
                        self.set_v(0.03, turn)
                    else:
                        turn = .05*(min_angle - 270) 
                        self.set_v(0.03, turn)
                else:  
                    # Keep moving forward, at a small angle
                    err = min_angle - 270
                    turn = .005*err
                    self.set_v(0.05,turn)
                    self.following = True
                
        return


    def image_callback(self, data):
        """Identify any orange obstacles in front of robot
        """
        # converts the incoming ROS message to cv2 format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create orange mask
        lower_orange = numpy.array([ 10, 100, 20])
        upper_orange = numpy.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

        h, w, d = image.shape

        # using moments() function, the center of the colored pixels is determined
        M = cv2.moments(mask_orange)
        # if there are any colored pixels found
        if M['m00'] > 0:
                # center of the colored pixels in the image
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                err = w/2 - cx
                if err < 2:
                    self.obstacle = err
                    return

        # If no obstacle, set variable to arbitary value
        self.obstacle = 9999
        return

    def restart_bot(self):
        """Re-initialize variables, for resetting world
        """
        self.found_obstacle = False
        self.approaching_obstacle = True
        self.robot_circling = False
        self.robot_turning = False
        self.obstacle = 9999
        self.angle_searched = 0

    def run(self):
        if self.DEBUG:
            rospy.spin()


# this is for running the node manually for debugging and testing
if __name__ == '__main__':
    node = AlecBot(DEBUG=True)
    node.run()
