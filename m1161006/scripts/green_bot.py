#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy, math
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
from bot import Bot

# How close we will get to wall.
distance = .4

class GreenBot(Bot):

    def __init__(self, odom_positions, DEBUG=False):
        self.name = "green_bot"#sydney
        super().__init__(self.name, odom_positions)

        self.initialized = False

        self.DEBUG = DEBUG

        if self.DEBUG:
            rospy.init_node(self.name)

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber(f'/{self.name}/camera/rgb/image_raw',
                Image, self.use_color_data, queue_size=10)

        # subscribe to the robot's scan topic
        rospy.Subscriber(f"/{self.name}/scan", LaserScan, self.callback, queue_size=10)
        # set up publisher and Twist to publish to /cmd_vel
        self.cmd_vel_pub = rospy.Publisher(f'/{self.name}/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        # initialize variables used for movement
        self.found_wall = False
        self.concave_turn = False
        self.obstacle = 9999
        self.following = False

        self.initialized = True


    def set_v(self, velocity, angular_velocity):
        """ The current velocity and angular velocity of the robot are set here
        """
        v1 = Vector3(velocity, 0.0, 0.0)
        v2 = Vector3(0.0, 0.0, angular_velocity)
        t = Twist(v1, v2)
        self.cmd_vel_pub.publish(t)


    def process_scan(self, data):
        """ Spinning behavior used for testing
        """
        self.twist.linear.x = -0.1
        self.cmd_vel_pub.publish(self.twist)
        return

    def use_color_data(self, data):
        """ Determine if there is obstacle ahead
        """
        # converts the incoming ROS message to cv2 format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # identify orange
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

        # if no obstacle, set variable to some fake value
        self.obstacle = 9999
        return

    def callback(self, data):
            """ Move robot to wall and begin followingit
            """
            if not self.initialized:
                return
            min_angle = numpy.argmin(np.asarray(data.ranges))#.index(min(distances))
            min_distance = data.ranges[min_angle]

            # check if we have found the wall yet
            if not self.found_wall:
                if self.obstacle < 9999: 
                    # If there is an obstacle ahead, modify orientation slightly to avoid it
                    turn = .2
                    self.set_v(0.2,turn)
                elif min_distance < distance and min_angle>10 and min_angle<350: 
                    # if there is a wall, but it's at the wrong angle, re-orient
                    diff = min_angle-180
                    diff_abs = np.abs(np.abs(diff)-180)/180
                    turn = .5*diff_abs if diff<0 else -.5*diff
                    self.set_v(0.0,turn)
                elif min_distance < distance: 
                    # if wall directly ahead, then stop
                    self.set_v(0,0)
                    self.found_wall = True 
                else: 
                    # Move forward in search of wall
                    self.set_v(0.2,0)
                return 
            else:
                if self.following and self.concave_turn:
                    # If it is time to turn an outer corner
                    if min_angle>268 and min_angle < 275: 
                        # If wall is again to the right, then turn is complete
                        self.concave_turn = False
                    elif self.turn_count > 50:
                        # if robot keeps turning, something went wrong, search for wall again
                        self.found_wall = False
                        self.concave_turn = False
                        self.following = False
                    else: 
                        # Turn at outer corner
                        distance_in_range = data.ranges[200:300]
                        turn = -60*3.14159265/180
                        self.set_v(0.1, turn)
                        self.turn_count = self.turn_count+1
                elif self.obstacle==9999 and self.following and data.ranges[0] <= distance+.1:
                    # Turn inner corner
                    turn = 30*3.14159265/180
                    self.set_v(0.05, turn)
                    rospy.sleep(1)
                elif self.following and data.ranges[275] > distance+.4:
                    # prepare for outer turn 
                    self.concave_turn = True
                    self.turn_count = 0
                else: 
                    # go straight
                    if min_angle < 267 and min_angle>90: 
                        # If angle between robot and wall too small, adjust
                        turn = -.02*(270 - min_angle) #-3*3.14159265/180
                        self.set_v(0,turn)
                    elif min_angle>267 and min_angle<275: 
                        # Angle is pretty good
                        if data.ranges[270]>.2:
                            # If robot is too far from wall, get closer
                            turn = -.02*(3) #-3*3.14159265/180
                            self.set_v(0.3,turn)
                        else: 
                            # Keep going straight
                            err = min_angle - 270
                            turn = .001*err
                            self.set_v(0.3,0)#turn)
                            self.following = True
                    elif min_angle>=275: 
                        # If angle between robot and wall too large, adjust
                        turn = .006*(min_angle - 270) #3*3.14159265/180
                        self.set_v(0, turn)
                    else: 
                        # If angle is very off, make larger tur
                        turn = .01*(min_angle + 90) #3*3.14159265/180
                        self.set_v(0, turn)
            return
                        
    def turn_to_wall(self, min_distance):
        """Makes robot turn to wall, used in older version
        """
        if min_distance < 3: 
            ang = min_distance*90*3.14159265/180
            self.set_v(0, ang)
            self.set_v(0,0)
        else: 
            ang = -90*3.14159265/180
            self.set_v(0, ang)
            self.set_v(0,0)
        return 

    def restart_bot(self):
        """ re-initialize variables for re-setting world
        """
        self.found_wall = False
        self.concave_turn = False
        self.obstacle = 9999
        self.following = False
        return

    def run(self):
        if self.DEBUG:
            rospy.spin()


# this is for running the node manually for debugging and testing
if __name__ == '__main__':
    node = SydneyBot(DEBUG=True)
    node.run()
