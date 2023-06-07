#!/usr/bin/env python3

import rospy, cv2, cv_bridge, numpy, math
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Vector3
import numpy as np
from bot import Bot

class KirBot(Bot):
    """ This is the predator robot. It has params that the genetic algorithm can set
    in order to test a new predator configuration.
    """

    def __init__(self, odom_positions, handle_odom_positions, DEBUG=False):
        super().__init__("kir_bot", odom_positions, handle_odom_positions)

        self.initialized = False

        self.DEBUG = DEBUG

        if self.DEBUG:
            rospy.init_node("kir_bot")

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber('/kir_bot/camera/rgb/image_raw',
                Image, self.image_callback, queue_size=10)

        # subscribe to the robot's scan topic
        rospy.Subscriber("/kir_bot/scan", LaserScan, self.process_scan, queue_size=10)

        # set up publisher and Twist to publish to /cmd_vel
        self.cmd_vel_pub = rospy.Publisher('/kir_bot/cmd_vel', Twist, queue_size=10)
        self.twist = Twist()

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        # max speed allowed for predator so it does not just fly around really fast
        self.max_speed = 0.75

        # these param values are the best we found from Generation 9 of training
        self.params = {
            "prey_weight": 0.7517077082714122, # weight for nearest prey direction
            "parallel_weight": 0.2961151492829175, # weight for driving parallel to nearest obstacle
            "away_weight": 0.6355636459871135, # weight for driving directly away from nearest obstacle

            # percentage prey-colored pixels that will force the predator to focus entirely on prey
            "prey_only_pixel_percent": 0.23153207282550584,

            # minimum target angle at which predator should only turn without driving
            "min_turn_only_angle": 0.3421713209284353,
            "base_speed": 0.5319568772282512, # base drive speed without error scaling
            "scaled_speed": 0.45962266127799195, # scales with angle err (lower err = higher speed)
            "angle_adjust_rate": 0.7144854185349765 # rate to adjust angle based on error from target angle
        }

        self.away_angle = 0
        self.parallel_angle = 0
        self.prey_angle = 0
        self.prey_in_frame = False
        self.pixel_percent = 0

        self.initialized = True


    def set_v(self, velocity, angular_velocity):
        """ The current velocity and angular velocity of the robot are set here
        """
        v1 = Vector3(velocity, 0.0, 0.0)
        v2 = Vector3(0.0, 0.0, angular_velocity)
        t = Twist(v1, v2)
        self.cmd_vel_pub.publish(t)


    def process_scan(self, data):
        if not self.initialized:
            return

        min_dist = None
        min_dist_index = None
        for i in range(len(data.ranges)):
            dist = data.ranges[i]
            if dist == math.inf:
                continue

            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_dist_index = i

        # determine here what what angle relative to the robot's forward would be needed
        # to go directly away and parallel to the nearest obstacle
        away_angle = 0
        parallel_angle = 0

        if min_dist is not None:
            away_angle = (min_dist_index / 360) * 2 * math.pi + math.pi

            if min_dist_index < 180:
                # left
                diff = min_dist_index - 90
            else:
                # right
                diff = min_dist_index - 270
            parallel_angle = diff / 360 * 2 * math.pi

        away_angle = self.normalize_radian_angle(away_angle)
        parallel_angle = self.normalize_radian_angle(parallel_angle)

        self.away_angle = away_angle
        self.parallel_angle = parallel_angle
        # target_angle = parallel_angle
        self.move_robot()

    def move_robot(self):
        """ This is where all the predator params come into play to determine how the
        predator should move
        """
        target_angle = 0

        # the multiplied weights always need to add up to 1

        total_weight = 0
        use_prey = self.prey_in_frame
        use_other_weights = True
        if self.pixel_percent >= self.params["prey_only_pixel_percent"]:
            use_other_weights = False

        if use_prey:
            total_weight += self.params["prey_weight"]
        if use_other_weights:
            total_weight += self.params["away_weight"] + self.params["parallel_weight"]

        if total_weight == 0:
            total_weight = 0.0001

        # calculate the target angle using the various weight params
        if use_prey:
            target_angle = self.prey_angle * (self.params["prey_weight"] / total_weight)
        if use_other_weights:
            target_angle += self.away_angle * (self.params["away_weight"] / total_weight) + self.parallel_angle * (self.params["parallel_weight"]  / total_weight)

        target_angle = self.normalize_radian_angle(target_angle)

        abs_err = abs(target_angle)

        # use navigation params to get the robot in the direction it desires
        err_handling_speed = self.params["base_speed"] * self.max_speed + self.params["scaled_speed"] / (abs_err + .0001)
        speed = min(self.max_speed, err_handling_speed)
        if abs_err > (self.params["min_turn_only_angle"] * 3):
            speed = 0

        angular_speed = target_angle * self.params["angle_adjust_rate"]
        self.set_v(speed, angular_speed)

    def normalize_degree_angle(self, angle):
        new_angle = angle
        while (new_angle <= -180):
            new_angle += 360
        while (new_angle > 180):
            new_angle -= 360
        return new_angle

    def normalize_radian_angle(self, angle):
        return math.radians(self.normalize_degree_angle(math.degrees(angle)))

    def image_callback(self, msg):
        """ Use this image callback to determine the most centered prey and the percentage
        of the image that this prey occupies as a way of determining approximately how
        close the prey is to the predator.
        """
        if not self.initialized:
            return
        
        self.prey_angle = 0
        self.prey_in_frame = False

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
