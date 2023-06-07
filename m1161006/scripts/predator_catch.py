#!/usr/bin/env python3

import rospy

from geometry_msgs.msg import Twist, Pose
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import math

class PredatorCatch(object):

    def __init__(self, max_time, reset_world, DEBUG=False):
        self.max_time = max_time
        self.DEBUG = DEBUG

        if self.DEBUG:
            rospy.init_node("predator_catch")

        self.reset_world = reset_world
        self.last_reset_time = rospy.get_time()

        self.last_dist_print = rospy.get_time()

        self.prev_predator_pose = None
        self.predator_travel_dists = []

        self.min_capture_time = 8

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.get_models, queue_size=1)

        # rospy.wait_for_service('/gazebo/delete_model')
        # self.delete_proxy = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)


    def handle_reset_timer(self):
        # handles the timer in charge of identifying if reset is needed
        now = rospy.get_time()
        time_diff = now - self.last_reset_time

        # if the time_diff is less than 0.5 seconds, this means that we must wait
        # 0.5 seconds between certain operations
        # otherwise, if we have passed the max time allowed, the world is reset as
        # the robot has failed to capture any prey
        if time_diff < .5:
            return True
        elif time_diff >= self.max_time:
            print(f'{self.max_time}s passed with no capture')
            self.reset(False)
            return True

        return False


    def clear_predator_poses(self):
        # we remove predator poses from our list if they are more than 3 seconds old
        now = rospy.get_time()
        removed = False
        while len(self.predator_travel_dists) > 0 and now - self.predator_travel_dists[0]["time"] > 3:
            self.predator_travel_dists.pop(0)
            removed = True

        return removed


    def predator_stuck(self, p1):
        # if there is no previous pose, we can't know if we are stuck
        if self.prev_predator_pose is None:
            self.prev_predator_pose = p1
            return False

        # calculate our movement between previous and current predator pose
        p2 = self.prev_predator_pose
        self.prev_predator_pose = p1
        dist = math.sqrt(pow(p1.position.x - p2.position.x, 2) + pow(p1.position.y - p2.position.y, 2))
        angle_change = abs(self.get_yaw_from_pose(p1) - self.get_yaw_from_pose(p2))

        # add to list of predator movements
        self.predator_travel_dists.append({
            "time": rospy.get_time(),
            "dist": dist,
            "angle_change": angle_change
        })
        # if we have cleared items from our list calculate the movement of the
        # robot in the last 3 seconds, if it is within margins, reset the world
        # as the predator is stuck
        if self.clear_predator_poses():
            tot_dist = 0
            tot_angle_change = 0
            for d in self.predator_travel_dists:
                tot_dist += d["dist"]
                tot_angle_change += d["angle_change"]

            if tot_dist < 0.1 and tot_angle_change < 0.1:
                print('Predator Stuck')
                self.reset(False)
                #return True
                return False

        return False


    def get_models(self, data):
        # make sure we want to be getting models
        if self.handle_reset_timer():
            return

        # add the prey names and poses to lists
        # also collect the predators pose
        predator_pose = Pose()
        prey_name = []
        prey_pose = []

        for i in range(len(data.name)):
            name = data.name[i]
            if self.is_predator_name(name):
                predator_pose = data.pose[i]
            elif self.is_prey_name(name):
                prey_name.append(name)
                prey_pose.append(data.pose[i])

        # check if a capture has happened
        self.handle_capture_test(predator_pose, prey_name, prey_pose)


    def get_odom_poses(self, odom_poses):
        # make sure we want to get odom poses
        if self.handle_reset_timer():
            return
            
        # use odometry data to keep track of location of predator and prey
        predator_pose = Pose()
        prey_name = []
        prey_pose = []

        for name in odom_poses:
            pose = odom_poses[name]
            if self.is_predator_name(name):
                predator_pose = pose
            elif self.is_prey_name(name):
                prey_name.append(name)
                prey_pose.append(pose)

        # check if a capture has happened
        self.handle_capture_test(predator_pose, prey_name, prey_pose)


    def is_predator_name(self, name):
        # check if robot name is the predator name
        return name == "kir_bot"


    def is_prey_name(self, name):
        # check if robot name is a prey name
        return name == "yellow_bot" or name == "blue_bot" or name == "green_bot"


    def get_yaw_from_pose(self, p):
        yaw = (euler_from_quaternion([
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w])
                [2])
        return yaw


    def actual_pose(self, p, log=False):
        # fix our pose to have proper x and y values
        x = p.position.x
        y = p.position.y
        yaw = self.get_yaw_from_pose(p)
        l = .06
        dx = l * math.cos(yaw)
        dy = l * math.sin(yaw)
        if log:
            print(dx, dy)

        return x - dx, y - dy


    def handle_capture_test(self, predator_pose, prey_name, prey_pose):
        # if the predator is stuck don't test for capture
        if self.predator_stuck(predator_pose):
            return

        prey_dist = []

        pred_x, pred_y = self.actual_pose(predator_pose)

        # calculate proper poses for prey and calculate distance from predator to prey
        for pose in prey_pose:
            prey_x, prey_y = self.actual_pose(pose)
            dist = math.sqrt(pow(prey_x - pred_x, 2) + pow(prey_y - pred_y, 2))
            prey_dist.append(dist)

        now = rospy.get_time()
        if now - self.last_dist_print > 1:
            self.last_dist_print = now

        # loop through distance to prey, if within margin, it is captured
        # we record the time it took for capture and reset the world
        for i in range(len(prey_dist)):
            # if prey_name[i] == 'alec_bot':
            #     self.actual_pose(prey_pose[i], True)

            if prey_dist[i] < 0.32:
                # prey has been captured
                print(f'Captured {prey_name[i]}')
                capture_diff = rospy.get_time() - self.last_reset_time
                # self.reset(max(self.min_capture_time, capture_diff))
                self.reset(True)
                # self.delete_proxy(prey_name[i])

                return


    def reset(self, captured):
        # reset elements for next run
        self.prev_predator_pose = None
        self.predator_travel_dists = []

        self.last_reset_time = rospy.get_time()
        self.reset_world(captured)
