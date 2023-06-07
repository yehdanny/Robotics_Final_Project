#!/usr/bin/env python3

import rospy, numpy

from geometry_msgs.msg import Twist, Vector3, Pose
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics, ModelState
from gazebo_msgs.srv import SetPhysicsProperties, SetModelState
from blue_bot import BlueBot
from green_bot import GreenBot
from yellow_bot import YellowBot
#from kir_bot import KirBot
from predator_rule import KirBot#rulebased
from predator_catch import PredatorCatch
from genetic_algorithm import GeneticAlgorithm

class RobotMovement(object):
    def __init__(self):
        self.initialized = False

        rospy.init_node("robot_movement")

        self.default_physics = False
        # set this to True to run training
        self.training = False
        self.disable_capture = False
        self.max_time = 30

        if self.training:
            self.genetic_algorithm = GeneticAlgorithm(self.max_time, load=True)
        
        self.kir_bot = None

        self.init_gazebo()

        rospy.sleep(2)

        self.reset_world()

        self.odom_positions = {}

        # run prey movement
        self.blue_bot = BlueBot(self.odom_positions)
        self.green_bot = GreenBot(self.odom_positions)
        self.yellow_bot = YellowBot(self.odom_positions)

        if self.disable_capture:
            odom_pose_callback = lambda x: x
        else:
            self.predator_catch = PredatorCatch(self.max_time, self.reset_world)
            odom_pose_callback = self.predator_catch.get_odom_poses

        self.kir_bot = KirBot(self.odom_positions, odom_pose_callback)
        if self.training:
            self.kir_bot.params = self.genetic_algorithm.get_params()

        print("Initialized")
        self.initialized = True


    def init_gazebo(self):
        # set up the services we use
        # reset world to reset world for repeated runs
        # set physics properties so we can modify the physics, allowing quicker test iterations
        # set model state so we can randomly spawn the robots
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_gazebo_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        rospy.wait_for_service('/gazebo/set_physics_properties')
        self.set_gazebo_physics_props = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.set_physics_props()


    def set_physics_props(self):
        # set various physics properties of the gazebo
        # this is used so that for testing/training we can rapidly have iterations
        ode_config = ODEPhysics()
        ode_config.auto_disable_bodies = False
        ode_config.sor_pgs_precon_iters = 0
        ode_config.sor_pgs_iters = 50
        ode_config.sor_pgs_w = 1.3
        ode_config.sor_pgs_rms_error_tol = 0.0
        ode_config.contact_surface_layer = 0.001
        ode_config.contact_max_correcting_vel = 100.0
        ode_config.cfm = 0.0
        ode_config.erp = 0.2
        ode_config.max_contacts = 20
        gravity = Vector3(0.0, 0.0, -9.8)

        time_step = 0.001
        max_update_rate = 0.0
        if self.default_physics:
            time_step = 0.001
            max_update_rate = 1000.0
        self.set_gazebo_physics_props(time_step, max_update_rate, gravity, ode_config)


    def reset_world(self, captured=None):
        # self.set_physics_props()
        self.reset_gazebo_world()

        if self.training:
            # if score_time is not None:
                # self.genetic_algorithm.set_score_by_time(score_time)
            if captured is not None:
                self.genetic_algorithm.set_score_by_capture(captured)

            if self.kir_bot is not None:
                self.kir_bot.params = self.genetic_algorithm.get_params()
            
            self.genetic_algorithm.print_progress()

        # This code is used to spawn the robots randomly when the world is reset
        # the first thing we do is create a zero_twist vector to set the robot's
        # velocities to 0
        zero_twist = Twist()
        zero_twist.linear.x = 0
        zero_twist.linear.y = 0
        zero_twist.linear.z = 0
        zero_twist.angular.x = 0
        zero_twist.angular.y = 0
        zero_twist.angular.z = 0

        # next we create ModelState objects for each robot that will be used to
        # publish their new spawn location
        # The first thing we do is set their model names and set their orientations
        # to 0
        kir_state = ModelState()
        kir_pose = Pose()
        kir_pose.orientation.x = 0
        kir_pose.orientation.y = 0
        kir_pose.orientation.z = 0
        kir_pose.orientation.w = 0
        kir_state.model_name = 'kir_bot'

        yellow_state = ModelState()
        yellow_pose = Pose()
        yellow_pose.orientation.x = 0
        yellow_pose.orientation.y = 0
        yellow_pose.orientation.z = 0
        yellow_pose.orientation.w = 0
        yellow_state.model_name = 'yellow_bot'

        green_state = ModelState()
        green_pose = Pose()
        green_pose.orientation.x = 0
        green_pose.orientation.y = 0
        green_pose.orientation.z = 0
        green_pose.orientation.w = 0
        green_state.model_name = 'green_bot'

        blue_state = ModelState()
        blue_pose = Pose()
        blue_pose.orientation.x = 0
        blue_pose.orientation.y = 0
        blue_pose.orientation.z = 0
        blue_pose.orientation.w = 0
        blue_state.model_name = 'blue_bot'

        # These are the chosen possible spawn locations
        # an image of them can be found in the readme
        spawn_locs = numpy.array(
            [[-2, 0, 0],
            [-2, 1, 0],
            [-1, 0, 0],
            [0, 0, 0],
            [0, -1, 0],
            [0, -2, 0],
            [0, -3, 0],
            [0, -4, 0],
            [1, 0, 0],
            [1, -1, 0],
            [1, -3, 0],
            [2, 0, 0],
            [2, -1, 0],
            [2, -2, 0],
            [3, 1, 0],
            [4, 1, 0],
            [3, -1, 0],
            [3, -3, 0]])

        # this selects a predator spawn location from the full list of spawn locations,
        # then limits the prey spawn locations such that it is not extremely easy for the
        # predator to capture the prey (specifically it prevents prey from spawning directly
        # in front of the predator)
        predator_spawn_index = numpy.random.choice(spawn_locs.shape[0])
        predator_spawn = spawn_locs[predator_spawn_index]
        spawn_locs = numpy.delete(spawn_locs, predator_spawn_index, axis=0)

        remove_indices = numpy.array([])
        for i in range(spawn_locs.shape[0]):
            spawn_loc = spawn_locs[i]
            if spawn_loc[1] == predator_spawn[1] and spawn_loc[0] > predator_spawn[0]:
                remove_indices = numpy.append(remove_indices, i)

        if remove_indices.shape[0] > 0:
            spawn_locs = numpy.delete(spawn_locs, remove_indices.astype(int), axis=0)

        # we now select the prey locations
        spawn_index = numpy.random.choice(spawn_locs.shape[0], 3, False)

        # set the predator spawn location with the previously selected values
        kir_pose.position.x = predator_spawn[0]
        kir_pose.position.y = predator_spawn[1]
        kir_pose.position.z = predator_spawn[2]

        # set each of the prey positions with the previously selected values
        yellow_pose.position.x = spawn_locs[spawn_index[0]][0]
        yellow_pose.position.y = spawn_locs[spawn_index[0]][1]
        yellow_pose.position.z = spawn_locs[spawn_index[0]][2]

        green_pose.position.x = spawn_locs[spawn_index[1]][0]
        green_pose.position.y = spawn_locs[spawn_index[1]][1]
        green_pose.position.z = spawn_locs[spawn_index[1]][2]

        blue_pose.position.x = spawn_locs[spawn_index[2]][0]
        blue_pose.position.y = spawn_locs[spawn_index[2]][1]
        blue_pose.position.z = spawn_locs[spawn_index[2]][2]

        # set poses of each state
        kir_state.pose = kir_pose
        yellow_state.pose = yellow_pose
        green_state.pose = green_pose
        blue_state.pose = blue_pose

        # set twist of each state
        kir_state.twist = zero_twist
        yellow_state.twist = zero_twist
        green_state.twist = zero_twist
        blue_state.twist = zero_twist

        # use the Gazebo Service to set the model state of each robot
        # this will place each robot in its randomly selected location
        self.set_model_state(kir_state)
        self.set_model_state(yellow_state)
        self.set_model_state(green_state)
        self.set_model_state(blue_state)

        try:
            self.green_bot.restart_bot()
            self.blue_bot.restart_bot()
        except: 
            #self.bots not yet initialized
            pass


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = RobotMovement()
    node.run()
