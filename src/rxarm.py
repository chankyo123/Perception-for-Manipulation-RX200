"""!
Implements the RXArm class.

The RXArm class contains:

* last feedback from joints
* functions to command the joints
* functions to get feedback from joints
* functions to do FK and IK
* A function to read the RXArm config file

You will upgrade some functions and also implement others according to the comments given in the code.
"""
import numpy as np
from functools import partial
from kinematics import FK_dh, FK_pox, get_pose_from_T
import time
import csv
import sys, os

from builtins import super
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from resource.config_parse import parse_dh_param_file
from sensor_msgs.msg import JointState
import rclpy
from rclpy.executors import SingleThreadedExecutor
from kinematics import IK_geometric

sys.path.append('../../interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot') 
from arm import InterbotixManipulatorXS
from mr_descriptions import ModernRoboticsDescription as mrd

"""
TODO: Implement the missing functions and add anything you need to support them
"""
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def _ensure_initialized(func):
    """!
    @brief      Decorator to skip the function if the RXArm is not initialized.

    @param      func  The function to wrap

    @return     The wraped function
    """
    def func_out(self, *args, **kwargs):
        if self.initialized:
            return func(self, *args, **kwargs)
        else:
            print('WARNING: Trying to use the RXArm before initialized')

    return func_out


class RXArm(InterbotixManipulatorXS):
    """!
    @brief      This class describes a RXArm wrapper class for the rx200
    """
    def __init__(self, dh_config_file=None):
        """!
        @brief      Constructs a new instance.

                    Starts the RXArm run thread but does not initialize the Joints. Call RXArm.initialize to initialize the
                    Joints.

        @param      dh_config_file  The configuration file that defines the DH parameters for the robot
        """
        super().__init__(robot_model="rx200")
        self.joint_names = self.arm.group_info.joint_names
        self.num_joints = 5
        # Gripper
        self.gripper_state = True
        # State
        self.initialized = False
        # Cmd
        self.position_cmd = None
        self.moving_time = 2.0
        self.accel_time = 0.5
        # Feedback
        self.position_fb = None
        self.velocity_fb = None
        self.effort_fb = None
        # DH Params
        self.dh_params = []
        self.dh_config_file = dh_config_file
        if (dh_config_file is not None):
            self.dh_params = RXArm.parse_dh_param_file(dh_config_file)
        #POX params
        self.M = np.array([[1.0, 0.0, 0.0, -3.0],
              [0.0, 1.0, 0.0, 0.426 * 1000],
              [0.0, 0.0, 1.0, 0.28 * 1000],
              [0.0, 0.0, 0.0, 1.0]])

        self.Slist = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [-1.0, .0, 0.0, 0.0, -0.10391 * 1000, 0.0],
                  [-1.0, 0.0, 0.0, 0.0, -0.30457 * 1000, 0.05 * 1000],
                  [-1.0, 0.0, 0.0, 0.0, -0.30457 * 1000, 0.25 * 1000],
                  [0.0, 1.0, 0.0, 0.30457 * 1000, 0.0, 0.0]])

    def set_trajectory_time(self, moving_time, accel_time):
        self.arm.set_trajectory_time(moving_time, accel_time)
        
    def initialize(self):
        """!
        @brief      Initializes the RXArm from given configuration file.

                    Initializes the Joints and serial port

        @return     True is succes False otherwise
        """
        self.initialized = False
        # Wait for other threads to finish with the RXArm instead of locking every single call
        time.sleep(0.25)
        """ Commanded Values """
        self.position = [0.0] * self.num_joints  # radians
        """ Feedback Values """
        self.position_fb = [0.0] * self.num_joints  # radians
        self.velocity_fb = [0.0] * self.num_joints  # 0 to 1 ???
        self.effort_fb = [0.0] * self.num_joints  # -1 to 1

        # Reset estop and initialized
        self.estop = False
        self.enable_torque()
        self.moving_time = 2.0
        self.accel_time = 0.5
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=False)
        self.gripper.release()
        self.initialized = True
        return self.initialized

    def sleep(self):
        self.moving_time = 2.0
        self.accel_time = 1.0
        self.arm.go_to_home_pose(moving_time=self.moving_time,
                             accel_time=self.accel_time,
                             blocking=True)
        self.arm.go_to_sleep_pose(moving_time=self.moving_time,
                              accel_time=self.accel_time,
                              blocking=False)
        self.initialized = False

    def set_positions(self, joint_positions):
        """!
         @brief      Sets the positions.

         @param      joint_angles  The joint angles
         """
        self.arm.set_joint_positions(joint_positions,
                                 moving_time=self.moving_time,
                                 accel_time=self.accel_time,
                                 blocking=False)

    def set_moving_time(self, moving_time):
        self.moving_time = moving_time

    def set_accel_time(self, accel_time):
        self.accel_time = accel_time

    def disable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 0)

    def enable_torque(self):
        """!
        @brief      Disables the torque and estops.
        """
        self.core.robot_set_motor_registers('group', 'all', 'Torque_Enable', 1)

    def get_positions(self):
        """!
        @brief      Gets the positions.

        @return     The positions.
        """
        return self.position_fb

    def get_velocities(self):
        """!
        @brief      Gets the velocities.

        @return     The velocities.
        """
        return self.velocity_fb

    def get_efforts(self):
        """!
        @brief      Gets the loads.

        @return     The loads.
        """
        return self.effort_fb

    def move_to_pose(self, pose, gripper, time_params=None):
        """
        Smoothly moves the arm to the specified pose by calculating inverse kinematics (IK)
        and setting joint positions with adaptive timing.

        Params:
            pose (array-like): A 6-element pose vector [x, y, z, roll, pitch, yaw].
            time_params (float or tuple, optional): 
                - If a single number, it's treated as the moving_time.
                - If a tuple, it should be (accel_time, moving_time).
        """

        pose = np.array(pose, copy=True)
        # print("pose : ", pose)
        # Validate pose length
        # if pose.shape[0] != 6:
        #     print(f"Error: Pose must have 6 elements [x, y, z, roll, pitch, yaw]. Received pose with {pose.shape[0]} elements.")
        #     return

        # pose = np.array([pose[0], pose[1], pose[2], np.pi/2])
        
        # Calculate inverse kinematics
        all_joint_positions = IK_geometric(pose)
        # print("all_joint_positions shape :", all_joint_positions)
        # print("num of poosible ik solutions : ", all_joint_positions.shape)
        joint_positions = all_joint_positions[1]
       

        # Determine timing parameters
        if time_params is not None:
            if isinstance(time_params, (int, float)):
                moving_time = time_params + 1
                accel_time = 1
            else:
                moving_time, accel_time = time_params
        else:
            moving_time = 1.5
            accel_time = .3

        # Attempt to set joint positions
        print(joint_positions)
        self.set_positions(joint_positions)

        time.sleep(1)
        
        if gripper == 'release':
            self.gripper.release()
            print("Gripper released (opened).")
        elif gripper == 'grip':
            self.gripper.grasp()
            print("Gripper grasped (closed).")
        else:
            print('Gripper stays the same.')

        # Wait for the movement to complete
        time.sleep(moving_time + 0.05)

        # Reset acceleration and moving time to default values
        self.set_accel_time(0.5)
        self.set_moving_time(2)


#   @_ensure_initialized

    def get_ee_pose(self):
        """!
        @brief      TODO Get the EE pose. Distances should be in mm

        @return     The EE pose as [x, y, z, phi, theta, psi]
        """
        joint_angles = self.position_fb.T
        ee_T = FK_pox(joint_angles, self.M, self.Slist)
        ee_pose = get_pose_from_T(ee_T)
        # print(ee_pose)
        # return T
        return ee_pose
        # return [0,0,0,0,0,0]


    def parse_pox_param_file(self):
        """!
        @brief      TODO Parse a PoX config file

        @return     0 if file was parsed, -1 otherwise 
        """
        return -1

    def parse_dh_param_file(self):
        print("Parsing DH config file...")
        dh_params = parse_dh_param_file(self.dh_config_file)
        print("DH config file parse exit.")
        return dh_params

    def get_dh_parameters(self):
        """!
        @brief      Gets the dh parameters.

        @return     The dh parameters.
        """
        return self.dh_params


class RXArmThread(QThread):
    """!
    @brief      This class describes a RXArm thread.
    """
    updateJointReadout = pyqtSignal(list)
    updateEndEffectorReadout = pyqtSignal(list)

    def __init__(self, rxarm, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      RXArm  The RXArm
        @param      parent  The parent
        @details    TODO: set any additional initial parameters (like PID gains) here
        """
        QThread.__init__(self, parent=parent)
        self.rxarm = rxarm
        self.node = rclpy.create_node('rxarm_thread')
        self.subscription = self.node.create_subscription(
            JointState,
            '/rx200/joint_states',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def callback(self, data):
        self.rxarm.position_fb = np.asarray(data.position)[0:5]
        self.rxarm.velocity_fb = np.asarray(data.velocity)[0:5]
        self.rxarm.effort_fb = np.asarray(data.effort)[0:5]
        self.updateJointReadout.emit(self.rxarm.position_fb.tolist())
        self.updateEndEffectorReadout.emit(self.rxarm.get_ee_pose())
        #for name in self.rxarm.joint_names:
        #    print("{0} gains: {1}".format(name, self.rxarm.get_motor_pid_params(name)))
        if (__name__ == '__main__'):
            print(self.rxarm.position_fb)

    def run(self):
        """ Spin the executor """
        try:
            while rclpy.ok():
                self.executor.spin_once(timeout_sec=0.02)
        finally:
            self.node.destroy_node()
            rclpy.shutdown()

    def stop(self):
        """!
        @brief      Stops the thread.
        """
        try:
            self.wait()
            self.executor.shutdown()
            self.node.destroy_node()
        except:
            pass


if __name__ == '__main__':
   rclpy.init()  # for test
   rxarm = RXArm()
   print(rxarm.joint_names)
   armThread = RXArmThread(rxarm)
   armThread.start()
   try:
       joint_positions = [-1.0, 0.5, 0.5, 0, 1.57]
       rxarm.initialize()

       rxarm.arm.go_to_home_pose()
       rxarm.set_gripper_pressure(0.5)
       rxarm.set_joint_positions(joint_positions,
                                 moving_time=2.0,
                                 accel_time=0.5,
                                 blocking=True)
       rxarm.gripper.grasp()
       rxarm.arm.go_to_home_pose()
       rxarm.gripper.release()
       rxarm.sleep()

   except KeyboardInterrupt:
       print("Shutting down")
   finally:
       if armThread.isRunning():
           armThread.stop()
       rclpy.shutdown()
