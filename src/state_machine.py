"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
from rxarm import RXArm
from Perception import *
from AprilTag import *
from camera import *
from kinematics import *
from Ball import *  
import kinematics

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm:RXArm, camera:Camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm:RXArm = rxarm
        self.camera:Camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.demo_waypoints = np.zeros(shape=(0,5))
        self.msg = True
        self.recording = False
        self.r = None
        self.min_moving_time = 0.8  # Minimum time for very close waypoints
        self.min_accel_time = 0.1
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]
        self.waypoint_pos = []
        self.waypoint_vel = []
        self.waypoint_gripper = []
        self.initalization_teach = False
        self.mouse_world = None
        

    def set_next_state(self, state, msg = True):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state
        self.msg = msg

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """            
            
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "teach_init":
            self.teach_init()

        if self.next_state == "teach":
            self.teach()

        if self.next_state == "repeat":
            self.repeat()
            
        if self.next_state == "change gripper":
            self.change_gripper_state()

        if self.next_state == "record":
            self.record(self.msg)
        else:
            self.recording = False

        if self.next_state == "replay":
            self.replay()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == 'ik_checkpoint3':
            self.ik_checkpoint3()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == 'click_pick':
            print('Click and Drop Mode Activated')
            self.click_drop_open()
            
        if self.next_state == 'click_place':
            self.click_drop_close()
            
        if self.next_state == "cubemode":
            self.cubemode(self. msg)

        if self.next_state == "ball":
            self.ball(self. msg)
            

    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"
        self.recording = False
    

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def dont_break_me(self):
        a = self.rxarm.position_fb 
        self.rxarm.set_trajectory_time(1, 0.5)
        self.rxarm.set_positions(a)
        
    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.dont_break_me()
        if self.r is not None:
            self.waypoints = self.r
        self.rxarm.enable_torque()
        self.status_message = "State: Execute - Executing motion plan"
        # self.waypoints = np.loadtxt("gs.out")
        for waypoint in self.waypoints:
            self.fluid_move(waypoint)
        
        np.savetxt("goodshit.out", self.waypoints)    
        # self.fluid_move(IK_geometric([250, 250, 50, np.pi/2]))
        
            # print("diff: ", diff)
        self.next_state = "idle"

    def teach_init(self):
        self.initalization_teach = False
        self.next_state = "idle"

    def change_gripper_state(self):
        if self.rxarm.gripper_state == True:
            self.rxarm.gripper_state = False
        else:
            self.rxarm.gripper_state = True
        self.next_state = "idle"

    def teach(self):
        
        if self.initalization_teach:
            self.waypoint_pos = []
            self.waypoint_vel = []
            self.waypoint_gripper = []

        self.waypoint_pos.append(self.rxarm.get_positions())
        print("pos : ", self.rxarm.get_positions())
        self.waypoint_vel.append(self.rxarm.get_velocities())
        self.waypoint_gripper.append(self.rxarm.gripper_state)
        print("gripper pos : ", self.rxarm.gripper_state)
        self.next_state = "idle"

    def repeat(self):
        print(self.waypoint_pos)
        print(self.waypoint_gripper)
        self.initalization_teach = True
    
        for i, pos in enumerate(self.waypoint_pos):
            print(i, pos, self.waypoint_gripper[i])
            self.rxarm.set_positions(pos)
            time.sleep(1)
            # if self.estop:
            #     return
            
            if self.waypoint_gripper[i]:
                # this is when gripper is open
                self.rxarm.gripper.release()
            else:
                # this is when gripper is closed
                self.rxarm.gripper.grasp()
            
            time.sleep(1)
        self.next_state = "idle"


    def reached_waypoint(self, waypoint, tolerance=0.07):
        """Check if the arm has reached the desired waypoint within a tolerance"""
        current_position = self.rxarm.get_positions()
        _reached = np.max(np.abs(current_position - waypoint)) < tolerance
        if _reached:
            time.sleep(0.02)
        return _reached
    
    def fluid_move(self, waypoint):
        curpoint = self.rxarm.get_positions()
        diff = 1 * np.max(np.abs(curpoint - waypoint))
        self.rxarm.set_positions(waypoint)
        if diff <= 0.3:
            pass
        moving_time = max(diff, self.min_moving_time)
        accel_time = max(diff / 4.0, self.min_accel_time)
        self.rxarm.set_accel_time(accel_time)
        self.rxarm.set_moving_time(moving_time)
        _curr = time.time()
        while not self.reached_waypoint(waypoint):
            # print("goin to waypoint")
            time.sleep(0.01)
            if (time.time() - _curr) > max(4, moving_time):
                break
            
    def record(self, msg = True):
        """!
        @brief Teach and repeat part1 (record the teach stuff)
        """
        self.dont_break_me()
        print(msg, self.recording)
        if not self.recording and msg:
            self.recording = True
            self.status_message = "State: RECORDING - ready to capture movements"
            self.rxarm.sleep()
            time.sleep(4.1)
            self.rxarm.disable_torque()
            self.r = None
        current_position = self.rxarm.position_fb

        if self.r is None:
            self.r = np.array([current_position])
            # print(f"Initial position recorded: {self.r}")
        else:
            if np.max(np.abs(self.r[-1] - current_position)) > 0.15:
                self.r = np.append(self.r, np.array([current_position]), 0)
                # print(self.r.size, self.r)

        self.status_message = "State: RECORDING - recording motion plan: " + str(current_position)
        
        if msg:
            self.next_state = "record"
        else:
            self.next_state = "idle"
            self.recording = False
            self.status_message = "State: RECORDING COMPLETED - Movements captured"
            
            time.sleep(2)
             
    def cubemode(self, msg):
        self.camera.set_detectCube(True)
        self.cubeTracker = CubeTracker()
        self.camera.set_cubeTracker(self.cubeTracker)
        
        found = self.cubeTracker.detect_cubes(self.camera.GridFrame)
        off = 0
        for cube in found:
            if not self.camera.is_in_center_square(cube.position):
                x = cube.position[0]
                y = cube.position[1]
                H_inv = self.camera.get_invHomography()
                pt_coords = np.dot(H_inv, np.array([x, y, 1.0]))
                pt_coords /= pt_coords[2]
                # pt_coords.flatten()
                [ix, iy, iz] = pt_coords
                depth = self.camera.DepthFrameRaw.copy()
                z = depth[int(iy)][int(ix)]

                shit = self.camera.GridFrame.copy()
            
                                
                wx, wy, wz = Perception.uv_to_world(ix, iy, z, 
                        invE = self.camera.get_invExtrinsic(), 
                        invK = self.camera.get_invIntrinsic())
                
                
                if self.camera.plane_coeffs:
                    adjust = self.camera.z_values[int(iy)][int(ix)]
                    wz -= adjust
                    wz += 3
                
                t, a = IK_geometric([wx, wy, wz + 40, np.pi/2])   
                self.fluid_move(a)
            
                a = IK_geometric([wx, wy, wz-14, np.pi/2])[1]
                # self.fluid_move(a)
                self.fluid_move(a)
                self.rxarm.gripper.grasp()
                # self.rxarm.initialize()
                self.to_safe()
                a = IK_geometric([0, 150, 10 + off, np.pi/2])[1]
                self.fluid_move(a)
                self.rxarm.gripper.release()
     
    def ball(self, msg):
        try:
            # [x, y, z] = Ball.find(self.camera.GridFrame)
            self.rxarm.gripper.release()
            self.dont_break_me()
            circles = Ball.find(self.camera.GridFrame)
            for circle in circles:
                x, y, r = np.array(circle[0]).flatten()
            
                H_inv = self.camera.get_invHomography()
                pt_coords = np.dot(H_inv, np.array([x, y, 1.0]))
                pt_coords /= pt_coords[2]
                # pt_coords.flatten()
                [ix, iy, iz] = pt_coords
                depth = self.camera.DepthFrameRaw.copy()
                z = depth[int(iy)][int(ix)]

            print(circles)
            shit = self.camera.GridFrame.copy()
            for circle in circles:
                x, y, r = np.array(circle[0]).flatten().astype(np.int16)
            # Draw the circle on the original image
                cv2.circle(shit, (x, y), r, (0, 255, 0), 4)
                cv2.circle(shit, (x, y), 2, (0, 0, 255), 3)
            cv2.imwrite("shit.png", shit)
        
            if len(circles) == 0:
                return    
                              
            wx, wy, wz = Perception.uv_to_world(ix, iy, z, 
                    invE = self.camera.get_invExtrinsic(), 
                    invK = self.camera.get_invIntrinsic())
            
            
            if self.camera.plane_coeffs:
                adjust = self.camera.z_values[int(iy)][int(ix)]
                wz -= adjust
                wz += 3
            print("FOUND BALL at:  ", [wx, wy, wz])
            self.rxarm.enable_torque()
            self.to_safe()
            t, a = IK_geometric([wx, wy, wz + 40, np.pi/2])   
            self.fluid_move(a)
            print("going to ball")
            a = IK_geometric([wx, wy, wz-14, np.pi/2])[1]
            # self.fluid_move(a)
            self.fluid_move(a)
            self.rxarm.gripper.grasp()
            # self.rxarm.initialize()
            self.to_safe()
            
            a = IK_geometric([375, 175, 350, 0])[1] 
            self.fluid_move(a)
            a = IK_geometric([356, 178, 210, np.pi/4])[1] 
            self.fluid_move(a)
            time.sleep(1.2)
            self.rxarm.gripper.release()
            a = IK_geometric([375, 175, 300, 0])[1] 
            self.fluid_move(a)
            
            
            a = IK_geometric([355, 100, 300, 0], force_t5=np.pi/2)[1] 
            self.fluid_move(a)
            a = IK_geometric([365, 100, 40, np.pi/4], force_t5=np.pi/2 )[1] 
            self.rxarm.set_positions(a)
            time.sleep(3)
            self.rxarm.set_moving_time(0.1)
            self.rxarm.set_accel_time(0.1)
            a = IK_geometric([375, 0, 40, np.pi/4], force_t5=np.pi/2)[1] 
            self.rxarm.set_positions(a)
            time.sleep(1)
            self.rxarm.set_moving_time(1)
            self.rxarm.set_accel_time(0.5)
            self.fluid_move(a)
            # self.dont_break_me()
            a = IK_geometric([350, 100, 300, np.pi/4], force_t5=np.pi/2)[1] 
            self.fluid_move(a)
            
            self.to_safe()
            
            
            
            
        except Exception as e:
            print(e)  
          
        finally:
            self.next_state = "ball"
        
        pass
         
    def replay(self):
        self.current_state = "replay"
        self.status_message = "State: Replay - Replaying recorded motion"
        for waypoint in self.demo_waypoints:
            curpoint = self.rxarm.get_positions()
            diff = 2 * np.max(np.abs(curpoint - waypoint))
            self.rxarm.set_positions(waypoint)
            self.rxarm.set_moving_time(diff)
            self.rxarm.set_accel_time(diff/4)
            time.sleep(diff)
            print("waypoint: ", waypoint)
            print("diff: ", diff)
            
        self.next_state = "idle"

    def ik_checkpoint3(self):
        M = self.rxarm.M
        Slist = self.rxarm.Slist
        
        desired_pose = self.rxarm.get_ee_pose()
        
        # print(desired_pose)
        
        thetas = kinematics.IK_geometric(M, Slist, desired_pose, returnAll=True)
        
        # Change to degrees
        print(thetas * 180/np.pi)
        
        self.next_state = "idle"

    def click_drop_open(self):
        # self.statusMessage = 'State: Click to pick up a block'
        self.current_state = 'click_pick'

        if self.camera.new_click:
            # print('Pick Click Done')
            self.click_drop_move()

    def click_drop_close(self):
        # self.statusMessage = 'State: Click to drop a block'
        
        self.current_state = 'click_place'

        if self.camera.new_click:
            self.rxarm.set_positions(np.array([0, 0, 0, 0, 0]))
            self.click_drop_move()

    def click_drop_move(self):
        """
        Moves the arm based on the current state: pick or place.
        """
        # world_pos1 = self.camera.uv2World(*self.camera.last_click)
        if self.mouse_world is not None:
            world_pos = self.mouse_world[:3].squeeze()
        # print(world_pos1, world_pos)
        self.camera.new_click = False
        
        delay_time = 2

        if self.current_state == 'click_pick':
            print('Pick Clicked')

            pickPos = self.mouse_world[:3].squeeze()
            # print(pickPos)
            
            pose = np.array([pickPos[0], pickPos[1], pickPos[2], np.pi/2])
            toppose = np.array([pickPos[0], pickPos[1], pickPos[2] + 90, np.pi/2])
            gripper = 'none'
            self.rxarm.move_to_pose(toppose, gripper)
            time.sleep(delay_time)
            
            gripper = 'grip'
            self.rxarm.move_to_pose(pose, gripper)
            time.sleep(delay_time)
            
            
            gripper = 'none'
            self.rxarm.move_to_pose(toppose, gripper)
            time.sleep(delay_time)
            
            self.next_state = 'click_place'
        else:
            print('Place Clicked')
            
            pickPos = self.mouse_world[:3].squeeze()
            
            pose = np.array([pickPos[0], pickPos[1], pickPos[2]+20, np.pi/2])
            toppose = np.array([pickPos[0], pickPos[1], pickPos[2] + 70, np.pi/2])
            gripper = 'none'
            self.rxarm.move_to_pose(toppose, gripper)
            time.sleep(delay_time)

            gripper = 'release'
            self.rxarm.move_to_pose(pose, gripper)
            time.sleep(delay_time)
            
            self.rxarm.move_to_pose(toppose, gripper)
            time.sleep(delay_time)

            self.next_state = 'click_pick'

    def to_safe(self):
        a = IK_geometric([0, 150, 200, np.pi/2])[1] 
        self.fluid_move(a)
        # a = IK_geometric([0, 50, 200, np.pi/4])[1] 
        # self.fluid_move(a)
        
    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"
        self.rxarm.enable_torque()
        self.dont_break_me()
        self.to_safe()
        
        """TODO Perform camera calibration routine here"""
        truth_source = AprilTag.FULL_A_LOC
        n = 8
        if len(self.camera.tag_detections.detections) < 8:
            truth_source = AprilTag.TRUE_A_LOC
            print("using minimum config of apriltag")
            n = 4
            
        # , self.camera.DepthFrameRaw[int(det.centre.y)][int(det.centre.x)]    
        centers = np.array([[float(det.centre.x), float(det.centre.y)]
                            for det in self.camera.tag_detections.detections]).astype(np.float32)[:n]
        truth = np.array(truth_source).astype(np.float32)[:n]
        # print("centers: ", centers)
        # print("truth: ", truth_source)
        extrinsic = Perception.recover_homogenous_transform_pnp(world_points=truth, image_points=centers)
        self.camera.set_extrinsic(extrinsic)
        # 
        
        # cal_image = self.camera.VideoFrame.copy()
        
        H = cv2.findHomography(centers[:4], DaWorld.SCALED_TAGS)[0]
        self.camera.set_homography(H)
        self.get_plane()
        self.camera.generate_plane_depth_array()
        # self.adjust_depth_image()
        self.next_state = "idle"
        # warped_rgb, warped_depth = Perception.transform_images(self.VideoFrame.copy(), H=H, depth_img = self.camera.DepthFrameRaw)

    def ransac_plane_fitting(self, world_points, iterations=100, threshold=0.1):
        best_inliers = []
        best_plane = None

        for _ in range(iterations):
            # Randomly select 3 points to define a plane
            sample_indices = np.random.choice(world_points.shape[0], 3, replace=False)
            sample_points = world_points[sample_indices]

            # Fit a plane to the 3 points
            centroid = sample_points.mean(axis=0)
            _, _, vh = np.linalg.svd(sample_points - centroid)
            normal = vh[-1]

            # Plane equation coefficients (ax + by + cz + d = 0)
            a, b, c = normal
            d = -np.dot(normal, centroid)

            # Calculate distances of all points to the plane
            distances = np.abs((world_points @ normal + d) / np.linalg.norm(normal))
            
            # Find inliers (points within the threshold distance)
            inliers = world_points[distances < threshold]
            
            # Update the best model if more inliers are found
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (a, b, c, d)

        return best_plane

    def get_plane(self):
        print("Starting adjust_depth with RANSAC")
        world_points = []
        for det in self.camera.tag_detections.detections[:4]:
            u, v = det.centre.x, det.centre.y        
            z = self.camera.DepthFrameRaw[int(v)][int(u)]
            wx, wy, wz = Perception.uv_to_world(u, v, z, 
                                                invE=self.camera.get_invExtrinsic(),
                                                invK=self.camera.get_invIntrinsic())
            print("looking at:  ",wx, wy, wz)
            world_points.append([wx, wy, wz])

        world_points = np.array(world_points)
        if world_points.shape[0] < 3:
            print("Not enough valid points to fit a plane.")
            return

        # Apply RANSAC to fit the best plane
        a, b, c, d = self.ransac_plane_fitting(world_points)
        self.camera.plane_coeffs = (a, b, c, d)
        
        print(f"RANSAC Plane coefficients: a={a}, b={b}, c={c}, d={d}")         
        
    

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)