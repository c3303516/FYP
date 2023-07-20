#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

###
# * DESCRIPTION OF CURRENT EXAMPLE:
# ===============================
# This example works as a simili-haptic demo.
#     
# The last actuator, the small one holding the interconnect, acts as a torque sensing device commanding the first actuator.
# The first actuator, the big one on the base, is controlled in torque and its position is sent as a command to the last one.
# 
# The script can be launched through command line with python3: python torqueControl_example.py
# The PC should be connected through ethernet with the arm. Default IP address 192.168.1.10 is used as arm address.
# 
# 1- Connection with the base:
#     1- A TCP session is started on port 10000 for most API calls. Refresh is at 25ms on this port.
#     2- A UDP session is started on port 10001 for BaseCyclic calls. Refresh is at 1ms on this port only.
# 2- Initialization
#     1- First frame is built based on arm feedback to ensure continuity
#     2- First actuator torque command is set as well
#     3- Base is set in low-level servoing
#     4- First frame is sent
#     3- First actuator is switched to torque mode
# 3- Cyclic thread is running at 1ms
#     1- Torque command to first actuator is set to a multiple of last actuator torque measure minus its initial value to
#        avoid an initial offset error
#     2- Position command to last actuator equals first actuator position minus initial delta
#     
# 4- On keyboard interrupt, example stops
#     1- Cyclic thread is stopped
#     2- First actuator is set back to position control
#     3- Base is set in single level servoing (default)
###

import sys
import os

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.ActuatorCyclicClientRpc import ActuatorCyclicClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import Session_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.RouterClient import RouterClientSendOptions

import time
import sys
import threading

#import custom scripts
from sinusoid import sinusoid, sinusoid_instant
import numpy as jnp
from params import robotParams
from numpy import pi,sin,cos,linalg
from gravcomp import gq
import csv



class TorqueExample:
    def __init__(self, router, router_real_time):

        # Maximum allowed waiting time during actions (in seconds)
        self.ACTION_TIMEOUT_DURATION = 20

        self.torque_amplification = 2.0  # Torque measure on last actuator is sent as a command to first actuator

        # Create required services
        device_manager = DeviceManagerClient(router)
        
        self.actuator_config = ActuatorConfigClient(router)
        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router_real_time)

        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()
        self.base_custom_data = BaseCyclic_pb2.CustomData()

        # Detect all devices
        device_handles = device_manager.ReadAllDevices()
        self.actuator_count = self.base.GetActuatorCount().count

        # Only actuators are relevant for this example
        for handle in device_handles.device_handle:
            if handle.device_type == Common_pb2.BIG_ACTUATOR or handle.device_type == Common_pb2.SMALL_ACTUATOR:
                self.base_command.actuators.add()
                self.base_feedback.actuators.add()

        # Change send option to reduce max timeout at 3ms
        self.sendOption = RouterClientSendOptions()
        self.sendOption.andForget = False
        self.sendOption.delay_ms = 0
        self.sendOption.timeout_ms = 3

        self.cyclic_t_end = 30  #Total duration of the thread in seconds. 0 means infinite.
        self.cyclic_thread = {}

        self.kill_the_thread = False
        self.already_stopped = False
        self.cyclic_running = False

    # Create closure to set an event after an END or an ABORT
    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            print("EVENT : " + \
                Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check
    

    def set_initial_position(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        print("Starting angular action movement ...")
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""
        
        # Place arm straight up
        for joint_id in range(self.actuator_count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = 0

        #set second joint to 45deg
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = 1        #as actuator count will max at 6
        joint_angle.value = 45

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        
        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(self.ACTION_TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Angular movement completed")
        else:
            print("Timeout on action notification wait")
        return finished
    
        return True

    def MoveToHomePosition(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
    
        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Zero":     #for candlestick
                action_handle = action.handle
                
            # if action.name == "Home":
                # action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)

        print("Waiting for movement to finish ...")
        finished = e.wait(self.ACTION_TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            time.sleep(1.)
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished

        return True
    
    def InitStorage(self, sampling_time_cyclic, t_end):
        #define storage for the q and velocity for processing
        i = self.actuator_count
        self.t = jnp.arange(0,t_end,sampling_time_cyclic)
        l = jnp.size(self.t)
        self.q_storage = jnp.zeros((i,l))
        self.vel_storage = jnp.zeros((i,l))
        self.userControl = jnp.zeros((3,l))
        self.controlHist = jnp.zeros((3,l))
        self.timeStore = jnp.zeros(l)

        
        trucated_t = jnp.arange(0,(t_end-2.),sampling_time_cyclic)      #this is used as a safety. Last 2 seconds will have 0 torques

        #also define the control actions that we're gonna take.

        u1 = sinusoid(self.t,0.,0.1,0.)     #t,origin,freq,amp)         #need a way to stabilise the sinusoid wrt. time.
        u2 = sinusoid(self.t,0.,0.5,0.)     #t,origin,freq,amp)
        u3 = sinusoid(self.t,0.,0.5,3.)     #t,origin,freq,amp)
        l_short = jnp.size(trucated_t)

        print('sampling time', sampling_time_cyclic)
        print('l',l,l_short)
        
        # for k in range(l):
        #     v1 = u1[k]
        #     v2 = u2[k]
        #     v3 = u3[k]
        #     if (k>l_short):
        #         v1 = 0
        #         v2 = 0
        #         v3 = 0
                
        #     self.userControl[:,[k]] = jnp.array([[v1],[v2],[v3]])


        return

    def InitCyclic(self, sampling_time_cyclic, t_end, print_stats):

        if self.cyclic_running:
            return True

        # Move to Home position first
        if not self.MoveToHomePosition():
            return False

        # Move to initial conditions
        # if not self.set_initial_position():
        #     return False

        print("Init Cyclic")
        sys.stdout.flush()

        base_feedback = self.SendCallWithRetry(self.base_cyclic.RefreshFeedback, 3)
        if base_feedback:
            self.base_feedback = base_feedback

            # Init command frame
            for x in range(self.actuator_count):
                self.base_command.actuators[x].flags = 1  # servoing
                self.base_command.actuators[x].position = self.base_feedback.actuators[x].position
            # 2nd actuator is going to be controlled in torque
            # To ensure continuity, torque command is set to opp measure to hold still
            self.base_command.actuators[1].torque_joint = -self.base_feedback.actuators[3].torque
                
            # 4th actuator is going to be controlled in torque
            # To ensure continuity, torque command is set to opp measure to hold still
            self.base_command.actuators[3].torque_joint = -self.base_feedback.actuators[3].torque

            # Sixth actuator is going to be controlled in torque
            # To ensure continuity, torque command is set to opp measure to hold still
            self.base_command.actuators[5].torque_joint = -self.base_feedback.actuators[5].torque

            # Set arm in LOW_LEVEL_SERVOING
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            # Send first frame
            self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)

            # Set second actuator in torque mode now that the command is equal to measure
            control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
            control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
            device_id = 2  # first actuator as id = 1, last is id = 7
            
            self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id)

            # Set fourth actuator in torque mode now that the command is equal to measure
            control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
            control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
            device_id = 4  # first actuator as id = 1, last is id = 7

            self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id)


            # Set sixth actuator in torque mode now that the command is equal to measure
            control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
            control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('TORQUE')
            device_id = 6  # first actuator as id = 1, last is id = 7

            self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id)

            # Init cyclic thread
            self.cyclic_t_end = t_end
            self.cyclic_thread = threading.Thread(target=self.RunCyclic, args=(sampling_time_cyclic, print_stats))
            self.cyclic_thread.daemon = True
            self.cyclic_thread.start()
            return True

        else:
            print("InitCyclic: failed to communicate")
            return False

    def RunCyclic(self, t_sample, print_stats):
        self.cyclic_running = True
        print("Run Cyclic")
        sys.stdout.flush()
        cyclic_count = 0  # Counts refresh
        stats_count = 0  # Counts stats prints
        failed_cyclic_count = 0  # Count communication timeouts

        controlActions_temp = self.userControl
        q_storage_temp = self.q_storage
        vel_storage_temp = self.vel_storage
        controlHist_temp = self.controlHist
        timetemp = self.timeStore

        #define sinusoide amplitudes, freqs
        amp1 = 0.
        freq1 = 0.2
        amp2 = 0.
        freq2 = 0.75
        amp3 = 0.
        freq3 = 0.75


        
        q_bold1 = (pi/180.)*self.base_feedback.actuators[0].position
        q_bold2 = (pi/180.)*self.base_feedback.actuators[2].position
        q_bold3 = (pi/180.)*self.base_feedback.actuators[4].position
        q_bold4 = (pi/180.)*self.base_feedback.actuators[6].position

        # Initial delta between first and last actuator
        # init_delta_position = self.base_feedback.actuators[0].position - self.base_feedback.actuators[self.actuator_count - 1].position

        # Initial first and last actuator torques; avoids unexpected movement due to torque offsets
        init_fourth_torque = self.base_feedback.actuators[3].torque
        init_second_torque = self.base_feedback.actuators[1].torque  
        init_sixth_torque = self.base_feedback.actuators[5].torque

        print('Initial Torque',init_second_torque,init_fourth_torque,init_sixth_torque)

        t_now = time.time()
        t_cyclic = t_now  # cyclic time
        t_stats = t_now  # print  time
        t_init = t_now  # init   time
        # tic = 0
        counter = 0
        print("Running torque control example for {} seconds".format(self.cyclic_t_end))

        while not self.kill_the_thread:
            t_now = time.time()

            # Cyclic Refresh
            if (t_now - t_cyclic) >= t_sample:
                t_cyclic = t_now
                
                # Position command to first actuator is set to measured one to avoid following error to trigger
                # Bonus: When doing this instead of disabling the following error, if communication is lost and first
                #        actuator continue to move under torque command, resulting position error with command will
                #        trigger a following error and switch back the actuator in position command to hold its position
                
                self.base_command.actuators[1].position = self.base_feedback.actuators[1].position
                self.base_command.actuators[3].position = self.base_feedback.actuators[3].position
                self.base_command.actuators[5].position = self.base_feedback.actuators[5].position

                q1 = (pi/180.)*self.base_feedback.actuators[1].position
                q2 = (pi/180.)*self.base_feedback.actuators[3].position
                q3 = (pi/180.)*self.base_feedback.actuators[5].position
                #Constant values. Will measure feedback to ensure model is always correct
                # print('q1',q1)
                q = jnp.array([[q_bold1],
                               [q1],
                               [q_bold2],
                               [q2],
                               [q_bold3],
                               [q3],
                               [q_bold4]])

                q1dot = self.base_feedback.actuators[1].velocity
                q2dot = self.base_feedback.actuators[3].velocity
                q3dot = self.base_feedback.actuators[5].velocity

                # print('q2dot', q2dot)

                t_elapsed = t_now - t_init
                #get control actions
                grav = gq(q)
                timetemp[counter] = t_elapsed  #time since script started

                if (t_elapsed < (self.cyclic_t_end - 2)):
                    v1 = sinusoid_instant(t_elapsed,0.,freq1,amp1)
                    v2 = sinusoid_instant(t_elapsed,0.,freq2,amp2)
                    v3 = sinusoid_instant(t_elapsed,0.,freq3,amp3)
                    # v3 = controlActions_temp[2,counter]
                else:           #set torques to 0 to slow down
                    print('slowing')
                    v1 = 0
                    v2 = 0
                    v3 = 0

                gq1 = grav[1,0]
                gq2 = grav[3,0]
                gq3 = grav[5,0]

                u1 = gq1 #+ v1
                u2 = gq2 #+ v2
                u3 = gq3 #+ v3
                print('u1',u1)
                self.base_command.actuators[1].torque_joint = u1
                # Grav comp is sent to fourth actuator
                self.base_command.actuators[3].torque_joint = u2
                # Grav comp is sent to sixth actuator
                self.base_command.actuators[5].torque_joint = u3

                # self.base_command.actuators[5].torque_joint = 0
                # self.base_command.actuators[5].torque_joint = -self.base_feedback.actuators[5].torque
                # Incrementing identifier ensure actuators can reject out of time frames
                self.base_command.frame_id += 1
                if self.base_command.frame_id > 65535:
                    self.base_command.frame_id = 0
                for i in range(self.actuator_count):
                    self.base_command.actuators[i].command_id = self.base_command.frame_id

                # Frame is sent
                try:
                    self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
                except:
                    failed_cyclic_count = failed_cyclic_count + 1
                cyclic_count = cyclic_count + 1

                #store
                controlHist_temp[:,[counter]] = jnp.array([[u1],[u2],[u3]])

                q_storage_temp[:,[counter]] = q
                vel_storage_temp[:,[counter]] = jnp.array([
                               [q1dot],
                               [q2dot],
                               [q3dot],
                               ])

                
                counter = counter + 1       #index

            # Stats Print
            if print_stats and ((t_now - t_stats) > 1):
                t_stats = t_now
                stats_count = stats_count + 1
                
                cyclic_count = 0
                failed_cyclic_count = 0
                sys.stdout.flush()

            if self.cyclic_t_end != 0 and (t_now - t_init > self.cyclic_t_end):
                print("Cyclic Finished")
                sys.stdout.flush()
                #get data where it needs to be
                self.userControl = controlActions_temp
                self.q_storage = q_storage_temp
                self.vel_storage = (pi/180.)*vel_storage_temp
                self.controlHist = controlHist_temp
                self.timeStore = timetemp
                # print(vel_storage_temp)
                self.a1 = amp1
                self.a2 = amp2
                self.a3 = amp3
                self.f1 = freq1
                self.f2 = freq2
                self.f3 = freq3

                break
        self.cyclic_running = False
        return True

    def StopCyclic(self):
        print ("Stopping the cyclic and putting the arm back in position mode...")
        if self.already_stopped:
            return

        # Kill the  thread first
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()
        
        # Set first actuator back in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        
        device_id = 2  # first actuator has id = 1
        self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id)
        device_id = 4  # first actuator has id = 1
        self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id)
        device_id = 6  # first actuator has id = 1
        self.SendCallWithRetry(self.actuator_config.SetControlMode, 3, control_mode_message, device_id)
        
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        self.cyclic_t_end = 0.1

        self.already_stopped = True
        
        print('Clean Exit')

    def SaveData(self):
        print('Saving Data')
        l = jnp.size(self.t)
        
        details = ['Saved Data from Physical Implementation! This file has double grav comp so the robot arm swings to a vertical position']
        values = ['Amp/Freqs: v1',self.a1,self.f1,'v2',self.a2,self.f2,'v3',self.a3,self.f3]
        header = ['Time', 'State History']
        with open('C:/Users/Mark/Documents/FYP/python_api/api_python/examples/108-Gen3_torque_control/data/freeswing_inverted_v1', 'w', newline='') as f:

            writer = csv.writer(f)
            # writer.writerow(simtype)
            writer.writerow(details)
            writer.writerow(values)
            writer.writerow(header)

            # writer.writerow(['Time', t])
            for i in range(l):
                timestamp = self.timeStore[i]           #time
                q1 = self.q_storage[0,i]               #postion
                q2 = self.q_storage[1,i]
                q3 = self.q_storage[2,i]
                q4 = self.q_storage[3,i]
                q5 = self.q_storage[4,i]
                q6 = self.q_storage[5,i]
                q7 = self.q_storage[6,i]
                qdot1 = self.vel_storage[0,i]               #momentum (transformed)
                qdot2 = self.vel_storage[1,i]
                qdot3 = self.vel_storage[2,i]
                v1 = self.controlHist[0,i]       #control values
                v2 = self.controlHist[1,i] 
                v3 = self.controlHist[2,i] 
                data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,q4,q5,q6,q7,qdot1,qdot2,qdot3, v1,v2,v3]
                
                writer.writerow(data)


        print('Data Saved!')
        return

    @staticmethod
    def SendCallWithRetry(call, retry,  *args):
        i = 0
        arg_out = []
        while i < retry:
            try:
                arg_out = call(*args)
                break
            except:
                i = i + 1
                continue
        if i == retry:
            print("Failed to communicate")
        return arg_out
    
class self:
    def __init___(self):
        return 'initialised'

def main():
    # Import the utilities helper module
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyclic_time", type=float, help="delay, in seconds, between cylic control call", default=0.001)   #was initially 0.001
    parser.add_argument("--duration", type=int, help="example duration, in seconds (0 means infinite)", default=30)
    parser.add_argument("--print_stats", default=True, help="print stats in command line or not (0 to disable)", type=lambda x: (str(x).lower() not in ['false', '0', 'no']))
    args = utilities.parseConnectionArguments(parser)



    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:

            example = TorqueExample(router, router_real_time)
            args.cyclic_time = 0.005
            example.InitStorage(args.cyclic_time, args.duration)
            success = example.InitCyclic(args.cyclic_time, args.duration, args.print_stats)

            if success:

                while example.cyclic_running:
                    try:
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        break
            
                example.StopCyclic()

                example.SaveData()

            return 0 if success else 1


if __name__ == "__main__":
    exit(main())
