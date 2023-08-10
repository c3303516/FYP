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
import jax
from jax.config import config
config.update("jax_enable_x64", True)
from sinusoid import sinusoid, sinusoid_instant
import jax.numpy as jnp
from params import robotParams
from jax.numpy import pi,sin,cos,linalg
from gravcomp import gq
import csv
from jax import grad, jacobian, jacfwd
from functools import partial
from scipy.linalg import solve_continuous_lyapunov
from jax.scipy.linalg import sqrtm
from scipy.optimize import least_squares
from massMatrix_holonomic import massMatrix_holonomic
import trajectories

################ HOMOGENEOUS TRANSFORMS ###############################

def skew(u):
    ans = jnp.block([[0., -u[2], u[1]],
                    [u[2], 0., -u[0]],
                    [-u[1], u[0], 0.]])
    return ans

def hatSE3(x):
    A = skew(x[3:5])
    return A


def rotx(mu):
    A = jnp.block([[1., 0., 0., 0.],
                   [0., jnp.cos(mu), -jnp.sin(mu), 0.],
                   [0., jnp.sin(mu), jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def roty(mu):
    A = jnp.block([[jnp.cos(mu), 0., jnp.sin(mu), 0.],
                   [0., 1., 0., 0.],
                   [-jnp.sin(mu), 0., jnp.cos(mu), 0.],
                   [0., 0., 0., 1.]])
    return A

def rotz(mu):
    A = jnp.block([[jnp.cos(mu), -jnp.sin(mu), 0., 0.],
                   [jnp.sin(mu), jnp.cos(mu), 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

    return A

def rotx_small(mu):
    A = jnp.block([[1., 0., 0.],
                [0., mu, -mu],
                [0., mu, mu],
                ])
    return A

def roty_small(mu):
    A = jnp.block([[mu, 0., mu],
                   [0., 1., 0.],
                   [-mu, 0., mu]
                   ])
    return A

def rotz_small(mu):
    A = jnp.block([[mu, -mu, 0.],
                   [mu, mu, 0.],
                   [0., 0., 1.]
                   ])
    return A

def tranx(mu):
    A = jnp.array([[1., 0., 0., mu],
                   [0., 1., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    return A

def trany(mu):
    A = jnp.array([[1., 0., 0., 0.],
                   [0., 1., 0., mu],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    return A

def tranz(mu):
    A = jnp.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.],
                   [0., 0., 1., mu],
                   [0., 0., 0., 1.]])
    return A

def hatSE3(x):
    S = jnp.block([
        skew(x[3:6]), x[0:3], jnp.zeros((4,1))
    ])
    return S

####### MASS MATRIX #############

def massMatrix_continuous(q_hat,qconstants):
   
    dFcdq = jnp.array([
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
    ])
    q_bold = dFcdq@q_hat + qconstants.at[0].get()
    # print(q_bold)

    q1 = q_bold.at[0,0].get()
    q3 = q_bold.at[1,0].get()
    q5 = q_bold.at[2,0].get()
    q7 = q_bold.at[3,0].get()

    q2 = q_hat.at[0,0].get()
    q4 = q_hat.at[1,0].get()
    q6 = q_hat.at[2,0].get()
    # print('q1',q1)

    A01 = tranz(s.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-s.d1)@trany(-s.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-s.d2)@trany(-s.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-s.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-s.l7)@rotz(q7)
    A7E = rotx(pi)@tranz(s.l8)    
    AEG = tranz(s.lGripper)

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    A0E = A07@A7E
    A0G = A0E@AEG

    # Geometric Jacobians

    c1 = s.c1
    c1x = c1.at[0].get()
    c1y = c1.at[1].get()
    c1z = c1.at[2].get()
    c2 = s.c2
    c2x = c2.at[0].get()
    c2y = c2.at[1].get()
    c2z = c2.at[2].get()
    c3 = s.c3
    c3x = c3.at[0].get()
    c3y = c3.at[1].get()
    c3z = c3.at[2].get()
    c4 = s.c4
    c4x = c4.at[0].get()
    c4y = c4.at[1].get()
    c4z = c4.at[2].get()
    c5 = s.c5
    c5x = c5.at[0].get()
    c5y = c5.at[1].get()
    c5z = c5.at[2].get()
    c6 = s.c6
    c6x = c6.at[0].get()
    c6y = c6.at[1].get()
    c6z = c6.at[2].get()
    c7 = s.c7
    c7x = c7.at[0].get()
    c7y = c7.at[1].get()
    c7z = c7.at[2].get()
    c8 = s.c8
    c8x = c8.at[0].get()
    c8y = c8.at[1].get()
    c8z = c8.at[2].get()
    cG = s.cGripper
    cGz = cG.at[2].get()



    A0c1 = tranx(c1x)@trany(c1y)@tranz(c1z)
    A0c2 = A01@tranx(c2x)@trany(c2y)@tranz(c2z)
    A0c3 = A02@tranx(c3x)@trany(c3y)@tranz(c3z)  
    A0c4 = A03@tranx(c4x)@trany(c4y)@tranz(c4z)
    A0c5 = A04@tranx(c5x)@trany(c5y)@tranz(c5z)
    A0c6 = A05@tranx(c6x)@trany(c6y)@tranz(c6z)
    A0c7 = A06@tranx(c7x)@trany(c7y)@tranz(c7z)
    A0c8 = A07@tranx(c8x)@trany(c8y)@tranz(c8z)
    A0cG = A0E@tranz(cGz)



            # Geometric Jacobians
    R01 = A01[0:3,0:3]     #rotation matrices
    R12 = A12[0:3,0:3]
    R23 = A23[0:3,0:3]
    R34 = A34[0:3,0:3]
    R45 = A45[0:3,0:3]
    R56 = A56[0:3,0:3]
    R67 = A67[0:3,0:3]
    R7E = A7E[0:3,0:3]
    REG = AEG[0:3,0:3]

    r100   = A01[0:3,[3]]
    r200   = A02[0:3,[3]]
    r300   = A03[0:3,[3]]
    r400   = A04[0:3,[3]]
    r500   = A05[0:3,[3]]
    r600   = A06[0:3,[3]]
    r700   = A07[0:3,[3]]
    r800   = A0E[0:3,[3]]
    rG00   = A0G[0:3,[3]]


    rc100   = A0c1[0:3,[3]]
    rc200   = A0c2[0:3,[3]]
    rc300   = A0c3[0:3,[3]]
    rc400   = A0c4[0:3,[3]]
    rc500   = A0c5[0:3,[3]]
    rc600   = A0c6[0:3,[3]]
    rc700   = A0c7[0:3,[3]]
    rc800   = A0c8[0:3,[3]]
    rcG00   = A0cG[0:3,[3]]

    z00 = jnp.array([[0.], [0.], [1.]])
    z01 = R01@z00
    z02 = R01@R12@z00
    z03 = R01@R12@R23@z00
    z04 = R01@R12@R23@R34@z00
    z05 = R01@R12@R23@R34@R45@z00
    z06 = R01@R12@R23@R34@R45@R56@z00
    z07 = R01@R12@R23@R34@R45@R56@R67@z00
    z08 = R01@R12@R23@R34@R45@R56@R67@R7E@z00
    z0G = R01@R12@R23@R34@R45@R56@R67@R7E@REG@z00

    ske1 = skew(z01)
    ske2 = skew(z02)
    ske3 = skew(z03)
    ske4 = skew(z04)
    ske5 = skew(z05)
    ske6 = skew(z06)
    ske7 = skew(z07)
    ske8 = skew(z08)
    skeG = skew(z0G)


    Jc2   = jnp.block([
        [ske1@(rc200-r100),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],   #jnp.zeros((3,1))],
        [z01,                 jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        ])
    Jc3   = jnp.block([
        [ske1@(rc300-r100),  ske2@(rc300-r200),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],  # jnp.zeros((3,1))],
        [z01,                z02              ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        ])
    Jc4   = jnp.block([
        [ske1@(rc400-r100),  ske2@(rc400-r200),  ske3@(rc400-r300),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))], #  jnp.zeros((3,1))],
        [z01,                z02,                 z03             ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]   #jnp.zeros((3,1))]
        ])
    Jc5   = jnp.block([
        [ske1@(rc500-r100),  ske2@(rc500-r200),  ske3@(rc500-r300),  ske4@(rc500-r400),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],  # jnp.zeros((3,1))],
        [z01,                z02,                z03,                z04              ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        ])
    Jc6   = jnp.block([
        [ske1@(rc600-r100),  ske2@(rc600-r200),  ske3@(rc600-r300),  ske4@(rc600-r400),  ske5@(rc600-r500),   jnp.zeros((3,1)),   jnp.zeros((3,1))],   #jnp.zeros((3,1))],
        [z01,                z02,                z03,                z04,                z05              ,   jnp.zeros((3,1)),   jnp.zeros((3,1))] #  jnp.zeros((3,1))]
        ])
    Jc7   = jnp.block([
        [ske1@(rc700-r100),  ske2@(rc700-r200),  ske3@(rc700-r300),  ske4@(rc700-r400),  ske5@(rc700-r500),  ske6@(rc700-r600),   jnp.zeros((3,1))],  #jnp.zeros((3,1))],
        [z01,                z02              ,  z03,                z04,                z05,                z06              ,   jnp.zeros((3,1))]   #jnp.zeros((3,1))]
        ])
    Jc8   = jnp.block([
        [ske1@(rc800-r100),  ske2@(rc800-r200),  ske3@(rc800-r300),  ske4@(rc800-r400),  ske5@(rc800-r500),  ske6@(rc800-r600),  ske7@(rc800-r700)], #  jnp.zeros((3,1))],
        [z01,                z02,                z03              ,  z04,                z05,                z06,                z07              ]  #,   jnp.zeros((3,1))]
        ])

    # Mass Matrix
    R02 = A02[0:3,0:3]
    R03 = A03[0:3,0:3]
    R04 = A04[0:3,0:3]
    R05 = A05[0:3,0:3]
    R06 = A06[0:3,0:3]
    R07 = A07[0:3,0:3]
    R08 = A0E[0:3,0:3]
    R0G = A0G[0:3,0:3]

    I2 = s.I2
    I3 = s.I3
    I4 = s.I4
    I5 = s.I5
    I6 = s.I6
    I7 = s.I7
    I8 = s.I8
    IG = s.IG


    M2 = Jc2.T@jnp.block([
        [jnp.multiply(s.m2,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R02.T@I2@R02 ]
    ])@Jc2
    M3 = Jc3.T@jnp.block([
        [jnp.multiply(s.m3,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R03.T@I3@R03 ]
    ])@Jc3
    M4 = Jc4.T@jnp.block([
        [jnp.multiply(s.m4,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R04.T@I4@R04 ]
    ])@Jc4
    M5 = Jc5.T@jnp.block([
        [jnp.multiply(s.m5,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R05.T@I5@R05 ]
    ])@Jc5
    M6 = Jc6.T@jnp.block([
        [jnp.multiply(s.m6,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R06.T@I6@R06 ]
    ])@Jc6
    M7 = Jc7.T@jnp.block([
        [jnp.multiply(s.m7,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R07.T@I7@R07 ]
    ])@Jc7
    M8 = Jc8.T@jnp.block([
        [jnp.multiply(s.m8,jnp.eye(3,3)), jnp.zeros((3,3))],
        [jnp.zeros((3,3)),            R08.T@I8@R08 ]
    ])@Jc8

    Mq = M2 + M3 + M4 + M5 + M6 + M7 + M8# + MG

    return Mq
    

################## DMDQ FUNCTIONS ##########################
def MqPrime(q_hat,constants):

    Mq = massMatrix_continuous(q_hat,constants)
    A = jnp.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])
    
    Mq_hat = jnp.transpose(A)@Mq@A

    Mprime = jnp.zeros(Mq_hat.size)
    # print('sizeMq', jnp.shape(Mq_hat))
    # print('sizeMprime',jnp.shape(Mprime))
    m,n = jnp.shape(Mq_hat)
    for i in range(m):
        # print('i', i)
        Mprime = Mprime.at[n*i:n*(i+1)].set(Mq_hat.at[0:m,i].get())
        
    return Mprime

def TqiPrime(q_hat,constants):           #used in creating a Tqinv stack for dTinvdq calcs. #Legacy
    Mq = massMatrix_continuous(q_hat,constants)
    A = jnp.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])
    
    Mq_hat = jnp.transpose(A)@Mq@A
    Tqi = jnp.real(sqrtm(Mq_hat))

    Tiprime = jnp.zeros(Tqi.size)
    # print('sizeMq', jnp.shape(Mq_hat))
    # print('sizeMprime',jnp.shape(Mprime))
    m,n = jnp.shape(Tqi)
    for i in range(m):
        # print('i', i)
        Tiprime = Tiprime.at[n*i:n*(i+1)].set(Tqi.at[0:m,i].get())
        
    return Tiprime

@jax.jit
def unravel(dMdq_temp):         #This rearranges the square matrix that gets input after jacobian calculation
    # could probably generalise this for any array
    (m,n,l) = jnp.shape(dMdq_temp)
    dMdq1 = jnp.zeros((n,n))
    dMdq2 = jnp.zeros((n,n))
    dMdq3 = jnp.zeros((n,n))
    # print('dmdqshpae',jnp.shape(dMdq1))

    for i in range(n):
        dMdq1 = dMdq1.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),0,0].get())
        dMdq2 = dMdq2.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),1,0].get())
        dMdq3 = dMdq3.at[0:n,i].set(dMdq_temp.at[n*i:n*(i+1),2,0].get())

    return dMdq1, dMdq2, dMdq3 

# @partial(jax.jit, static_argnames=['Ti'])
@jax.jit
def dTi(Ti,dM):
    dTi = solve_continuous_lyapunov(Ti,dM)
    return dTi

@jax.jit
def Vq(q_hat, qconstants):
    #Function has to do FKM again to enable autograd to work
    dFcdq = jnp.array([
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
    ])
    q_bold = dFcdq@q_hat + qconstants.at[0].get()

    q1 = q_bold.at[0].get()
    q3 = q_bold.at[1].get()
    q5 = q_bold.at[2].get()
    q7 = q_bold.at[3].get()

    q2 = q_hat.at[0].get()
    q4 = q_hat.at[1].get()
    q6 = q_hat.at[2].get()

    A01 = tranz(s.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-s.d1)@trany(-s.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-s.d2)@trany(-s.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(s.d2)@tranz(-s.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-s.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-s.l7)@rotz(q7)

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    # A0E = A07@A7E
    # A0G = A0E@AEG

    c1 = s.c1
    c1x = c1.at[0].get()
    c1y = c1.at[1].get()
    c1z = c1.at[2].get()
    c2 = s.c2
    c2x = c2.at[0].get()
    c2y = c2.at[1].get()
    c2z = c2.at[2].get()
    c3 = s.c3
    c3x = c3.at[0].get()
    c3y = c3.at[1].get()
    c3z = c3.at[2].get()
    c4 = s.c4
    c4x = c4.at[0].get()
    c4y = c4.at[1].get()
    c4z = c4.at[2].get()
    c5 = s.c5
    c5x = c5.at[0].get()
    c5y = c5.at[1].get()
    c5z = c5.at[2].get()
    c6 = s.c6
    c6x = c6.at[0].get()
    c6y = c6.at[1].get()
    c6z = c6.at[2].get()
    c7 = s.c7
    c7x = c7.at[0].get()
    c7y = c7.at[1].get()
    c7z = c7.at[2].get()
    c8 = s.c8
    c8x = c8.at[0].get()
    c8y = c8.at[1].get()
    c8z = c8.at[2].get()
    cG = s.cGripper
    cGz = cG.at[2].get()

    A0c1 = tranx(c1x)@trany(c1y)@tranz(c1z)
    A0c2 = A01@tranx(c2x)@trany(c2y)@tranz(c2z)
    A0c3 = A02@tranx(c3x)@trany(c3y)@tranz(c3z)  
    A0c4 = A03@tranx(c4x)@trany(c4y)@tranz(c4z)
    A0c5 = A04@tranx(c5x)@trany(c5y)@tranz(c5z)
    A0c6 = A05@tranx(c6x)@trany(c6y)@tranz(c6z)
    A0c7 = A06@tranx(c7x)@trany(c7y)@tranz(c7z)
    A0c8 = A07@tranx(c8x)@trany(c8y)@tranz(c8z)
    rc100   = A0c1[0:3,3]
    rc200   = A0c2[0:3,3]
    rc300   = A0c3[0:3,3]
    rc400   = A0c4[0:3,3]
    rc500   = A0c5[0:3,3]
    rc600   = A0c6[0:3,3]
    rc700   = A0c7[0:3,3]
    rc800   = A0c8[0:3,3]

    g0 = jnp.array([[0.],[0.],[-s.g]])
    gprime = jnp.transpose(g0)

    V = -s.m1*gprime@rc100-s.m2*gprime@rc200-s.m3*gprime@rc300-s.m4*gprime@rc400-s.m5*gprime@rc500-s.m6*gprime@rc600 -s.m7*gprime@rc700 -s.m8*gprime@rc800 #-s.mGripper*gprime@rcG00
    # s.V = V
    return V.at[0].get()


def holonomicConstraint(q_hat, qconstants):
    dFcdq = jnp.array([
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
        [0.,0.,0.],
    ])
    q_bold = dFcdq@q_hat + qconstants
    q_bold = q_bold.at[0].get()         #remove the extra array
    return q_bold

############################# TRACKING PROBLEM ###################################

@partial(jax.jit, static_argnames=['struct'])
def errorIKM(q, q_last, xestar,struct):

    #q0 is start point. q is iteration to solve for xstar
    #xstar is currently just positon, no angles
    #EFFECTOR IKM
    q2 = q[0]      #make the full q vector
    q4 = q[1]
    q6 = q[2]
    # print('struct',struct)
    q = jnp.array([     #this is qhat
        [q2, q4, q6]
        ])

    q = jnp.transpose(q)
    qconstants = struct.constants
    q_bold = qconstants.at[:,0].get()     #constrained variabless
    # print('constants',q_bold)
    
    q1 = q_bold.at[0].get()
    q3 = q_bold.at[1].get()
    q5 = q_bold.at[2].get()
    q7 = q_bold.at[3].get()

    A01 = tranz(struct.l1)@rotx(pi)@rotz(q1)          
    A12 = rotx(pi/2)@tranz(-struct.d1)@trany(-struct.l2)@rotz(q2)
    A23 = rotx(-pi/2)@trany(struct.d2)@tranz(-struct.l3)@rotz(q3)
    A34 = rotx(pi/2)@tranz(-struct.d2)@trany(-struct.l4)@rotz(q4)
    A45 = rotx(-pi/2)@trany(struct.d2)@tranz(-struct.l5)@rotz(q5)
    A56 = rotx(pi/2)@trany(-struct.l6)@rotz(q6)
    A67 = rotx(-pi/2)@tranz(-struct.l7)@rotz(q7)
    A7E = rotx(pi)@tranz(struct.l8)    
    AEG = tranz(struct.lGripper)

    A02 = A01@A12
    A03 = A02@A23
    A04 = A03@A34
    A05 = A04@A45
    A06 = A05@A56
    A07 = A06@A67
    A0E = A07@A7E
    #end effector pose
    r0E0 = A0E[0:3,[3]]

    pose_weights = jnp.array([1.,1.])
    sqK = 1000*jnp.diag(pose_weights)                            #weights on x and z error
    e_pose = jnp.array([r0E0[0]-xestar[0] ,r0E0[2]-xestar[2]] )  #penalised error from xe to xestar

    e_q = q - q_last            #penalises movement from q0
    q_weights = jnp.array([1.,1.,1.])              #penalises specific joint movement. Currently set to use base joint more
    sqW = 1*jnp.diag(q_weights)

    sqM = jnp.block([
        [sqW,               jnp.zeros((3, 2))],
        [jnp.zeros((2,3)),  sqK              ]
    ])
#     s = effectorAnalyticalJacobian(q, param, s);      #this is here for future reference if I want to fix angles
#     JA = s.JA;
#     J = [sqW;
#          sqK*JA];

    e = sqM@jnp.block([[e_q],[e_pose]])         # this isn't quadratic?? FUCK
                       #coudl use jacobian and feed through. might decrease time
    # print(e)
    e = e[:,0]           
    return e

def solveIKM(traj,init_guess,params):
    #solves for displacement only
    guess = init_guess
    # print(jnp.shape(guess))
    m,n = jnp.shape(traj)
    # print('m,n', m,n)
    q_d = jnp.zeros((m,n))

    bound = ([s.qlower.at[1].get(),s.qlower.at[3].get(),s.qlower.at[5].get()],[s.qupper.at[1].get(),s.qupper.at[3].get(),s.qupper.at[5].get()])


    for i in range(n):
        # point = traj.at[0:3,i].get()
        point = traj[0:3,i]
        # errorIKM(q, q0, xestar,struct):       #function handle for ref
        q0 = jnp.transpose(jnp.array([guess]))
        # print(jnp.shape(q0))
        result = least_squares(errorIKM,guess,bounds=bound,args=(q0,point,s))        #starts guess at guess, and penalised movement from last q_d point
        guess = result.x
        q_d = q_d.at[:,i].set(guess)
        # print(q_d.at[:,i].get())

    return q_d

def diff_finite(q_d,t_span):
    #this function approxmates xdot from IKM solution q_d. The final momentum will always be zero, as the approxmition will shorten the array
    l = jnp.size(t_span)
    qdot = jnp.zeros(jnp.shape(q_d))        #end will be zero
    # print('qdotsize',jnp.shape(qdot))

    for i in range(l):
        qdot = qdot.at[:,[i]].set(q_d.at[:,[i+1]].get()- q_d.at[:,[i]].get())
    return qdot

############################ CONTROLLER #############################

@jax.jit
def control(x_err,Tq,Cq,Kp,Kd,alpha,Tqinv,dTqinvdq1,dTqinvdq2,dTqinvdq3,qddot,qddotdot,tau,p,p_d):
    #This control function provides the 'v' input detailed in the paper. This v is added to gravity compensation.
    q_tilde = x_err.at[0:3].get()
    p_tilde = x_err.at[3:6].get()

    D_hat = jnp.zeros((3,3))
    v_input = alpha*(Cq - D_hat - Kd)@Kp@(q_tilde + alpha*p_tilde) - Tq@Kp@(q_tilde + alpha*p_tilde) - Kd@p_tilde
    # print(v)
    dTiqdot_dq1 = dTqinvdq1@qddot               #dp_d/dq = dTinv/dq @ qd_dot
    dTiqdot_dq2 = dTqinvdq2@qddot
    dTiqdot_dq3 = dTqinvdq3@qddot

    dp_d_dq = jnp.block([dTiqdot_dq1,dTiqdot_dq2,dTiqdot_dq3])      #this is from product rule

    v = -(Cq - D_hat)@p_d + dp_d_dq@Tq@p + Tqinv@qddotdot + tau + v_input          #total control law from equaton 17 now here.
    
    return v

@jax.jit
def Cqpfunc(p,Tq,dTqinvdq_val):
    #Calculation of the Coriolis Damping matrix (Check if this is correct name)
    dTqinvdq1 = dTqinvdq_val.at[0].get()
    dTqinvdq2 = dTqinvdq_val.at[1].get()
    dTqinvdq3 = dTqinvdq_val.at[2].get()
    # print('p',p)
    dTqinv_phatdq1 = dTqinvdq1@p
    dTqinv_phatdq2 = dTqinvdq2@p
    dTqinv_phatdq3 = dTqinvdq3@p
    # print('dTqinv',dTqinv_phatdq1)
    temphat = jnp.block([dTqinv_phatdq1, dTqinv_phatdq2, dTqinv_phatdq3])
    temphatT = jnp.transpose(temphat)
    # print('temp',temp)
    Ctemp = temphatT - temphat
    # print(Ctemp)
    Cq = Tq@Ctemp@Tq

    return Cq




#### CLASS FOR CONTROL OF THE ROBOT
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
    
    # @partial(jax.jit, static_argnames=['q0'])
    def V(self, q0):
        q2 = q0.at[1,0].get()      #pass the joint variable to the specific joints 
        q4 = q0.at[3,0].get()
        q6 = q0.at[5,0].get()
        q1 = q0.at[0,0].get()              #pass out to joints. Values of holonomic constraint
        q3 = q0.at[2,0].get()
        q5 = q0.at[4,0].get()
        q7 = q0.at[6,0].get()


        l1 = 156.4e-3
        l2 = 128.4e-3
        l3 = 210.4e-3
        l4 = 210.4e-3
        l5 = 208.4e-3
        l6 = 105.9e-3
        l7 = 105.9e-3
        l8 = 61.5e-3

        d1 = 5.4e-3
        d2 = 6.4e-3


        g = 9.81
        m1 = 1.697
        m2 = 1.377
        m3 = 1.1636
        m4 = 1.1636
        m5 = 0.930
        m6 = 0.678
        m7 = 0.678
        m8 = 0.5
        A01 = tranz(l1)@rotx(pi)@rotz(q1)          
        A12 = rotx(pi/2)@tranz(-d1)@trany(-l2)@rotz(q2)
        A23 = rotx(-pi/2)@trany(d2)@tranz(-l3)@rotz(q3)
        A34 = rotx(pi/2)@tranz(-d2)@trany(-l4)@rotz(q4)
        A45 = rotx(-pi/2)@trany(d2)@tranz(-l5)@rotz(q5)
        A56 = rotx(pi/2)@trany(-l6)@rotz(q6)
        A67 = rotx(-pi/2)@tranz(-l7)@rotz(q7)
    
        # A01 = jnp.array([[1., 0., 0., 0.],
        #                 [0.,-1., 0., 0.],
        #                 [0., 0.,-1., 0.1564],
        #                 [0., 0., 0., 1.]])@rotz(q1) 
        
        # A12 = jnp.array([[1., 0., 0., 0.],
        #                 [0., 0.,-1., 0.0054],
        #                 [0., 1., 0.,-0.1284],
        #                 [0., 0., 0., 1.]])@rotz(q2)
        
        # A23 = jnp.array([[1., 0., 0., 0.],
        #                 [0., 0., 1., -0.2104],
        #                 [0.,-1., 0., -0.0064],
        #                 [0., 0., 0., 1.]])@rotz(q3)
        
        # A34 = jnp.array([[1., 0., 0., 0.],
        #                 [0., 0.,-1., 0.0064],
        #                 [0., 1., 0.,-0.2104],
        #                 [0., 0., 0., 1.]])@rotz(q4)
        
        # A45 = jnp.array([[1., 0., 0., 0.],
        #                 [0., 0., 1.,-0.2084],
        #                 [0.,-1., 0.,-0.0064],
        #                 [0., 0., 0., 1.]])@rotz(q5)
        
        # A56 = jnp.array([[1., 0., 0., 0.],
        #                 [0., 0.,-1., 0.],
        #                 [0., 1., 0.,-0.1059],
        #                 [0., 0., 0., 1.]])@rotz(q6)
        
        # A67 = jnp.array([[1., 0., 0., 0.],
        #                 [0., 0., 1.,-0.1059],
        #                 [0.,-1., 0., 0.],
        #                 [0., 0., 0., 1.]])@rotz(q7)
        
        A7E = jnp.array([[1., 0., 0., 0.],
                        [0.,-1., 0., 0.],
                        [0., 0.,-1.,-0.0615],
                        [0., 0., 0., 1.]])

        A02 = A01@A12
        A03 = A02@A23
        A04 = A03@A34
        A05 = A04@A45
        A06 = A05@A56
        A07 = A06@A67
        # A0E = A07@A7E
        # A0G = A0E@AEG

        # print(A01-A01old)
        # print(A12-A12old)
        # print(A23-A23old)
        # print(A34-A34old)
        # print(A45-A45old)
        # print(A56-A56old)
        # print(A67-A67old)
        # print(A7E-A7Eold)

        # c1 = jnp.transpose(jnp.array([-6.48e-4, -1.66e-4, 8.4487e-2]))       #frame 0, link 1
        # c2 = jnp.transpose(jnp.array([-2.3e-5, -1.0364e-2, -7.336e-2]))   #frame 1, link 2
        # c3 = jnp.transpose(jnp.array([-4.4e-5, -9.958e-2, -1.3278e-2]))  #frame 2, link 3
        # c4 = jnp.transpose(jnp.array([-4.4e-5, -6.641e-3, -1.17892e-1]))  #frame 3, link 4
        # c5 = jnp.transpose(jnp.array([-1.8e-5, -7.5478e-2, -1.5006e-2]))  #frame 4, link 5
        # c6 = jnp.transpose(jnp.array([1e-6, -9.432e-3, -6.3883e-2]))  #frame 5, link 6
        # c7 = jnp.transpose(jnp.array([1e-6, -4.5483e-2, -9.650e-3]))  #frame 6, link 7
        # c8 = jnp.transpose(jnp.array([-2.81e-4, -1.1402e-2, -2.9798e-2]))  #frame 7, link E
        # cGripper = jnp.transpose(jnp.array([0.,0.,5.8e-2]))
        ## Mass Matrix
        c1x = -6.48e-4
        c1y = -1.66e-4
        c1z =  8.4487e-2
        
        c2x = -2.3e-5
        c2y = -1.0364e-2
        c2z = -7.336e-2
        
        c3x = -4.4e-5
        c3y = -9.958e-2
        c3z = -1.3278e-2
        
        c4x = -4.4e-5
        c4y = -6.641e-3
        c4z = -1.17892e-1
        
        c5x = -1.8e-5
        c5y = -7.5478e-2
        c5z = -1.5006e-2
        
        c6x = 1e-6
        c6y = -9.432e-3
        c6z = -6.3883e-2
        
        c7x = 1e-6
        c7y = -4.5483e-2
        c7z = -9.650e-3
        
        c8x = -2.81e-4
        c8y = -1.1402e-2
        c8z = -2.9798e-2

        A0c1 = tranx(c1x)@trany(c1y)@tranz(c1z)
        A0c2 = A01@tranx(c2x)@trany(c2y)@tranz(c2z)
        A0c3 = A02@tranx(c3x)@trany(c3y)@tranz(c3z)
        A0c4 = A03@tranx(c4x)@trany(c4y)@tranz(c4z)
        A0c5 = A04@tranx(c5x)@trany(c5y)@tranz(c5z)
        A0c6 = A05@tranx(c6x)@trany(c6y)@tranz(c6z)
        A0c7 = A06@tranx(c7x)@trany(c7y)@tranz(c7z)
        A0c8 = A07@tranx(c8x)@trany(c8y)@tranz(c8z)
        # A0cG = A0E@tranz(cGz)

        #         # Geometric Jacobians
        # R01 = A01[0:3,0:3]     #rotation matrices
        # R12 = A12[0:3,0:3]
        # R23 = A23[0:3,0:3]
        # R34 = A34[0:3,0:3]
        # R45 = A45[0:3,0:3]
        # R56 = A56[0:3,0:3]
        # R67 = A67[0:3,0:3]
        # R7E = A7E[0:3,0:3]

        # r100   = A01[0:3,[3]]
        # r200   = A02[0:3,[3]]
        # r300   = A03[0:3,[3]]
        # r400   = A04[0:3,[3]]
        # r500   = A05[0:3,[3]]
        # r600   = A06[0:3,[3]]
        # r700   = A07[0:3,[3]]
        rc100   = A0c1[0:3,[3]]
        rc200   = A0c2[0:3,[3]]
        rc300   = A0c3[0:3,[3]]
        rc400   = A0c4[0:3,[3]]
        rc500   = A0c5[0:3,[3]]
        rc600   = A0c6[0:3,[3]]
        rc700   = A0c7[0:3,[3]]
        rc800   = A0c8[0:3,[3]]

        # z00 = jnp.array([[0.], [0.], [1.]])
        # z01 = R01@z00
        # z02 = R01@R12@z00
        # z03 = R01@R12@R23@z00
        # z04 = R01@R12@R23@R34@z00
        # z05 = R01@R12@R23@R34@R45@z00
        # z06 = R01@R12@R23@R34@R45@R56@z00
        # z07 = R01@R12@R23@R34@R45@R56@R67@z00
        # # z08 = R01@R12@R23@R34@R45@R56@R67@R7E@z00

        # ske1 = skew(z01)
        # ske2 = skew(z02)
        # ske3 = skew(z03)
        # ske4 = skew(z04)
        # ske5 = skew(z05)
        # ske6 = skew(z06)
        # ske7 = skew(z07)

        # Jc2   = jnp.block([
        #     [ske1@(rc200-r100),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],   #jnp.zeros((3,1))],
        #     [z01,                 jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        #     ])
        # Jc3   = jnp.block([
        #     [ske1@(rc300-r100),  ske2@(rc300-r200),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],  # jnp.zeros((3,1))],
        #     [z01,                z02              ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        #     ])
        # Jc4   = jnp.block([
        #     [ske1@(rc400-r100),  ske2@(rc400-r200),  ske3@(rc400-r300),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))], #  jnp.zeros((3,1))],
        #     [z01,                z02,                 z03             ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]   #jnp.zeros((3,1))]
        #     ])
        # Jc5   = jnp.block([
        #     [ske1@(rc500-r100),  ske2@(rc500-r200),  ske3@(rc500-r300),  ske4@(rc500-r400),   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))],  # jnp.zeros((3,1))],
        #     [z01,                z02,                z03,                z04              ,   jnp.zeros((3,1)),   jnp.zeros((3,1)),   jnp.zeros((3,1))]  # jnp.zeros((3,1))]
        #     ])
        # Jc6   = jnp.block([
        #     [ske1@(rc600-r100),  ske2@(rc600-r200),  ske3@(rc600-r300),  ske4@(rc600-r400),  ske5@(rc600-r500),   jnp.zeros((3,1)),   jnp.zeros((3,1))],   #jnp.zeros((3,1))],
        #     [z01,                z02,                z03,                z04,                z05              ,   jnp.zeros((3,1)),   jnp.zeros((3,1))] #  jnp.zeros((3,1))]
        #     ])
        # Jc7   = jnp.block([
        #     [ske1@(rc700-r100),  ske2@(rc700-r200),  ske3@(rc700-r300),  ske4@(rc700-r400),  ske5@(rc700-r500),  ske6@(rc700-r600),   jnp.zeros((3,1))],  #jnp.zeros((3,1))],
        #     [z01,                z02              ,  z03,                z04,                z05,                z06              ,   jnp.zeros((3,1))]   #jnp.zeros((3,1))]
        #     ])
        # Jc8   = jnp.block([
        #     [ske1@(rc800-r100),  ske2@(rc800-r200),  ske3@(rc800-r300),  ske4@(rc800-r400),  ske5@(rc800-r500),  ske6@(rc800-r600),  ske7@(rc800-r700)], #  jnp.zeros((3,1))],
        #     [z01,                z02,                z03              ,  z04,                z05,                z06,                z07              ]  #,   jnp.zeros((3,1))]
        #     ])


        # g0 = jnp.array([[0],[0],[-g]])
        # tauc2 = jnp.block([[m2*g0],[jnp.zeros((3,1))]])
        # tauc3 = jnp.block([[m3*g0],[jnp.zeros((3,1))]])
        # tauc4 = jnp.block([[m4*g0],[jnp.zeros((3,1))]])
        # tauc5 = jnp.block([[m5*g0],[jnp.zeros((3,1))]])
        # tauc6 = jnp.block([[m6*g0],[jnp.zeros((3,1))]])
        # tauc7 = jnp.block([[m7*g0],[jnp.zeros((3,1))]])
        # tauc8 = jnp.block([[m8*g0],[jnp.zeros((3,1))]])
        # grav      = -jnp.transpose(( + jnp.transpose(tauc2)@Jc2 + jnp.transpose(tauc3)@Jc3 + jnp.transpose(tauc4)@Jc4 + jnp.transpose(tauc5)@Jc5 + jnp.transpose(tauc6)@Jc6 + jnp.transpose(tauc7)@Jc7 + jnp.transpose(tauc8)@Jc8))


        # # Mq = jnp.transpose(holonomicTransform)@Mq7@holonomicTransform  #transform needed to produce Mq_hat
        # gq1 = grav[1,0]
        # gq2 = grav[3,0]
        # gq3 = grav[5,0]
        # return gq1, gq2, gq3
        g0 = jnp.array([[0.],[0.],[-g]])
        gprime = jnp.transpose(g0)

        V = -m1*gprime@rc100-m2*gprime@rc200-m3*gprime@rc300-m4*gprime@rc400-m5*gprime@rc500-m6*gprime@rc600 -m7*gprime@rc700 -m8*gprime@rc800 #-mGripper*gprime@rcG00
        # V = V
        # print('V',V)
        return V.at[0].get()


    def set_initial_position(self):

        print("Starting angular action movement ...")
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""
        
        # Place arm straight up
        for joint_id in range(self.actuator_count):
            # print('jointid',joint_id)
            if (joint_id == 1):
                joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.joint_identifier = joint_id
                joint_angle.value = 45
            else:
                joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
                joint_angle.joint_identifier = joint_id
                joint_angle.value = 0



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
            time.sleep(1.)
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

        # print(action_list)

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
        # self.t = jnp.arange(0,t_end,sampling_time_cyclic)
        # l = jnp.size(self.t)
        self.q_storage = jnp.zeros((3,l))
        self.vel_storage = jnp.zeros((3,l))
        self.userControl = jnp.zeros((3,l))
        self.controlHist = jnp.zeros((3,l))
        self.timeStore = jnp.zeros(l)
        self.hconStore = jnp.zeros(l)

        # trucated_t = jnp.arange(0,(t_end-2.),sampling_time_cyclic)      #this is used as a safety. Last 2 seconds will have 0 torques

        # l_short = jnp.size(trucated_t)

        # print('sampling time', sampling_time_cyclic)
        # print('l',l,l_short)
        


        return

    def InitCyclic(self, sampling_time_cyclic, t_end, print_stats):

        if self.cyclic_running:
            return True

        # Move to Home position first
        if not self.MoveToHomePosition():
            return False

        # Move to initial conditions
        # if not self.set_initial_position():
            # return False

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
            self.base_command.actuators[1].torque_joint = -self.base_feedback.actuators[1].torque
                
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
        hcon_storage_temp = self.hconStore

        #define sinusoide amplitudes, freqs
        amp1 = 5.
        freq1 = 0.2
        amp2 = 5.
        freq2 = 0.2
        amp3 = 0.
        freq3 = 0.5

        
        self.dV_func = jacfwd(self.V)
        
        q_bold1 = self.base_feedback.actuators[0].position
        q_bold2 = self.base_feedback.actuators[2].position
        q_bold3 = self.base_feedback.actuators[4].position
        q_bold4 = self.base_feedback.actuators[6].position

        constants = jnp.array([[q_bold1],[q_bold2],[q_bold3],[q_bold4],])
        s.constants = constants
        # Initial delta between first and last actuator
        # init_delta_position = self.base_feedback.actuators[0].position - self.base_feedback.actuators[self.actuator_count - 1].position

        # Initial first and last actuator torques; avoids unexpected movement due to torque offsets
        init_fourth_torque = self.base_feedback.actuators[3].torque
        init_second_torque = self.base_feedback.actuators[1].torque  
        init_sixth_torque = self.base_feedback.actuators[5].torque

        D_con = jnp.zeros((3,3))

        print('Initial Torque',init_second_torque,init_fourth_torque,init_sixth_torque)

        t_now = time.time()
        t_cyclic = t_now  # cyclic time
        t_stats = t_now  # print  time
        t_init = t_now  # init   time
        no_of_actuators = self.actuator_count
        end_time = self.cyclic_t_end
        # tic = 0
        counter = 0
        Hcon = 0            #ignore for a bit
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
                
                q1 = self.base_feedback.actuators[1].position
                q2 = self.base_feedback.actuators[3].position
                q3 = self.base_feedback.actuators[5].position
                self.base_command.actuators[1].position = q1
                self.base_command.actuators[3].position = q2
                self.base_command.actuators[5].position = q3

                q = (pi/180.)*jnp.array([           #Holonomic constrainted
                                        [q1],
                                        [q2],
                                        [q3]])

                q1dot = self.base_feedback.actuators[1].velocity
                q2dot = self.base_feedback.actuators[3].velocity
                q3dot = self.base_feedback.actuators[5].velocity

                t_elapsed = t_now - t_init
                #get control actions
                timetemp = timetemp.at[counter].set(t_elapsed)  #time since script started

                Mq_hat, Tq, Tqinv = massMatrix_holonomic(q,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
                qdot = jnp.array([[q1dot],[q2dot],[q3dot]])
                p = Tq@qdot
                dMdq = massMatrixJac(q,constants)
                dMdq1, dMdq2, dMdq3 = unravel(dMdq)

                dTqinvdq1 = solve_continuous_lyapunov(Tqinv,dMdq1)
                dTqinvdq2 = solve_continuous_lyapunov(Tqinv,dMdq2)
                dTqinvdq3 = solve_continuous_lyapunov(Tqinv,dMdq3)

                dTqinv_block = jnp.array([dTqinvdq1,dTqinvdq2,dTqinvdq3])
                # dMdq_block = jnp.array([dMdq1, dMdq2, dMdq3])
                dVdq = dV_func(q,constants)

                Cqp = Cqpfunc(p,Tq,dTqinv_block)     #Calculate value of C(q,phat) Matrix.
   
                tau = Tq@dVdq           #tranform into momentum trnasform dynamics
                qddot = dq_d.at[:,[counter]].get()  
                qddotdot = ddq_d.at[:,[counter]].get()  
                p_d = Tqinv@qddot                  #as p0 = Mq*qdot, and p = Tq*p0
                errq = q - q_d.at[:,[counter]].get()
                errp = p -p_d           #change for real or fake error.
                # errp = jnp.zeros((3,1))     ##point stabilisinng for now
                err = jnp.block([[errq],[errp]])    #define error           now running off phat. THis was p_d before to act only on position error.
                v = control(err,Tq,Cqp,Kp,Kd,alpha,Tqinv,dTqinvdq1,dTqinvdq2,dTqinvdq3,qddot,qddotdot,tau,p,p_d)

                # Hcon = 0.5*jnp.transpose(errp.at[:,0].get())@errp.at[:,0].get() + 0.5*jnp.transpose(errq.at[:,0].get() + alpha*errp.at[:,0].get())@Kp@(errq.at[:,0].get() + alpha*errp.at[:,0].get())

                # dTiqdot_dq1 = dTqinvdq1@qddot               #dp_d/dq = dTinv/dq @ qd_dot
                # dTiqdot_dq2 = dTqinvdq2@qddot
                # dTiqdot_dq3 = dTqinvdq3@qddot

                # dp_d_dq = jnp.block([dTiqdot_dq1,dTiqdot_dq2,dTiqdot_dq3])      #this is from product rule

                # v = -(Cqp - D_con)@p_d + dp_d_dq@Tq@p + Tqinv@qddotdot + tau + v_input          #total control law from equaton 17 now here.
                # # print('v',v)

                if (t_elapsed > (end_time - 5)):
                    print('slowing')
                    v = Tq@dVdq         #grav comp only
                    

                u1 = v.at[0,0].get()
                u2 = v.at[1,0].get()
                u3 = v.at[2,0].get() 
                print('u1',u1)
                self.base_command.actuators[1].torque_joint = u1.tolist()
                # Grav comp is sent to fourth actuator
                self.base_command.actuators[3].torque_joint = u2.tolist()
                # Grav comp is sent to sixth actuator
                self.base_command.actuators[5].torque_joint = u3.tolist()
                
                 # Incrementing identifier ensure actuators can reject out of time frames
                self.base_command.frame_id += 1
                if self.base_command.frame_id > 65535:
                    self.base_command.frame_id = 0
                for i in range(no_of_actuators):
                    self.base_command.actuators[i].command_id = self.base_command.frame_id

                # Frame is sent
                try:
                    self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.sendOption)
                except:
                    failed_cyclic_count = failed_cyclic_count + 1
                cyclic_count = cyclic_count + 1

                #store
                controlHist_temp = controlHist_temp.at[:,[counter]].set(jnp.array([[u1],[u2],[u3]]))
                q_storage_temp=q_storage_temp.at[:,[counter]].set(q)
                vel_storage_temp = vel_storage_temp.at[:,[counter]].set(jnp.array([[q1dot],
                                                        [q2dot],
                                                        [q3dot],
                                                        ]))
                hcon_storage_temp = hcon_storage_temp.at[counter].set(Hcon)
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

                #Redefine position commands
                q1 = self.base_feedback.actuators[1].position
                q2 = self.base_feedback.actuators[3].position
                q3 = self.base_feedback.actuators[5].position
                self.base_command.actuators[1].position = q1
                self.base_command.actuators[3].position = q2
                self.base_command.actuators[5].position = q3

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
        l = jnp.size(t)

        qStore = self.q_storage
        tStore = self.timeStore
        vStore = self.vel_storage
        uStore = self.controlHist
        hconStore = self.hconStore
        
        details = ['Saved Data from Physical Implementation! 50HZ CONTROLLER']
        values = ['Amp/Freqs, origon: ',amplitude,frequency,origin]
        header = ['Time', 'State History']
        with open('/root/FYP/Kinova/examples/108-Gen3_torque_control/data/50Hz_Control_test', 'w', newline='') as f:

            writer = csv.writer(f)
            # writer.writerow(simtype)
            writer.writerow(details)
            writer.writerow(values)
            writer.writerow(header)

            # writer.writerow(['Time', t])
            for i in range(l):
                timestamp = tStore.at[i].get()           #time
                q1 = qStore.at[0,i].get()               #postion
                q2 = qStore.at[1,i].get()
                q3 = qStore.at[2,i].get()
                qdot1 = vStore.at[0,i].get()               #FOR MOMENTUM LATER
                qdot2 = vStore.at[1,i].get()
                qdot3 = vStore.at[2,i].get()
                v1 = uStore.at[0,i].get()       #control values
                v2 = uStore.at[1,i].get() 
                v3 = uStore.at[2,i].get() 
                hcon = hconStore.at[i].get()
                data = ['Time:', timestamp  , 'x:   ', q1,q2,q3,qdot1,qdot2,qdot3, v1,v2,v3,hcon]
                
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
    
class self:         #for code
    def __init___(self):
        return 'initialised'
    
class self2:        # for control
    def __init___(self2):
        return 'initialised'
    
s = self2()
s = robotParams(s)

s.constants = jnp.array([[0.],[0.],[0.],[0.],])

holonomicTransform = jnp.array([
        [0.,0.,0.],
        [1.,0.,0.],
        [0.,0.,0.],
        [0.,1.,0.],
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,0.,0.],
    ])


## Define 50Hz Control
endTime = 20        #run for 30 seconds
ControlFreq = 50.       #Hz
dt = 1./ControlFreq
t = jnp.arange(0,endTime,dt)
l = jnp.size(t)
n = 3      #hard code but yolo

alpha = 0.005
Kp = 1.*jnp.eye(n)       #saved tunings
Kd = 0.05*jnp.eye(n)

massMatrixJac = jacfwd(MqPrime)
dV_func = jacfwd(Vq,argnums=0)

q_0 = jnp.zeros((3,1))
#Run through to compile stuff
V_0 = Vq(q_0,s.constants)
Mqh0, Tq0, Tq0inv = massMatrix_holonomic(q_0,s)   #Get Mq, Tq and Tqinv for function to get dTqdq
dMdq0 = massMatrixJac(q_0,s.constants)
dMdq10, dMdq20, dMdq30 = unravel(dMdq0)
dTq0invdq1 = solve_continuous_lyapunov(Tq0inv,dMdq10)
dTq0invdq2 = solve_continuous_lyapunov(Tq0inv,dMdq20)
dTq0invdq3 = solve_continuous_lyapunov(Tq0inv,dMdq30)
dTqinv0 = jnp.array([dTq0invdq1,dTq0invdq2,dTq0invdq3])

##################### TRACKING PROBLEM PARAMETERS
#solve IKM to find q_d.
origin = jnp.array([[-0.134],[1.175]])            #circle origin, or point track. XZ coords.as system is XZ planar
frequency = 0.001
amplitude = 0.1


#    0.134579066939093
#   -0.024600000000000
#    1.175219789049534
traj = 'point'      #Name Trajectory Function
# traj = 'planar_circle'      #Name Trajectory Function

traj_func = getattr(trajectories,traj)
xe = traj_func(t,origin,frequency,amplitude)        #frequency and amplitude aren't used in point tracking , but function asks for these parameters. Probably really messy implementation
# print('xe',xe)
q_init_guess = jnp.zeros(3)         #initialise initial IKM Guess

#TRAJECTORY TRACK
print('Solving IKM....')
q_d = solveIKM(xe,q_init_guess,s)      #solve IKM so trajectory is changed to generalised idsplacement coordinates.
dq_d = diff_finite(q_d,t)           #velocity
ddq_d = diff_finite(dq_d,t)         #acceleration
print('IKM Solved!')

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
            
            args.cyclic_time = dt
            args.duration = endTime         ##Edit this above in the thingo
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
